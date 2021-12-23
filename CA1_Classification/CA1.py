import os
import cv2
import json
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import splitfolders
import shutil
import random
# Scipy required from ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.models import Model
from keras.models import load_model

TRAINING_FOLDER = "G:/bdd100k/images/100k/train"
VALIDATION_FOLDER = "G:/bdd100k/images/100k/val"
EXTRACTION_PATH = "G:/bdd100k/extracted_images/"
PREPROCESSING_PATH = "G:/bdd100k/preprocessed/"
LABELS_PATH = "G:/bdd100k/labels/"
STORAGE_PATH = "G:/bdd100k/extracted_images/"
TEST_FOLDER = "G:/bdd100k/preprocessed/test_processed/"

DESIRED_IMAGE_SIZE = 50
IGNORE_CLASSES = {"other vehicle", "other person", "trailer", "lane", "drivable area"}

# 'with' statement automatically handles closing of file in Python 2.5 or higher.
with open(LABELS_PATH + 'det_20/det_train.json') as f:
    training_attributes = json.load(f)

with open(LABELS_PATH + 'det_20/det_val.json') as f:
    validation_attributes = json.load(f)

"""
Checks to ensure if the images x or y coordinate is not 0 
:param image_size The image size to be checked
"""


def validate_image_size(image_size):
    return image_size[0] != 0 and image_size[1] != 0


"""
Counts the number of images per class
:param json The json containing labels to be counted
"""


def count_classes_amounts(json):
    class_amounts = {}
    for item in json:
        if "labels" in item:
            for image in item['labels']:
                category = image['category']
                if category not in IGNORE_CLASSES:
                    if category in class_amounts:
                        class_amounts[category] += 1
                    else:
                        class_amounts[category] = 1
    return class_amounts


"""
Counts the total number of images is a list - designed to work with count_classes amounts
:param class_list The class list to be totalled
"""


def get_total_images(class_list):
    total = 0
    for i in class_list:
        total += class_list[i]
    return total


"""
Plots the number of items/images in a class
:param class_amounts List of the total number of items/images per class
"""


def plot_class_amounts(class_amounts):
    names = list(class_amounts.keys())
    values = list(class_amounts.values())
    plt.figure(figsize=(15, 3))
    plt.bar(range(len(class_amounts)), values, tick_label=names, align='edge', width=0.3)
    plt.show()


'''
load_images_from_file extracts the images required to train, validate and test the model. It stores these images
in a master directory containing sub directories labelled after the image category (car, person, plane).
:param data_type Training, test or validation data
:param folder The path to the folder the data is stored in.
:param json_attributes The json attributes required to extract the images.
'''


def load_images_from_file(data_type, folder, json_attributes):
    count = 0
    if not os.path.exists(STORAGE_PATH + data_type):
        os.makedirs(STORAGE_PATH + data_type)
    else:
        raise OSError("Directory already exists please rename file before running again.")

    for i in json_attributes:
        image = Image.open(os.path.join(folder, i['name'].strip()))
        if 'labels' in i:
            for z in i['labels']:
                if z['category'].lower().strip() not in IGNORE_CLASSES:
                    box = (z['box2d']['x1'], z['box2d']['y1'], z['box2d']['x2'], z['box2d']['y2'])
                    cropped_image = image.crop(box)
                    if validate_image_size(cropped_image.size):
                        if not os.path.exists(STORAGE_PATH + data_type + '/' + z['category']):
                            os.makedirs(STORAGE_PATH + data_type + '/' + z['category'])
                        cropped_image.save(STORAGE_PATH + data_type + '/' + z['category'] + '/' + str(z['id']) + '.jpg')
        else:
            count += 1
    print(count)


"""
Equalises an images histogram
:param img The image to equalise
"""


def equalise(img):
    img = cv2.equalizeHist(img)
    return img


"""
Pre-processes an image correcting shape, aspect ratio, colour, padding and normalisation
:param image The image to be pre-processed
"""


def process_image(image):
    # Convert Image to GreyScale
    if image.shape[0] >= DESIRED_IMAGE_SIZE and image.shape[1] >= DESIRED_IMAGE_SIZE:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        old_size = gray.shape[:2]

        # Set ratio of the image.
        ratio = float(DESIRED_IMAGE_SIZE) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])
        if new_size[0] != 0 and new_size[1] != 0:
            image = cv2.resize(gray, (new_size[1], new_size[0]))

            # Smooth the image, using GaussianBlur
            blur = cv2.GaussianBlur(image, (3, 3), 0)

            # Add black padding to image
            delta_w = DESIRED_IMAGE_SIZE - new_size[1]
            delta_h = DESIRED_IMAGE_SIZE - new_size[0]
            top, bottom = delta_h // 2, delta_h - (delta_h // 2)
            left, right = delta_w // 2, delta_w - (delta_w // 2)
            color = [0, 0, 0]
            new_img = cv2.copyMakeBorder(blur, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
            new_img = equalise(new_img)
            new_img = new_img / 255
            return new_img
    return None


"""
Extracts images from directory for pre-processing, passes image to process_image and saves the image to pre-process
directory. 
:param data_type The type of data being processed training, validation or test
:param source The location of the data to be processed.
"""


def preprocessing_extracted_images(data_type, source):
    count = 0
    if not os.path.exists(PREPROCESSING_PATH + data_type):
        os.makedirs(PREPROCESSING_PATH + data_type)
    else:
        raise OSError("Directory already exists please rename file before running again.")

    main_directory = EXTRACTION_PATH + source
    for subdir, dirs, files in os.walk(main_directory):
        for file in files:
            if file.endswith(".jpg"):
                print(os.path.join(subdir, file))
                class_name = os.path.basename(subdir)
                image = process_image(cv2.imread(os.path.join(subdir, file)))
                if image is not None:
                    if not os.path.exists(PREPROCESSING_PATH + data_type + '/' + class_name):
                        os.makedirs(PREPROCESSING_PATH + data_type + '/' + class_name)
                    cv2.imwrite(PREPROCESSING_PATH + data_type + '/' + class_name + '/' + file, image)
                else:
                    count += 1
    print(count)


"""
Uses the split-folders library to split folders on the disk in half - generating validation and test datasets
"""


def split_data():
    splitfolders.ratio(PREPROCESSING_PATH + "validation_processed", output=PREPROCESSING_PATH + "validation",
                       seed=1337, ratio=(.5, .5), group_prefix=None)
    shutil.move(PREPROCESSING_PATH + "validation/val", PREPROCESSING_PATH)
    shutil.move(PREPROCESSING_PATH + "validation/train", PREPROCESSING_PATH)

    shutil.rmtree(PREPROCESSING_PATH + "validation_processed")
    shutil.rmtree(PREPROCESSING_PATH + "validation")

    os.rename(PREPROCESSING_PATH + "val", PREPROCESSING_PATH + "validation_processed")
    os.rename(PREPROCESSING_PATH + "train", PREPROCESSING_PATH + "test_processed")


"""
Randomly deletes images from directories. https://stackoverflow.com/a/67962542
:param source The path to the source directory
:param The label of the directory to delete
:param n Number of samples to be deleted
"""


def undersampling(source, label, n):
    path = PREPROCESSING_PATH + source + '/' + label
    n = n
    img_names = os.listdir(path)
    img_names = random.sample(img_names, n)
    for image in img_names:
        print(image)
        f = os.path.join(path, image)
        os.remove(f)


def letnet_model():
    model = Sequential()
    # Output from this convolution layer is 30 24x24 feature matrices
    model.add(Conv2D(15, (5, 5), input_shape=(DESIRED_IMAGE_SIZE, DESIRED_IMAGE_SIZE, 1), activation='relu'))
    # Output will be  30 12x12 matrices
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Output will be 15 10x10 matrices
    model.add(Conv2D(15, (3, 3), activation='relu'))
    # Output will be 15 5x5
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(Adam(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def model_v2():
    model = Sequential()
    model.add(Conv2D(60, (5, 5), input_shape=(DESIRED_IMAGE_SIZE, DESIRED_IMAGE_SIZE, 1), activation='relu'))
    model.add(Conv2D(60, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(30, (3, 3), activation='relu'))
    model.add(Conv2D(30, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    model.compile(Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def model_v3():
    model = Sequential()
    model.add(Conv2D(60, (5, 5), input_shape=(DESIRED_IMAGE_SIZE, DESIRED_IMAGE_SIZE, 1), activation='relu'))
    model.add(Conv2D(60, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(30, (3, 3), activation='relu'))
    model.add(Conv2D(30, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    model.compile(Adam(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def model_v4():
    model = Sequential()
    model.add(Conv2D(60, (5, 5), input_shape=(DESIRED_IMAGE_SIZE, DESIRED_IMAGE_SIZE, 1), activation='relu'))
    model.add(Conv2D(60, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(30, (3, 3), activation='relu'))
    model.add(Conv2D(30, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    model.compile(Adam(learning_rate=0.1), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def model_v5():
    model = Sequential()
    model.add(Conv2D(100, (5, 5), input_shape=(DESIRED_IMAGE_SIZE, DESIRED_IMAGE_SIZE, 1), activation='relu'))
    model.add(Conv2D(100, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(50, (3, 3), activation='relu'))
    model.add(Conv2D(50, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    model.compile(Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model



try:
    # Training Set Lost 20 Images
    load_images_from_file("training_extracted", TRAINING_FOLDER, training_attributes)
    # Validation Set Lost 4 Images
    load_images_from_file("validation_extracted", VALIDATION_FOLDER, validation_attributes)
    print("Extraction Complete!")
    # Note files had to be deleted to run these functions as the new json attribute file
    # contains additional class categories that we do not require for this assignment
    # (other person, other vehicle & trailer).
    preprocessing_extracted_images("training_processed", "training_extracted")
    preprocessing_extracted_images("validation_processed", "validation_extracted")
    print("Preprocessing Complete!")
except OSError as e:
    print(e)

training_attributes_class_amounts = count_classes_amounts(training_attributes)
validation_attributes_class_amounts = count_classes_amounts(validation_attributes)

plot_class_amounts(training_attributes_class_amounts)
plot_class_amounts(validation_attributes_class_amounts)


split_data()

undersampling('training_processed', 'car', 1)

undersampling('validation_processed', 'car', 1)

undersampling('test_processed', 'car', 1)

path = PREPROCESSING_PATH + 'test_processed/car'
img_names = os.listdir(path)
print(len(img_names), " size of car test set")

path = PREPROCESSING_PATH + 'test_processed/traffic sign'
img_names = os.listdir(path)
print(len(img_names), " size of car traffic sign set")


# Total Training Dataset: 1273707
total_training = get_total_images(training_attributes_class_amounts)
# Total Validation Dataset: 185945
total_validation = get_total_images(validation_attributes_class_amounts)
print("Total Training: " + str(total_training) + " Total Validation: " + str(total_validation))

# Output car image to understand the effects of preprocessing.
f = plt.figure()
f.add_subplot(1, 2, 1)
image_right = np.asarray(
    Image.open(EXTRACTION_PATH + "training_extracted/car/5.jpg"))
plt.imshow(image_right)
f.add_subplot(1, 2, 2)
image_left = np.asarray(
    Image.open(PREPROCESSING_PATH + "training_processed/car/5.jpg"))
plt.imshow(image_left)
plt.show(block=True)


class_model = letnet_model()
print(class_model.summary())
#
data_generator = ImageDataGenerator()
# # prepare an iterators for each dataset
train_it = data_generator.flow_from_directory(os.path.join(PREPROCESSING_PATH, 'training_processed'),
                                              color_mode="grayscale", target_size=(DESIRED_IMAGE_SIZE,
                                                                                   DESIRED_IMAGE_SIZE),
                                              batch_size=200, class_mode='categorical')
val_it = data_generator.flow_from_directory(os.path.join(PREPROCESSING_PATH, 'validation_processed'),
                                            color_mode="grayscale", target_size=(DESIRED_IMAGE_SIZE, DESIRED_IMAGE_SIZE), batch_size=200,
                                            class_mode='categorical')

X, y = train_it.next()
print('Batch shape=%s, min=%.3f, max=%.3f' % (X.shape, X.min(), X.max()))

history = class_model.fit(train_it,
                          epochs=3,
                          verbose=1,
                          validation_data=val_it)

plt.subplot(2, 1, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper right')

plt.subplot(2, 1, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='lower right')

plt.show()
class_model.save('model_v5.h5')


final_model = load_model('model_v5.h5')
test_it = data_generator.flow_from_directory(TEST_FOLDER,
                                             color_mode="grayscale", target_size=(DESIRED_IMAGE_SIZE,
                                                                                  DESIRED_IMAGE_SIZE), batch_size=200,
                                             class_mode='categorical')

results = final_model.evaluate(test_it, verbose=1)
print("test loss, test acc:", results)
print("An image below DESIRED_IMAGE_SIZE in pixels will not be accepted, by the processed image function")

img = cv2.imread('INSERT IMAGE URL FOR PREDICTION')
img = process_image(img)
img = img.reshape(1, DESIRED_IMAGE_SIZE, DESIRED_IMAGE_SIZE, 1)
print(img.shape)

prediction = final_model.predict(img)
print("Predicted sign: " + str(np.argmax(final_model.predict(img), axis=1)))
print(train_it.class_indices)
