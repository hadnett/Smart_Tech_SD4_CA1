import os
import cv2
import json
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
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

TRAINING_FOLDER = "/Volumes/HADNETT/4th_Year/Smart Tech/CA1_Data/bdd100k/images/100k/train"
VALIDATION_FOLDER = "/Volumes/HADNETT/4th_Year/Smart Tech/CA1_Data/bdd100k/images/100k/val"
EXTRACTION_PATH = "/Volumes/HADNETT/4th_Year/Smart Tech/CA1_Data/extracted_images/"
PREPROCESSING_PATH = "/Volumes/HADNETT/4th_Year/Smart Tech/CA1_Data/preprocessed/"

STORAGE_PATH = "/Volumes/HADNETT/4th_Year/Smart Tech/CA1_Data/extracted_images/"

DESIRED_IMAGE_SIZE = 50

# 'with' statement automatically handles closing of file in Python 2.5 or higher.
with open('/Volumes/HADNETT/4th_Year/Smart Tech/CA1_Data/labels/det_20/det_train.json') as f:
    training_attributes = json.load(f)

with open('/Volumes/HADNETT/4th_Year/Smart Tech/CA1_Data/labels/det_20/det_val.json') as f:
    validation_attributes = json.load(f)


def validate_image_size(image_size):
    return image_size[0] != 0 and image_size[1] != 0


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
                if not z['category'].lower().strip() == "drivable area" and not z['category'].lower().strip() == "lane":
                    box = (z['box2d']['x1'], z['box2d']['y1'], z['box2d']['x2'], z['box2d']['y2'])
                    cropped_image = image.crop(box)
                    if validate_image_size(cropped_image.size):
                        if not os.path.exists(STORAGE_PATH + data_type + '/' + z['category']):
                            os.makedirs(STORAGE_PATH + data_type + '/' + z['category'])
                        cropped_image.save(STORAGE_PATH + data_type + '/' + z['category'] + '/' + str(z['id']) + '.jpg')
        else:
            count += 1
    print(count)


def process_image(image):
    # Convert Image to GreyScale
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
        return new_img
    return None


def preprocessing_extracted_images(data_type, source):
    count = 0
    if not os.path.exists(PREPROCESSING_PATH + data_type):
        os.makedirs(PREPROCESSING_PATH + data_type)
    else:
        raise OSError("Directory already exists please rename file before running again.")

    main_directory = "/Volumes/HADNETT/4th_Year/Smart Tech/CA1_Data/extracted_images/" + source
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


try:
    # Training Set Lost 20 Images
    load_images_from_file("training_extracted", TRAINING_FOLDER, training_attributes)
    # Validation Set Lost 4 Images
    load_images_from_file("validation_extracted", VALIDATION_FOLDER, validation_attributes)
    print("Extraction Complete!")
    # Note files had to be deleted to run these functions as the new json attribute file
    # contains additional class categories that we do not require for this assignment
    # (other person, other vehicle & trailer).
    preprocessing_extracted_images("training_processed1", "training_extracted")
    preprocessing_extracted_images("validation_processed1", "validation_extracted")
    print("Preprocessing Complete!")
except OSError as e:
    print(e)


def letnet_model():
    model = Sequential()
    # Output from this convolution layer is 30 24x24 feature matrices
    model.add(Conv2D(15, (5, 5), input_shape=(50, 50, 1), activation='relu'))
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


# Output car image to understand the effects of preprocessing.
f = plt.figure()
f.add_subplot(1, 2, 1)
image_right = np.asarray(
    Image.open("/Volumes/HADNETT/4th_Year/Smart Tech/CA1_Data/extracted_images/training_extracted/car/4.jpg"))
plt.imshow(image_right)
f.add_subplot(1, 2, 2)
image_left = np.asarray(
    Image.open("/Volumes/HADNETT/4th_Year/Smart Tech/CA1_Data/preprocessed/training_processed/car/4.jpg"))
plt.imshow(image_left)
plt.show(block=True)

class_model = letnet_model()
print(class_model.summary())

data_generator = ImageDataGenerator()
# prepare an iterators for each dataset
train_it = data_generator.flow_from_directory(os.path.join(PREPROCESSING_PATH, 'training_processed'),
                                              color_mode="grayscale", target_size=(50, 50), batch_size=100,
                                              class_mode='categorical')
val_it = data_generator.flow_from_directory(os.path.join(PREPROCESSING_PATH, 'validation_processed'),
                                            color_mode="grayscale", target_size=(50, 50), batch_size=100,
                                            class_mode='categorical')

X, y = train_it.next()
print('Batch shape=%s, min=%.3f, max=%.3f' % (X.shape, X.min(), X.max()))

history = class_model.fit(train_it,
                          epochs=5,
                          verbose=1,
                          validation_data=val_it)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, loss, 'r', "Training Loss")
plt.plot(epochs, val_loss, 'b', "Validation Loss")
plt.figure()
