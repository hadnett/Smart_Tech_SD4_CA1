import os
import cv2
import json
from PIL import Image

training_folder = "/Volumes/HADNETT/4th_Year/Smart Tech/CA1_Data/bdd100k/images/100k/train"
validation_folder = "/Volumes/HADNETT/4th_Year/Smart Tech/CA1_Data/bdd100k/images/100k/val"

storage_path = "/Volumes/HADNETT/4th_Year/Smart Tech/CA1_Data/extracted_images/"

# 'with' statement automatically handles closing of file in Python 2.5 or higher.
with open('/Volumes/HADNETT/4th_Year/Smart Tech/CA1_Data/labels/bdd100k_labels_images_train.json') as f:
    training_attributes = json.load(f)

with open('/Volumes/HADNETT/4th_Year/Smart Tech/CA1_Data/labels/bdd100k_labels_images_val.json') as f:
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

    if not os.path.exists(storage_path + data_type):
        os.makedirs(storage_path + data_type)
    else:
        raise OSError("Directory already exists please rename file before running again.")

    for i in json_attributes:
        image = Image.open(os.path.join(folder, i['name'].strip()))
        for z in i['labels']:
            if not z['category'].lower().strip() == "drivable area" and not z['category'].lower().strip() == "lane":
                box = (z['box2d']['x1'], z['box2d']['y1'], z['box2d']['x2'], z['box2d']['y2'])
                cropped_image = image.crop(box)
                if validate_image_size(cropped_image.size):
                    if not os.path.exists(storage_path + data_type + '/' + z['category']):
                        os.makedirs(storage_path + data_type + '/' + z['category'])
                    cropped_image.save(storage_path + data_type + '/' + z['category'] + '/' + str(z['id']) + '.jpg')


try:
    # Training Set Lost 20 Images
    load_images_from_file("training_extracted", training_folder, training_attributes)
    # Validation Set Lost 4 Images
    load_images_from_file("validation_extracted", validation_folder, validation_attributes)
except OSError as e:
    print(e)
