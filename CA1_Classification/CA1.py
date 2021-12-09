import os
import cv2
import json

training_folder = "/Volumes/HADNETT/4th_Year/Smart Tech/CA1_Data/bdd100k/images/100k/train"
validation_folder = "/Volumes/HADNETT/4th_Year/Smart Tech/CA1_Data/bdd100k/images/100k/val"

# 'with' statement automatically handles closing of file in Python 2.5 or higher.
with open('/Users/williamhadnett/Documents/4th_Year/Smart Tech/Labels_test.txt') as f:
    training_attributes = json.load(f)

'''
load_images_from_file extracts the images required to train, validate and test the model. It stores these images
in a master directory containing sub directories labelled after the image category (car, person, plane). 

:param data_type Training, test or validation data
:param folder The path to the folder the data is stored in.
:param json_attributes The json attributes required to extract the images. 
'''


def load_images_from_file(data_type, folder, json_attributes):

    if not os.path.exists(data_type):
        os.makedirs(data_type)
    else:
        raise OSError("Directory already exists please rename file before running again.")

    for i in json_attributes:
        image = cv2.imread(os.path.join(folder, i['name']))
        for z in i['labels']:
            if not z['category'] == "drivable area" and not z['category'] == "lane":
                y1 = int(z['box2d']['y1'])
                y2 = int(z['box2d']['y2'])
                x1 = int(z['box2d']['x1'])
                x2 = int(z['box2d']['x2'])
                cropped_image = image[y1:y2, x1:x2]
                if not os.path.exists(data_type + '/' + z['category']):
                    os.makedirs(data_type + '/' + z['category'])
                cv2.imwrite(data_type + '/' + z['category'] + '/' + str(z['id']) + '.jpg', cropped_image)


try:
    load_images_from_file("training_extracted", training_folder, training_attributes)
except OSError as e:
    print(e)
