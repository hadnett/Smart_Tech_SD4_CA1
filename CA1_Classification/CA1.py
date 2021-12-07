import os
import cv2
import json

training_folder = "/Volumes/HADNETT/4th_Year/Smart Tech/CA1_Data/bdd100k/images/100k/train"

with open('/Users/williamhadnett/Documents/4th_Year/Smart Tech/Labels_test.txt') as f:
    data = json.load(f)

for i in data:
    image = cv2.imread(os.path.join(training_folder, i['name']))
    for z in i['labels']:
        if not z['category'] == "drivable area" and not z['category'] == "lane":
            y1 = int(z['box2d']['y1'])
            y2 = int(z['box2d']['y2'])
            x1 = int(z['box2d']['x1'])
            x2 = int(z['box2d']['x2'])
            cropped_image = image[y1:y2, x1:x2]
            if not os.path.exists(z['category']):
                os.makedirs(z['category'])
            cv2.imwrite(z['category'] + '/' + str(z['id']) + '.jpg', cropped_image)
