# Smart Technologies - CA1 - Convolutional neural network project for the classification of vehicles and other objects that may be found when driving. # 

## Project Team: William Hadnett & Aaron Reihill ##

#### Requirements: ####

* OpenCV (pip install opencv-python)
* PIL/Pillow (pip install Pillow)
* MatPlotLib (pip install matplotlib)
* Numpy (pip install numpy)
* Tensorflow (pip install tensorflow)
* Scipy (pip install scipy)  
* split-folders (pip install split-folders)

#### Change Paths ####
Once ready to run make sure to change the file paths to match your computer,
the variables needed to be changed are located at the top of the file.
* TRAINING_FOLDER 
* VALIDATION_FOLDER
* EXTRACTION_PATH
* PREPROCESSING_PATH
* LABELS_PATH
* STORAGE_PATH 
* TEST_FOLDER 

#### Running Prediction ####
To run prediction you need to insert a  url to the desired image you would like to predict with
this model. 

Line 379 - img = cv2.imread('INSERT IMAGE URL FOR PREDICTION')
