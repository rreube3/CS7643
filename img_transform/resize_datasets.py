# Resize and crop images from Original_Datasets google drive folder
import os
import cv2

# get current working directory (script should be in same location as original dataset folder)
cwd = os.getcwd()

# resize and crop images in the original dataset directory
original_directory = cwd + '\\Original_Datasets'
# loop through datasets
for sub_dataset in os.listdir(original_directory):
    # do not augment images in DRIVE (these already have proper dimensions 565x584)
    if sub_dataset != 'DRIVE':
        # loop through folders in CHASE and STARE datasets
        for sub_folder in os.listdir(original_directory + '\\' + sub_dataset):
            sub_directory = original_directory + '\\' + sub_dataset + '\\' + sub_folder
            # loop through individual images, resize, and save to Resized_Datasets directory
            for img_file in os.listdir(sub_directory):
                img_raw = cv2.imread(sub_directory + '\\' + img_file)
                height, width = 584, 565
                img_resize = cv2.resize(img_raw, (width, height))
                cv2.imwrite(cwd + '\\Resized_Datasets\\' + sub_dataset + '\\' + sub_folder + '\\' + img_file, img_resize)