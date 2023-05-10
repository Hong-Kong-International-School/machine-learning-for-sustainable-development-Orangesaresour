# ############################################################################
# Reorganize data into subfolders based on their labels
# then use keras flow_from_dir or pytorch ImageFolder to read images with 
# folder names as labels

#Sort images to subfolders first 
# import pandas as pd
# import os
# import shutil

# # Dump all images into a folder and specify the path:
# data_dir = os.getcwd() + "/dataverse_files/all_images/"

# # Path to destination directory where we want subfolders
# dest_dir = os.getcwd() + "/dataverse_files/ham_organized/"

# # Read the csv file containing image names and corresponding labels
# skin_df2 = pd.read_csv('dataverse_files/HAM10000_metadata')
# print(skin_df2['dx'].value_counts())

# label=skin_df2['dx'].unique().tolist()  #Extract labels into a list
# label_images = []


# # Copy images to new folders
# for i in label:
#     os.mkdir(dest_dir + str(i) + "/")
#     sample = skin_df2[skin_df2['dx'] == i]['image_id']
#     label_images.extend(sample)
#     for id in label_images:
#         shutil.copyfile((data_dir + "/"+ id +".jpg"), (dest_dir + i + "/"+id+".jpg"))
#     label_images=[]    

#Now we are ready to work with images in subfolders



#convert subfolders to train/test
"""pip install split-folders
"""

import splitfolders
input_folder = 'dataverse_files/ham_organized/'

# import os
# # assign directory
 
# # iterate over files in
# # that directory
# for filename in os.listdir(input_folders):
#     if not filename.startswith('.'):
#         print(filename)
        
splitfolders.ratio(input_folder, output="data_processed", seed=42, ratio=(0.8,0.2),group_prefix=None)
        
