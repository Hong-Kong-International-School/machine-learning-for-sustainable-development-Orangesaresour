import tensorflow as tf
import numpy as np
from PIL import Image
# from tensorflow.keras.applications.resnet50 import preprocess_input
# Load the saved model
model = tf.keras.models.load_model('previous_models/HAM10000(4)-modelv9-100epoch-softmax-imgnet-trainableTrue-topless-avgpooling.h5')
class_labels = {
    0: "bcc",
    1: "bkl",
    2: "mel",
    3: "nv"
}
# Load the test image
# test_image = Image.open('DataCombined/test/Melanoma/atypical-nevi-dermoscopy-100.jpg')
test_image = Image.open('dataverse_files/ham_organized/bcc/ISIC_0025260.jpg')
test_array = np.array(test_image)
# Preprocess the test image
preprocessed_test_image = tf.keras.applications.resnet50.preprocess_input(test_array)

# Generate predictions for the test image
predictions = model.predict(np.array([preprocessed_test_image]))
primary_pred_idx = predictions.argmax()
print(predictions)
print(class_labels[primary_pred_idx])
# Save the predictions to a file
# np.savetxt('predictions.txt', predictions)


# import os
# import glob
# import cv2
# import numpy as np

# # Path to the folder containing the images
# folder_path = "/DataCombined/test/Eczema"

# # List all image files in the folder
# image_files = glob.glob(os.path.join(folder_path, "*.jpg"))

# # Load each image and preprocess it for your model
# images = []
# for file in image_files:
#     # Load the image using OpenCV
#     img = cv2.imread(file)
#     # Preprocess the image (e.g. resize, normalize, etc.)
#     img = preprocess_input(img)
#     # Add the preprocessed image to the list of images
#     images.append(img)

# # Convert the list of images to a numpy array
# images = np.array(images)

# # Use the model to predict the classes of the images
# predictions = model.predict(images)

# # Print the predicted class for each image
# for i, file in enumerate(image_files):
#     print("Image {}: {}".format(file, predictions[i]))