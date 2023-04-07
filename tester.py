import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.resnet import preprocess_input
# Load the saved model
model = tf.keras.models.load_model('combinedData(5)-modelv5-55epoch-1010steps-softmax-imgnet-trainableTrue-topless-avgpooling.h5')

# Load the test image
test_image = Image.open('Eczema_Photos/eczema-fingertips-29.jpg')
test_array = np.array(test_image)
# Preprocess the test image
preprocessed_test_image = preprocess_input(test_array)

# Generate predictions for the test image
predictions = model.predict(np.array([preprocessed_test_image]))
print(predictions)
# Save the predictions to a file
# np.savetxt('predictions.txt', predictions)