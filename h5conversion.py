import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications import ResNet50

# num_classes = 5
# graph = tf.Graph()
# model = Sequential()
# model.add(ResNet50(include_top=False, pooling='avg', weights='imagenet')) 
# model.add(Dense(num_classes, activation='softmax'))
# model.layers[0].trainable = True

# model.load_weights('checkpoints_temp/cp-0055.ckpt')
# model.summary()


# model.save('combinedData(5)-modelv2-55epoch-7spe-softmax-imgnet-trainableTrue-topless-avgpooling.h5')
# prediction = model.predict('eczema-arms-6.jpg')
# print(prediction)




# def h5_to_coreml():
#     # Load the TensorFlow model
#     tf_model_path = 'previous_models/combinedData-modelv3-100epoch-7spe-softmax-imgnet-trainableTrue-topless-avgpooling.h5'
#     tf_model = tf.keras.models.load_model(tf_model_path)
#     output_layer_names = [layer.name for layer in tf_model.outputs]

#     # Convert the TensorFlow model to a Core ML model
#     coreml_model = tfcoreml.convert(tf_model_path,
#                                     input_name_shape_dict={'input': (None, 224, 224, 3)},
#                                     output_feature_names=output_layer_names)

#     # Save the Core ML model to a file
#     coreml_model_path = 'combinedData-modelv3-100epoch-7spe-softmax-imgnet-trainableTrue-topless-avgpooling.mlmodel'
#     coremltools.utils.save_spec(coreml_model.get_spec(), coreml_model_path)