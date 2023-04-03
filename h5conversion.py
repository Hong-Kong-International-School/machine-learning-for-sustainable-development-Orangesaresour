import tensorflow as tf
import tfcoreml
import coremltools

# Load the TensorFlow model
tf_model_path = 'previous_models/combinedData-modelv3-100epoch-7spe-softmax-imgnet-trainableTrue-topless-avgpooling.h5'
tf_model = tf.keras.models.load_model(tf_model_path)
output_layer_names = [layer.name for layer in tf_model.outputs]

# Convert the TensorFlow model to a Core ML model
coreml_model = tfcoreml.convert(tf_model_path,
                                input_name_shape_dict={'input': (None, 224, 224, 3)},
                                output_feature_names=output_layer_names)

# Save the Core ML model to a file
coreml_model_path = 'combinedData-modelv3-100epoch-7spe-softmax-imgnet-trainableTrue-topless-avgpooling.mlmodel'
coremltools.utils.save_spec(coreml_model.get_spec(), coreml_model_path)