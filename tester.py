import tensorflow as tf
tf_model_path = 'previous_models/combinedData-modelv3-100epoch-7spe-softmax-imgnet-trainableTrue-topless-avgpooling.h5'
tf_model = tf.keras.models.load_model(tf_model_path)

