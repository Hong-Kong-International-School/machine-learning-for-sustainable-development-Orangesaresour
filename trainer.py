import random
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

seed = 0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
image_size = (224, 224)

data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = data_generator.flow_from_directory(
        '/Users/derke/Documents/GitHub/machine-learning-for-sustainable-development-Orangesaresour/DataCombined/train',
        target_size=image_size,
        batch_size=120,
        class_mode='categorical')

validation_generator = data_generator.flow_from_directory(
        '/Users/derke/Documents/GitHub/machine-learning-for-sustainable-development-Orangesaresour/DataCombined/test',
        target_size=image_size,
        batch_size=50,
        class_mode='categorical')

num_classes = 3
# resnet_weights_path = '/Users/derke/Desktop/DermTest/resnet101/resnet101_weights_tf_dim_ordering_tf_kernels.h5'

checkpoint_path = "checkpoints_temp/cp-{epoch:04d}.ckpt"
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    verbose=1,
    period=5)  

model = Sequential()
model.add(ResNet50(include_top=False, pooling='avg', weights='imagenet')) #try f
model.add(Dense(num_classes, activation='softmax'))

model.layers[0].trainable = True
model.summary()
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
earlystop_callback = EarlyStopping(monitor='val_loss', verbose = 1, patience=5)

model.fit(
        train_generator,
        steps_per_epoch=7,
        epochs=100,
        validation_data=validation_generator,
        callbacks=[checkpoint_callback, earlystop_callback]
        )

model.save('combinedData-modelv1-100epoch-7spe-softmax-imgnet-trainableTrue-topless-avgpooling.h5')