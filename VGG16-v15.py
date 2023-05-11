import random
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from keras.optimizers import SGD

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from keras.callbacks import LearningRateScheduler

seed = 0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
image_size = (224, 224)
class_labels = {
    "bkl": 0,
    "mel": 1,
    "nv": 2
}
num_classes = len(class_labels)

datagen_withaug = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest',
    preprocessing_function=preprocess_input)

datagen_withoutaug = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_generator = datagen_withaug.flow_from_directory(
        '/Users/derke/Documents/GitHub/machine-learning-for-sustainable-development-Orangesaresour/ham_polished/train',
        target_size=image_size,
        batch_size=32,
        class_mode='categorical',
        classes=list(class_labels.keys()))

validation_generator = datagen_withoutaug.flow_from_directory(
        '/Users/derke/Documents/GitHub/machine-learning-for-sustainable-development-Orangesaresour/ham_polished/val',
        target_size=image_size,
        class_mode='categorical',
        classes=list(class_labels.keys()))


# resnet_weights_path = '/Users/derke/Desktop/DermTest/resnet101/resnet101_weights_tf_dim_ordering_tf_kernels.h5'
checkpoint_path = "checkpoints_temp/cp-{epoch:04d}.ckpt"
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_best_only=True,
    verbose=1,
    period=5)  




def vgg16():
    # Define the input shape
    input_shape = (224, 224, 3)
    input_tensor = Input(shape=input_shape)

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(input_tensor)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # Classification block
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense(num_classes, activation='softmax', name='predictions')(x)

    # Create the model
    model = Model(inputs=input_tensor, outputs=x, name='vgg16')
    return model



model = vgg16(num_classes)
earlystop_callback = EarlyStopping(monitor='val_loss', verbose = 2, patience=10)


model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(
        train_generator,
        epochs=100,
        validation_data=validation_generator,
        callbacks=[earlystop_callback]
        )

model.save('HAM10000(3)-vgg16-v1.h5')