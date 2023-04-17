import random
import numpy as np
import tensorflow as tf
from PIL import Image
from keras import Sequential
from keras.layers import Dense
from keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler

seed = 0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
image_size = (224, 224)
class_labels = {
    "bcc": 0,
    "bkl": 1,
    "mel": 2,
    "nv": 3
}

datagen_withaug = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    fill_mode='nearest',
    preprocessing_function=preprocess_input)

datagen_withoutaug = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_generator = datagen_withaug.flow_from_directory(
        '/Users/derke/Documents/GitHub/machine-learning-for-sustainable-development-Orangesaresour/ham_polished/train',
        target_size=image_size,
        batch_size=40,
        class_mode='categorical',
        classes=list(class_labels.keys()))

validation_generator = datagen_withoutaug.flow_from_directory(
        '/Users/derke/Documents/GitHub/machine-learning-for-sustainable-development-Orangesaresour/ham_polished/val',
        target_size=image_size,
        class_mode='categorical',
        classes=list(class_labels.keys()))

num_classes = 4
# resnet_weights_path = '/Users/derke/Desktop/DermTest/resnet101/resnet101_weights_tf_dim_ordering_tf_kernels.h5'
checkpoint_path = "checkpoints_temp/cp-{epoch:04d}.ckpt"
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_best_only=True,
    verbose=1,
    period=5)  

model = Sequential()
model.add(ResNet50(include_top=False, pooling='avg', weights=None)) #try f
model.add(Dense(num_classes, activation='softmax'))

model.layers[0].trainable = True
model.summary()

earlystop_callback = EarlyStopping(monitor='val_loss', verbose = 1, patience=10)

def lr_schedule(epoch):
    lr = 0.1
    if epoch > 40:
        lr /= 5
    if epoch > 70:
        lr /= 5
    if epoch > 110:
        lr /= 5
    return lr

sgd = SGD(learning_rate=0.0, momentum=0.9, decay=0.0, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
lr_scheduler = LearningRateScheduler(lr_schedule)

model.fit(train_generator,
          epochs=150,
          validation_data=validation_generator,
          callbacks=[earlystop_callback, lr_scheduler])

model.save('HAM10000(4)-modelv11-150epoch-softmax-imgnet-trainableTrue-topless-avgpooling.h5')