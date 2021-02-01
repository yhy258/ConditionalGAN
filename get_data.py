import numpy as np
import tensorflow as tf
from Config import *
from keras import datasets
from keras.utils import to_categorical


IMG_SHAPE = model_config.imag_shape
BATCH_SIZE = train_config.BATCH_SIZE
BUFFER_SIZE = 60000
noise_dim = model_config.z_dim

(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
# *tuple -> 소괄호 제외한 것 사용할 때|
train_images = train_images.reshape(train_images.shape[0], *IMG_SHAPE).astype('float32')
train_images = (train_images - 127.5) / 127.5

test_images = test_images.reshape(test_images.shape[0], *IMG_SHAPE).astype('float32')
test_images = (test_images - 127.5) / 127.5




train_images_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
train_labels_dataset = tf.data.Dataset.from_tensor_slices(train_labels).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)