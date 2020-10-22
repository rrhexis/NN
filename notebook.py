from PIL import Image, ImageFilter
from matplotlib import gridspec
import numpy as np
import requests
from io import BytesIO
import matplotlib.pyplot as plt
from scipy import signal
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, \
                                    Dense, \
                                    MaxPool2D,\
                                    Dropout, \
                                    Flatten, \
                                    BatchNormalization
from tensorflow.keras.datasets import mnist
import seaborn as sns
from sklearn.metrics import confusion_matrix
import itertools


mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = np.expand_dims(x_train, axis=3)
x_test = np.expand_dims(x_test, axis=3)
input_shape = x_train[0].shape

def display_digit(num):
    label = y_train[num]
    image = x_train[num]
    plt.title('Example: {}  Label: {}'.format(num, label))
    plt.imshow(image, cmap=plt.get_cmap('gray_r'))
    plt.show()

model = Sequential([
  Conv2D(32,kernel_size=(3,3), input_shape=input_shape, activation ='relu', padding='same'),
  MaxPool2D(pool_size=(2, 2)),

	Conv2D(32,kernel_size=(3,3), activation ='relu', padding='same'),
	MaxPool2D(pool_size=(2, 2)),

	Conv2D(32,kernel_size=(3,3), activation ='relu', padding = 'same'),
  MaxPool2D(pool_size=(2, 2)),

	Conv2D(64,kernel_size=(3,3), activation ='relu', padding='same'),
  MaxPool2D(pool_size=(2, 2)),

  Conv2D(64,kernel_size=(3,3), activation ='relu', padding='same'),

	Flatten(),
  Dense(128, activation=tf.nn.leaky_relu),
  Dropout(0.25),
  Dense(10, activation=tf.nn.softmax)])

model.compile(optimizer="adam",
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])
model.summary()

history = model.fit(x_train,
                    y_train,
                    epochs=5,
                    validation_data=(x_test, y_test))



loss, acc = model.evaluate(x_train, y_train)
print("Loss = {}, accuracy = {}".format(loss, acc))
loss, acc = model.evaluate(x_test, y_test)
print("Loss = {}, accuracy = {}".format(loss, acc))

layer_outputs = [layer.output for layer in model.layers[:-4]]
activation_model = tf.keras.Model(inputs = model.input, outputs = layer_outputs)

test_im = x_train[2979] # Random image
activations = activation_model.predict(test_im.reshape(1,28,28,1))
layer_activation = activations[0]

for a in activations: 
  print(a.shape) 


fig, image = plt.subplots(3, 1, figsize=(8, 8))

for k in range(3):
  image[k].imshow(np.vstack((np.hstack([activations[k][0, :, :, i] for i in range(0, 16)]),
                  np.hstack([activations[k][0, :, :, i] for i in range(16, 32)]))))
  image[k].axis('off')
image[0].set_title('conv2D_1')
image[1].set_title('Max_Pool')
image[2].set_title('conv2D_2')

plt.show()
