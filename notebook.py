from PIL import Image, ImageFilter
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

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(100, activation=tf.nn.leaky_relu),
  tf.keras.layers.Dropout(0.0),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=5, validation_data = (x_test, y_test))

loss, acc = model.evaluate(x_test, y_test)
print("Loss = {}, accuracy = {}".format(loss, acc))

loss, acc = model.evaluate(x_train, y_train)
print("Loss = {}, accuracy = {}".format(loss, acc))

x_train = np.expand_dims(x_train, axis=3)
x_test = np.expand_dims(x_test, axis=3)
input_shape = x_train[0].shape

model2 = Sequential([
  Conv2D(32,kernel_size=(3,3), input_shape=input_shape, activation ='relu', padding='same'),
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
model2.compile(optimizer="adam",
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])
model2.summary()

history2 = model2.fit(x_train,
                    y_train,
                    epochs=2,
                    validation_data=(x_test, y_test))

loss, acc = model2.evaluate(x_train, y_train)
print("Loss = {}, accuracy = {}".format(loss, acc))
loss, acc = model2.evaluate(x_test, y_test)
print("Loss = {}, accuracy = {}".format(loss, acc))


plt.figure(figsize=(10,7))
plt.xlabel('Epochs')
plt.ylabel("Accuracy")
val = plt.plot(history.epoch, history.history['accuracy'], label = 'линейная', color = 'blue')
val2 = plt.plot(history2.epoch, history2.history['accuracy'], label = 'сверточная', color = 'green')
plt.title('accutacy(epoch)', loc = 'center')

plt.legend()
plt.xlim([0,max(history.epoch)])
plt.show()
