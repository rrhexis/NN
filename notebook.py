import tensorflow as tf
import numpy as np
mnist = tf.keras.datasets.mnist

import matplotlib.pyplot as plt

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

print(x_train.shape)
print(y_train.shape)

x_train.resize(x_train.shape[0], x_train.shape[1] * x_train.shape[2] + 1)
for i in range(len(y_train)):
  x_train[i][-1] = y_train[i]

x_test.resize(x_test.shape[0], x_test.shape[1] * x_test.shape[2] + 1)
for i in range(len(y_test)):
  x_test[i][-1] = y_test[i]



model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(785, )),
  tf.keras.layers.Dense(200, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.0),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10)

loss, acc = model.evaluate(x_test, y_test)
print("Loss = {}, accuracy = {}".format(loss, acc))

loss, acc = model.evaluate(x_train, y_train)
print("Loss = {}, accuracy = {}".format(loss, acc))





