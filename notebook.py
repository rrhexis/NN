import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train.shape

#overfit model
model_overfit = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dropout(0.0),
  tf.keras.layers.Dense(200, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.0),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model_overfit.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])
history_overfit = model_overfit.fit(x_train, y_train, epochs=1, validation_data=(x_test, y_test))

loss_overfit_test, acc_overfit_test = model_overfit.evaluate(x_test, y_test)
print("Loss = {}, accuracy = {}".format(loss_overfit_test, acc_overfit_test))

loss_overfit, acc_overfit = model_overfit.evaluate(x_train, y_train)
print("Loss = {}, accuracy = {}".format(loss_overfit, acc_overfit))

#standart model
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(80, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.1),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=1, validation_data=(x_test, y_test))

loss_test, acc_test = model.evaluate(x_test, y_test)
print("Loss = {}, accuracy = {}".format(loss_test, acc_test))

loss, acc = model.evaluate(x_train, y_train)
print("Loss = {}, accuracy = {}".format(loss, acc))

plt.figure(figsize=(10,7))
plt.xlabel('Epochs')
plt.ylabel("Sparse_categorical_crossentropy")
val_overfit = plt.plot(history.epoch, history.history['val_'+'loss'],
                   '--', label='Val standart')
val = plt.plot(history_overfit.epoch, history_overfit.history['val_'+'loss'],
                   '--', label='Val overfit')
plt.plot(history.epoch, history.history["loss"],
             label='Train standart')
plt.plot(history_overfit.epoch, history_overfit.history["loss"],
             label='Train overfit')
plt.legend()
plt.xlim([0,max(history.epoch)])
plt.show()


print("loss(overfit) / loss(standart) = ", loss_overfit_test / loss_test)

