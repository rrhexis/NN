import tensorflow as tf
mnist = tf.keras.datasets.mnist

import matplotlib.pyplot as plt

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(100, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.0),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=3, validation_data = (x_test, y_test))

loss, acc = model.evaluate(x_test, y_test)
print("Loss = {}, accuracy = {}".format(loss, acc))

loss, acc = model.evaluate(x_train, y_train)
print("Loss = {}, accuracy = {}".format(loss, acc))

plt.figure(figsize=(10,7))
plt.xlabel('Epochs')
plt.ylabel("Sparse_categorical_crossentropy")
val = plt.plot(history.epoch, history.history['accuracy'],
                   '--', label='Val', color = 'blue')

plt.legend()
plt.xlim([0,max(history.epoch)])
plt.show()
