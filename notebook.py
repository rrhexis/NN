import tensorflow as tf
import numpy as np
mnist = tf.keras.datasets.mnist

import matplotlib.pyplot as plt

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

def display_digit(num, x, y, vector = None):
    label = y[num]
    image = x[num]
    if vector is None:
        plt.title('Example: {}  Label: {}'.format(num, label))
        plt.imshow(image, cmap=plt.get_cmap('gray_r'))
    else:
        plt.figure(figsize=(6,3))
        plt.subplot(1,2,1)
        plt.title('Real label: {}'.format(label))
        plt.imshow(image, cmap=plt.get_cmap('gray_r'))
        plt.subplot(1,2,2)
        thisplot = plt.bar(range(10), vector, color="#777777")
        plt.ylim([0, 1]) 
        plt.xticks([])
        plt.yticks([])
        predicted_label = np.argmax(vector)
        thisplot[predicted_label].set_color('red')
        plt.title('Predicted label: {}'.format(predicted_label))
    plt.show()


model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(200, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.1),
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


trash = (np.asarray(np.where(y_test != np.argmax(model.predict(x_test), axis = 1)))).ravel()
print(trash)
print(len(trash))
#plt.hist(trash, edgecolor = 'black')
#plt.show()

for i in range(1):
  display_digit(trash[i], x_test, y_test, model(x_test[trash[i]:trash[i]+1,:,:])[0])

