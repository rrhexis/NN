import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train.shape

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

display_digit(20, x_train, y_train)

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(100, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=50, validation_data=(x_test, y_test))

plt.figure(figsize=(16,10))
plt.xlabel('Epochs')
plt.ylabel("Sparse_categorical_crossentropy")
val = plt.plot(history.epoch, history.history['val_'+'loss'],
                   '--', label='Val')
plt.plot(history.epoch, history.history["loss"], color=val[0].get_color(),
             label='Train')
plt.legend()
plt.xlim([0,max(history.epoch)])
plt.show()

for i in range(10):
  display_digit(i,x_test,y_test,model(x_test[i:i+1,:,:])[0])

model.fit(x_train, y_train, epochs=3)

loss, acc = model.evaluate(x_test, y_test)
print("Loss = {}, accuracy = {}".format(loss, acc))

loss, acc = model.evaluate(x_train, y_train)
print("Loss = {}, accuracy = {}".format(loss, acc))

predictions = model.predict(x_test[0:1,:,:])
print(predictions)
print(y_test[0])