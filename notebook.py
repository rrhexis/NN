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

"""history = model.fit(x_train,
                    y_train,
                    epochs=1,
                    validation_data=(x_test, y_test))



loss, acc = model.evaluate(x_train, y_train)
print("Loss = {}, accuracy = {}".format(loss, acc))
loss, acc = model.evaluate(x_test, y_test)
print("Loss = {}, accuracy = {}".format(loss, acc)) """

layer_outputs = [layer.output for layer in model.layers[:-4]]
activation_model = tf.keras.Model(inputs = model.input, outputs = layer_outputs)

test_im = x_train[2979] # Random image
activations = activation_model.predict(test_im.reshape(1,28,28,1))
layer_activation = activations[0]

for a in activations: 
  print(a.shape) 

"""w=10
h=10
fig=plt.figure(figsize=(10, 10))
columns = 8
rows = 4
for i in range(1, columns*rows + 1):
    img = activations[0][0, :, :, i - 1]
    #fig.add_subplot(rows, columns, i)
    axs = plt.subplots(rows, columns)
    plt.subplots_adjust(wspace=0.0, hspace=0.0)
    print(type(axs))
    print(type(i//columns - 1))
    axs[(i//columns - 1), i%columns].imshow(img)
    axs.set_cmap('hot')
    axs.axis('off')top=1.-0.5/(k*nrow+1), bottom=0.5/(nrow+1), 
         left=0.5/(ncol+1), right=1-0.5/(ncol+1)

plt.show()"""


ncol = 8
nrow = 4
fig = plt.figure(figsize=(4*nrow + 1, ncol + 1)) 
gs1 = gridspec.GridSpec(4*nrow, ncol, width_ratios=[1, 1, 1, 1, 1, 1, 1, 1],
         wspace=0.0, hspace=0.0, top=0.95, bottom=0.05, left=0.17, right=0.845)




for i in range(1, ncol*nrow + 1):

    if i == 4 : 
      string1 = 'conv_2D_1',
      string2 = 'max_pool',
      string3 = 'conv_2D_2', 
    else:
      string1 = string2 = string3 = None

    im = activations[0][0, :, :, i - 1]
    ax1 = plt.subplot(gs1[((i-1)//ncol), (i-1)%(ncol)])
    ax1.axis('off')
    res1 = ax1.imshow(im)
    ax1.set_title(string1)
    im2 = activations[1][0, :, :, i - 1]
    ax2 = plt.subplot(gs1[((i-1)//ncol)+5, (i-1)%(ncol)], title = string2)
    ax2.axis('off')
    res2 = ax2.imshow(im2)
    im3 = activations[2][0, :, :, i - 1]
    ax3 = plt.subplot(gs1[((i-1)//ncol)+10, (i-1)%(ncol)],  title = string3)
    ax3.axis('off')
    res3 = ax3.imshow(im3)


plt.show()
