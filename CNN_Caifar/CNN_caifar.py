# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10

(x_train,y_train),(x_test,y_test) = cifar10.load_data()

import matplotlib.pyplot as plt

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Function to plot images
def plot_images(images, labels, class_names):
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i])
        plt.xlabel(class_names[labels[i][0]])
    plt.show()

# Plot the first 25 images with their labels
plot_images(x_train, y_train, class_names)

x_train = x_train.astype("float32")/255.0
x_test = x_test.astype("float32")/255.0
# sequential api for caifan data sets
model = keras.Sequential(
    [
        keras.Input(shape=(32,32,3)),
        layers.Conv2D(32,3,padding='valid',activation='relu'),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Conv2D(64,3,activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128,3,activation='relu'),
        layers.Flatten(),
        layers.Dense(64,activation='relu'),
        layers.Dense(10),
    ]
)
model.compile(
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits = True),
    optimizer = keras.optimizers.Adam(learning_rate=3e-4),
    metrics = ['accuracy'],
)

model.fit(x_train,y_train,batch_size=64,epochs = 10,verbose = 2)
model.evaluate(x_test,y_test,batch_size = 64,verbose= 2)
# Funcional API for caifan data set 
def my_model():
  inputs = keras.Input(shape=(32,32,3))
  x = layers.Conv2D(32,3)(inputs)
  x = layers.BatchNormalization()(x)
  x = keras.activations.relu(x)
  x = layers.MaxPooling2D()(x)
  x = layers.Conv2D(64,5,padding = 'same')(x)
  x = layers.BatchNormalization()(x)
  x = keras.activations.relu(x)
  x = layers.Conv2D(128,3)(x)
  x = layers.BatchNormalization()(x)
  x = keras.activations.relu(x)
  x = layers.Flatten()(x)
  x = layers.Dense(64,activation = 'relu')(x)
  outputs = layers.Dense(10)(x)
  return keras.Model(inputs=inputs,outputs = outputs)

model = my_model()
model.compile(
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits = True),
    optimizer = keras.optimizers.Adam(learning_rate=3e-4),
    metrics = ['accuracy'],
)
model.fit(x_train,y_train,batch_size = 64,epochs = 10,verbose = 2)
model.evaluate(x_test,y_test,batch_size = 64,verbose =2)

model.summary()

