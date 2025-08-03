
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

(x_train,y_train),(x_test,y_test) = mnist.load_data()
x_train = x_train.reshape(-1,28*28).astype("float32")/255.0
x_test = x_test.reshape(-1,28*28).astype("float32")/255.0

def show_image(image):
    plt.imshow(image.reshape(28, 28), cmap='gray')
    plt.axis('off')
    plt.show()

# Display a sample image from the training data
for i in range(5):

  sample_index = i
  show_image(x_train[sample_index])

print(x_train.shape)
print(y_train.shape)

# sequential API
model = keras.Sequential(
  [
      keras.Input(shape=(28*28)),
      layers.Dense(512,activation = 'relu'),
      layers.Dense(256,activation = 'relu'),
      layers.Dense(10),
  ]
)

model.compile(
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer = keras.optimizers.Adam(learning_rate =0.001),
    metrics = ["accuracy"],
)

model.fit(x_train,y_train,batch_size = 32,epochs = 5,verbose = 1)
model.evaluate(x_test,y_test,batch_size = 32,verbose = 2)

print(model.summary())

# functional api
inputs = keras.Input(shape=(784))
x = layers.Dense(512,activation = 'sigmoid',name = "first_layers")(inputs)
x = layers.Dense(256,activation = 'sigmoid',name = 'second_layers')(x)
outputs = layers.Dense(10,activation = 'softmax')(x)
model = keras.Model(inputs=inputs,outputs = outputs)
model.compile(
      loss = keras.losses.SparseCategoricalCrossentropy(from_logits = False),
      optimizer = keras.optimizers.Adam(learning_rate = 0.01),
      metrics = ["accuracy"],
)

model.fit(x_train,y_train,batch_size = 32,epochs = 5,verbose = 2)
model.evaluate(x_test,y_test,batch_size = 32,verbose = 2)

print(model.summary())

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Define the model architecture
inputs = keras.Input(shape=(784))
x = layers.Dense(512, activation='sigmoid', name="first_layer")(inputs)
x = layers.Dense(256, activation='sigmoid', name='second_layer')(x)
outputs = layers.Dense(10, activation='softmax')(x)
model = keras.Model(inputs=inputs, outputs=outputs)

# List of optimizers
opt = ['GradientDescent', 'Momentum', 'Adagrad', 'RMSprop', 'Adam']

# Dictionary to map optimizer names to their corresponding classes
optimizer_classes = {
    'GradientDescent': tf.keras.optimizers.SGD,
    'Momentum': lambda: tf.keras.optimizers.SGD(momentum=0.9),
    'Adagrad': tf.keras.optimizers.Adagrad,
    'RMSprop': tf.keras.optimizers.RMSprop,
    'Adam': tf.keras.optimizers.Adam
}

# Loop through each optimizer and compile the model
for optimizer_name in opt:
    if optimizer_name == 'Momentum':
        optimizer_instance = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
    else:
        optimizer_class = optimizer_classes[optimizer_name]
        optimizer_instance = optimizer_class(learning_rate=0.01)

    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        optimizer=optimizer_instance,
        metrics=["accuracy"],
    )
    print(f"Compiled model with optimizer: {optimizer_name}")
    model.fit(x_train,y_train,batch_size = 32,epochs = 5,verbose = 2)
    model.evaluate(x_test,y_test,batch_size = 32,verbose = 2)
