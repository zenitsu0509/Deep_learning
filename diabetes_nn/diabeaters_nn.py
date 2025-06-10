import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("/content/diabetes.csv")
print(df.head())

X = df.drop(['Outcome'],axis = 1)
y = df['Outcome']

scaler = StandardScaler()
X = scaler.fit_transform(X)

inputs = keras.Input(shape = (X.shape[1],))
# x = layers.BatchNormalization()(inputs)
x = layers.Dense(64,activation = 'relu')(inputs)
x = layers.BatchNormalization()(x)
x = layers.Dense(128,activation = 'relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(128,activation = 'relu')(x)
x = layers.BatchNormalization()(x)
outputs = layers.Dense(2,activation = 'softmax')(x)
model = keras.Model(inputs = inputs,outputs = outputs)
model.compile(
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits = False),
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics = ["accuracy"],
)

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = .25,random_state = 123)

model.fit(x_train,y_train,batch_size = 32,epochs = 10,verbose = 2)
model.evaluate(x_test,y_test,batch_size = 32,verbose  =2)

from keras.utils import plot_model
plot_model(model)

"""**Functional API**"""

inputs = keras.Input(shape=(X.shape[1],))

# Model architecture
x = layers.Dense(64, activation='relu')(inputs)
x = layers.BatchNormalization()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.BatchNormalization()(x)

# Output layer for regression
outputs = layers.Dense(1, activation='linear')(x)

model = keras.Model(inputs=inputs, outputs=outputs)

# Compile the model with an appropriate loss function for regression
model.compile(
    loss = 'binary_crossentropy',
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics = ["accuracy"],
)

# Training the model
model.fit(x_train, y_train, epochs=10, validation_split=0.2, batch_size=32)

# Evaluate the model
model.evaluate(x_test, y_test,batch_size = 32,verbose = 2)

data = pd.read_csv('/content/diabetes.csv')
data.head()
data.info()
data.describe()

"""**Sequential API**"""

model = Sequential()
model.add(Dense(64, input_dim=x_train.shape[1], activation='relu'))
model.add(Dense(32, activation='leaky_relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, epochs=50, batch_size=32, validation_data=(x_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)

print(f'Test Accuracy: {accuracy:.4f}')

# Plot training history
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()
