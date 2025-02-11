# Import Library
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras import Model
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt

# Load MNIST Dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data(path='mnist.npz')

# Normalize Input Data (Min:0, Max:1)
x_train, x_test = x_train / 255.0, x_test / 255.0

# Convert Labels to oneHot (10 Classes for 0 to 9 Numbers)
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Define Neural Network Layers
input_layer = Input(shape=(28, 28))
flatten = Flatten()(input_layer)
h1 = Dense(64, activation='sigmoid', use_bias=True)(flatten)
h2 = Dense(64, activation='sigmoid', use_bias=True)(h1)
output_layer = Dense(10, activation='sigmoid', use_bias=True)(h2)

# Define Neural Network Model
model = Model(input_layer, output_layer)

# Get Model Summary
model.summary()

# Compile Model
model.compile(optimizer=SGD(learning_rate=0.01), loss='mse', metrics=['accuracy'])

# Train Model
result = model.fit(x_train, y_train, epochs=300, validation_data=(x_test, y_test))

# Plot Training Progress
plt.plot(result.history['accuracy'], label='Accuracy')
plt.plot(result.history['val_accuracy'], label='Validation Accuracy')
plt.plot(result.history['loss'], label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend()
plt.show()
