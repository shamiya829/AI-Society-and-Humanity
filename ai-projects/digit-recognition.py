# NeuralNine

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# import data 
mnist = tf.keras.datasets.mnist

# load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# specify train and test data
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# modify model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

# compile & fit model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metric=['accuracy'])
model.fit(x_train,y_train, epochs=3)

# save/train model so there's no need to retrain
model.save('digit.recog.model')

# test model
model = tf.keras.models.load_model('model')

loss, accuracy = model.evaluate(x_test, y_test)

print(loss)
print(accuracy)

