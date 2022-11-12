"""
Implementing Feedforward neural networks with Keras and TensorFlow
a. Import the necessary packages
b. Load the training and testing data (MNIST/CIFAR10)
c. Define the network architecture using Keras
d. Train the model using SGD
e. Evaluate the network
f. Plot the training loss and accurac
"""
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential

from keras import layers
from keras.layers import Dense
from keras.optimizers import SGD
from keras.datasets import mnist
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import argparse

#ap = argparse.ArgumentParser()
#ap.add_argument("-o", "--output", required=True,help="path to the output loss/accuracy plot")
#args = vars(ap.parse_args())


# grab the MNIST dataset (if this is your first time using this
# dataset then the 11MB download may take a minute)
print("[INFO] accessing MNIST...")
((trainX, trainY), (testX, testY)) = mnist.load_data()
# each image in the MNIST dataset is represented as a 28x28x1
# image, but in order to apply a standard neural network we must
# first "flatten" the image to be simple list of 28x28=784 pixels
trainX = trainX.reshape((trainX.shape[0], 28 * 28 * 1))
testX = testX.reshape((testX.shape[0], 28 * 28 * 1))
# scale data to the range of [0, 1]
trainX = trainX.astype("float32") / 255.0
testX = testX.astype("float32") / 255.0

# convert the labels from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

model = Sequential()
model.add(Dense(256, input_shape=(784,), activation="sigmoid"))
model.add(Dense(128, activation="sigmoid"))
model.add(Dense(10, activation="softmax"))

print("[INFO] training network...")
sgd = SGD(0.01)
model.compile(loss="categorical_crossentropy", optimizer=sgd,metrics=["accuracy"])
#H = model.fit(trainX, trainY, validation_data=(testX, testY),epochs=100, batch_size=128)
m1=model.fit(trainX, trainY, validation_split=0.33,epochs=100, batch_size=128, verbose=0)
print("[INFO] evaluating network...")
loss, acc = model.evaluate(testX, testY, verbose=0)
print('Test Accuracy: %.3f' % acc)

predictions = model.predict(testX, batch_size=128)
print(classification_report(testY.argmax(axis=1),predictions.argmax(axis=1),target_names=[str(x) for x in lb.classes_]))

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(m1.history['accuracy'])
plt.plot(m1.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.savefig('mnist_accuracy.png')

plt.figure()
plt.plot(m1.history['loss'])
plt.plot(m1.history['val_loss'])
# plt.plot(np.arange(0, 100), m1.history["loss"], label="train_loss")
# plt.plot(np.arange(0, 100), m1.history["val_loss"], label="val_loss")
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.savefig('mnist_loss.png')