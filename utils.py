import tensorflow as tf
from tensorflow import keras
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

# Load and compile Keras model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])


def showDataDist(y):
    ax = sns.countplot(x=y)
    ax.bar_label(ax.containers[0])
    ax.set(title="Data distribution on Client 1")
    plt.show()

def getData(dist, x, y):
    dx = []
    dy = []
    counts = [0 for i in range(10)]
    for i in range(len(x)):
        if counts[y[i]]<dist[y[i]]:
            dx.append(x[i])
            dy.append(y[i])
            counts[y[i]] += 1
        
    return np.array(dx), np.array(dy)


def getMnistDataSet():
    # Load dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train[..., np.newaxis]/255.0, x_test[..., np.newaxis]/255.0

    return x_train, y_train, x_train, y_train

def genOutDir():
    if not os.path.exists('out'):
        os.mkdir('out')

def plotServerData(data):
    df = pd.DataFrame(data)
    # plt.plot(df['loss'],color = 'b', label = 'loss')
    plt.plot(df['accuracy'],color = 'g', label = 'accuracy')
    plt.legend(loc = 'lower right')
    plt.xlabel('Rounds')
    plt.ylabel('accuracy')
    plt.show()

def plotClientData(data):
    df = pd.DataFrame(data)
    plt.plot(df['accuracy'],color = 'b', label = 'training accuracy')
    plt.plot(df['val_accuracy'],color = 'g', label = 'validation accuracy')
    plt.legend(loc = 'lower right')
    plt.xlabel('Rounds')
    plt.ylabel('accuracy')
    plt.show()