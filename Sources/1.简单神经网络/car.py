import os
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, Sequential, losses, optimizers
import matplotlib
from matplotlib import pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
gpu = tf.config.experimental.list_physical_devices('GPU')
assert len(gpu) > 0, "NO GPU"
tf.config.experimental.set_memory_growth(gpu[0], True)
matplotlib.rcParams['font.size'] = 18
matplotlib.rcParams['figure.figsize'] = [9, 7]
matplotlib.rcParams['figure.titlesize'] = 18
matplotlib.rcParams['font.family'] = ['STKaiti']
matplotlib.rcParams['axes.unicode_minus'] = False


def norm(x, mean, std):
    return (x - mean) / std


class NetWork(keras.Model):
    def __init__(self):
        super(NetWork, self).__init__()
        self.fc1 = layers.Dense(64, activation='relu')
        self.fc2 = layers.Dense(64, activation='relu')
        self.fc3 = layers.Dense(1)

    def call(self, inputs, training=None):
        x = self.fc1(inputs)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


def main():
    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
    dataset_path = keras.utils.get_file("auto-mpg.data", url)
    column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model Year', 'Origin']
    dataset = pd.read_csv(dataset_path, names=column_names, sep=' ', skipinitialspace=True, comment='\t', na_values='?')
    dataset = dataset.dropna()

    origin = dataset.pop('Origin')
    dataset['USA'] = (origin == 1) * 1.
    dataset['Europe'] = (origin == 2) * 1.
    dataset['Japan'] = (origin == 3) * 1.

    train = dataset.sample(frac=0.8, random_state=12)
    test = dataset.drop(train.index)

    train_labels = train.pop('MPG')
    test_labels = test.pop('MPG')

    mean = np.mean(np.array(train), axis=0)
    std = np.std(np.array(train), axis=0)

    train_db = tf.data.Dataset.from_tensor_slices((norm(train.values, mean, std), train_labels.values)).shuffle(1000).batch(32)

    model = NetWork()
    optimizer = optimizers.Adam(lr=1e-3)

    train_loss = []
    test_loss = []

    for epoch in range(200):
        for step, (x, y) in enumerate(train_db):
            with tf.GradientTape() as tape:
                out = model(x)
                loss = tf.reduce_mean(losses.mse(y, out))
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if step % 10 == 0:
                print(epoch, step, float(loss))

        train_loss.append(float(loss))
        out = model(norm(test.values, mean, std))
        test_loss.append(float(tf.reduce_mean(losses.mse(out, test_labels.values))))

    plt.figure()
    plt.plot(range(len(train_loss)), train_loss, color='C0', label='train loss')
    plt.plot(range(len(train_loss)), train_loss, color='C0', marker='s')
    plt.plot(range(len(test_loss)), test_loss, color='C1', label='test loss')
    plt.plot(range(len(test_loss)), test_loss, color='C1', marker='s')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title('MSE - Epoch')
    plt.show()


if __name__ == '__main__':
    main()
