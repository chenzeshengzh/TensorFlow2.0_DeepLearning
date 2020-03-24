import os
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
gpu = tf.config.experimental.list_physical_devices('GPU')
assert len(gpu) > 0
tf.config.experimental.set_memory_growth(gpu[0], True)
matplotlib.rcParams['font.size'] = 18
matplotlib.rcParams['figure.figsize'] = [9, 7]
matplotlib.rcParams['figure.titlesize'] = 18
matplotlib.rcParams['font.family'] = ['STKaiti']
matplotlib.rcParams['axes.unicode_minus'] = False


def himmelblau(x):
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2


def plot_surface():
    x = np.arange(-6, 6, 0.1)
    y = np.arange(-6, 6, 0.1)
    X, Y = np.meshgrid(x, y)
    Z = himmelblau([X, Y])

    fig = plt.figure('himmelblau')
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z)
    ax.view_init(60, -30)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()


def gradient_fun():
    x = tf.constant([4., 0.])
    for step in range(200):
        with tf.GradientTape() as tape:
            tape.watch([x])
            y = himmelblau(x)
        grads = tape.gradient(y, [x])[0]
        x -= 0.01 * grads
        if step % 20 == 19:
            print(f'step = {step}, x = {x.numpy()}, y = {y.numpy()}')


if __name__ == '__main__':
    plot_surface()
    gradient_fun()
