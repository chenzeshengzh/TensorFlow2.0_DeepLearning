import os

import matplotlib
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras import layers, Sequential, losses, datasets, optimizers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
matplotlib.rcParams['font.size'] = 18
matplotlib.rcParams['figure.titlesize'] = 18
matplotlib.rcParams['figure.figsize'] = [9, 7]
matplotlib.rcParams['font.family'] = ['STKaiti']
matplotlib.rcParams['axes.unicode_minus'] = False
gpu = tf.config.experimental.list_physical_devices('GPU')
assert len(gpu) > 0, 'NO GPU'
tf.config.experimental.set_memory_growth(gpu[0], True)


def main():
    loss_with_epochs = []
    batch_size = 512
    epochs = 30

    # get mnist dataset
    (x, y), (x_test, y_test) = datasets.mnist.load_data()
    x = 2 * tf.convert_to_tensor(x, dtype=tf.float32) / 255. - 1
    y = tf.convert_to_tensor(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=10)
    train_db = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(1000).batch(batch_size)

    # build model
    model = Sequential([
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(10)
    ])
    optimizer = optimizers.Adam(lr=1e-3)

    # start training
    for epoch in range(epochs):
        for step, (x, y) in enumerate(train_db):
            with tf.GradientTape() as tape:
                x = tf.reshape(x, (-1, 28 * 28))
                out = model(x)
                loss = tf.reduce_mean(losses.mse(out, y), axis=0)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            if step % 100 == 0:
                print(epoch, step, 'loss:', float(loss))
        loss_with_epochs.append(loss.numpy())

    # test
    x = 2 * tf.convert_to_tensor(x_test, dtype=tf.float32) / 255. - 1
    y = tf.convert_to_tensor(y_test, dtype=tf.int32)
    y = tf.one_hot(y, depth=10)
    out = model(tf.reshape(x, (-1, 28 * 28)))
    correct = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(y, axis=1), tf.argmax(out, axis=1)), dtype=tf.int32))
    print('acc:', correct.numpy() / y_test.shape[0])

    # plot loss along epochs
    plt.figure()
    plt.plot(range(len(loss_with_epochs)), loss_with_epochs, label='训练误差')
    plt.plot(range(len(loss_with_epochs)), loss_with_epochs, 's')
    plt.legend()
    plt.title('MSE - Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.show()


if __name__ == '__main__':
    main()
