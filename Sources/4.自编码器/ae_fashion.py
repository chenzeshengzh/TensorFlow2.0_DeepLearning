import os

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow import keras
from tensorflow.keras import layers, Sequential, losses, optimizers, datasets

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
gpu = tf.config.experimental.list_physical_devices('GPU')
assert len(gpu) > 0
tf.config.experimental.set_memory_growth(gpu[0], True)

(x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data()
x_train = tf.cast(x_train, dtype=tf.float32) / 255.
x_test = tf.cast(x_test, dtype=tf.float32) / 255.
train_db = tf.data.Dataset.from_tensor_slices(x_train).shuffle(1000).batch(256)
test_db = tf.data.Dataset.from_tensor_slices(x_test).batch(256)


def save_images(imgs, name):
    new_image = Image.new('L', (200, 200))
    index = 0
    for i in range(0, 200, 20):
        for j in range(0, 200, 20):
            im = imgs[index]
            im = Image.fromarray(im, mode='L')
            new_image.paste(im, (i, j))
            index += 1
    new_image.save(name)


class AE(keras.Model):
    def __init__(self):
        super(AE, self).__init__()
        self.encoder = Sequential([
            layers.Dense(256, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(64)
        ])
        self.decoder = Sequential([
            layers.Dense(128, activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Dense(784)
        ])

    def call(self, inputs, training=None):
        x = self.encoder(inputs)
        x = self.decoder(x)
        return x


def main():
    model = AE()
    optimizer = optimizers.Adam(lr=1e-3)
    for epoch in range(100):
        for step, x in enumerate(train_db):
            inputs = tf.reshape(x, [-1, 784])
            with tf.GradientTape() as tape:
                logits = model(inputs)
                loss = tf.reduce_mean(losses.binary_crossentropy(inputs, logits, from_logits=True))
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
        print(f'Epoch:{epoch}, Loss:{loss.numpy()}')
        x = next(iter(test_db))
        logits = model(tf.reshape(x, [-1, 784]))
        x_hat = tf.sigmoid(logits)
        x_hat = tf.reshape(x_hat, [-1, 28, 28])
        x_concat = tf.concat([x[:50], x_hat[:50]], axis=0)
        x_concat = x_concat.numpy() * 255.
        x_concat = x_concat.astype(dtype=np.uint8)
        save_images(x_concat, f'ae_images/rec_epoch_{epoch}.png')


if __name__ == '__main__':
    main()
