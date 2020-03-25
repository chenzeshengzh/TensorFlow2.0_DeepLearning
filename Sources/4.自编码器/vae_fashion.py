import os

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow import keras
from tensorflow.keras import layers, datasets, optimizers, losses

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
gpu = tf.config.experimental.list_physical_devices('GPU')
assert len(gpu) > 0
tf.config.experimental.set_memory_growth(gpu[0], True)

(x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data()
x_train = tf.cast(x_train, dtype=tf.float32) / 255.
x_test = tf.cast(x_test, dtype=tf.float32) / 255.
train_db = tf.data.Dataset.from_tensor_slices(x_train).shuffle(1000).batch(512)
test_db = tf.data.Dataset.from_tensor_slices(x_test).batch(512)


def save_image(imgs, name):
    new_image = Image.new('L', (200, 200))
    index = 0
    for i in range(0, 200, 20):
        for j in range(0, 200, 20):
            im = imgs[index]
            im = Image.fromarray(im, 'L')
            new_image.paste(im, (i, j))
            index += 1
    new_image.save(name)


class VAE(keras.Model):
    def __init__(self, units):
        super(VAE, self).__init__()
        self.fc1 = layers.Dense(128)
        self.fc2 = layers.Dense(units)
        self.fc3 = layers.Dense(units)
        self.fc4 = layers.Dense(128)
        self.fc5 = layers.Dense(784)

    def reparameterize(self, mu, log_var):
        eps = tf.random.normal(log_var.shape)
        std = tf.exp(log_var) ** 0.5
        z = mu + std * eps
        return z

    def encoder(self, x):
        h = tf.nn.relu(self.fc1(x))
        mu = self.fc2(h)
        log_var = self.fc3(h)
        return mu, log_var

    def decoder(self, z):
        out = tf.nn.relu(self.fc4(z))
        out = self.fc5(out)
        return out

    def call(self, inputs, training=None):
        mu, log_var = self.encoder(inputs)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decoder(z)
        return x_hat, mu, log_var


def main():
    model = VAE(64)
    optimizer = optimizers.Adam(lr=1e-3)
    for epoch in range(100):
        for step, x in enumerate(train_db):
            x = tf.reshape(x, (-1, 784))
            with tf.GradientTape() as tape:
                x_rec_logits, mu, log_var = model(x)
                rec_loss = tf.nn.sigmoid_cross_entropy_with_logits(x, x_rec_logits)
                # rec_loss = losses.binary_crossentropy(x, x_rec_logits, from_logits=True)
                rec_loss = tf.reduce_mean(rec_loss)
                kl_div = -0.5 * (log_var + 1 - mu ** 2 - tf.exp(log_var))
                kl_div = tf.reduce_mean(kl_div)
                loss = rec_loss + kl_div * 1.
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
        print(f'Epoch:{epoch}, Loss:{loss.numpy()}')

        # rec
        x = next(iter(test_db))
        logits, _, _ = model(tf.reshape(x, (-1, 784)))
        x_hat = tf.sigmoid(logits)
        x_hat = tf.reshape(x_hat, (-1, 28, 28))
        x_concat = tf.concat([x[:50], x_hat[:50]], axis=0)
        x_concat = x_concat.numpy() * 255.
        x_concat = x_concat.astype(np.uint8)
        save_image(x_concat, f'rec/epoch_{epoch}.png')

        # vae
        z = tf.random.normal((512, 64))
        logits = model.decoder(z)
        x_hat = tf.sigmoid(logits)
        x_hat = tf.reshape(x_hat, (-1, 28, 28))
        x_hat = x_hat.numpy() * 255.
        x_hat = x_hat.astype(np.uint8)
        save_image(x_hat, f'ori/epoch_{epoch}.png')


if __name__ == '__main__':
    main()
