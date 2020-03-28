import os
import glob

import multiprocessing
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow import keras
from tensorflow.keras import layers, Sequential, optimizers, losses

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
gpu = tf.config.experimental.list_physical_devices('GPU')
assert len(gpu) > 0
tf.config.experimental.set_memory_growth(gpu[0], True)


def make_anime_dataset(img_paths,
                       batch_size,
                       resize=64,
                       drop_remainder=True,
                       shuffle=True,
                       repeat=1):
    @tf.function
    def _map_fn(img):
        img = tf.image.resize(img, [resize, resize])
        img = tf.clip_by_value(img, 0, 255)
        img = img / 127.5 - 1
        return img
    dataset = disk_image_batch_dataset(img_paths,
                                       batch_size,
                                       drop_remainder=drop_remainder,
                                       map_fn=_map_fn,
                                       shuffle=shuffle,
                                       repeat=repeat)
    img_shape = (resize, resize, 3)
    len_dataset = len(img_paths) // batch_size
    return dataset, img_shape, len_dataset


def disk_image_batch_dataset(img_paths,
                             batch_size,
                             labels=None,
                             drop_remainder=True,
                             n_prefetch_batch=1,
                             filter_fn=None,
                             map_fn=None,
                             n_map_threads=None,
                             filter_after_map=False,
                             shuffle=True,
                             shuffle_buffer_size=None,
                             repeat=None):
    if labels is None:
        memory_data = img_paths
    else:
        memory_data = (img_paths, labels)

    def parse_fn(path, *label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        return (img,) + label

    if map_fn:
        def map_fn_(*args):
            return map_fn(*parse_fn(*args))
    else:
        map_fn_ = parse_fn
    dataset = memory_data_batch_dataset(memory_data,
                                        batch_size,
                                        drop_remainder,
                                        n_prefetch_batch,
                                        filter_fn,
                                        map_fn_,
                                        n_map_threads,
                                        filter_after_map,
                                        shuffle,
                                        shuffle_buffer_size,
                                        repeat)
    return dataset


def memory_data_batch_dataset(memory_data,
                              batch_size,
                              drop_remainder=True,
                              n_prefetch_batch=1,
                              filter_fn=None,
                              map_fn=None,
                              n_map_threads=None,
                              filter_after_map=False,
                              shuffle=True,
                              shuffle_buffer_size=None,
                              repeat=None):
    dataset = tf.data.Dataset.from_tensor_slices(memory_data)
    dataset = batch_dataset(dataset,
                            batch_size,
                            drop_remainder,
                            shuffle,
                            repeat,
                            n_prefetch_batch,
                            filter_fn,
                            map_fn,
                            n_map_threads,
                            filter_after_map,
                            shuffle_buffer_size)
    return dataset


def batch_dataset(dataset,
                  batch_size,
                  drop_remainder=True,
                  shuffle=True,
                  repeat=None,
                  n_prefetch_batch=1,
                  filter_fn=None,
                  map_fn=None,
                  n_map_threads=None,
                  filter_after_map=False,
                  shuffle_buffer_size=None):
    if n_map_threads is None:
        n_map_threads = multiprocessing.cpu_count()
    if shuffle and shuffle_buffer_size is None:
        shuffle_buffer_size = max(batch_size * 128, 2048)
    if shuffle:
        dataset = dataset.shuffle(shuffle_buffer_size)
    if not filter_after_map:
        if filter_fn:
            dataset = dataset.filter(filter_fn)
        if map_fn:
            dataset = dataset.map(map_fn, num_parallel_calls=n_map_threads)
    else:
        if map_fn:
            dataset = dataset.map(map_fn, num_parallel_calls=n_map_threads)
        if filter_fn:
            dataset = dataset.filter(filter_fn)
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    dataset = dataset.repeat(repeat).prefetch(n_prefetch_batch)
    return dataset


class Generator(keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        filter = 64
        self.conv1 = layers.Conv2DTranspose(filter * 8, 4, 1, 'valid', use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2DTranspose(filter * 4, 4, 2, 'same', use_bias=False)
        self.bn2 = layers.BatchNormalization()
        self.conv3 = layers.Conv2DTranspose(filter * 2, 4, 2, 'same', use_bias=False)
        self.bn3 = layers.BatchNormalization()
        self.conv4 = layers.Conv2DTranspose(filter * 1, 4, 2, 'same', use_bias=False)
        self.bn4 = layers.BatchNormalization()
        self.conv5 = layers.Conv2DTranspose(3, 4, 2, 'same', use_bias=False)

    def call(self, inputs, training=None):
        x = tf.reshape(inputs, (inputs.shape[0], 1, 1, inputs.shape[1]))
        x = tf.nn.relu(x)
        x = tf.nn.relu(self.bn1(self.conv1(x), training=training))
        x = tf.nn.relu(self.bn2(self.conv2(x), training=training))
        x = tf.nn.relu(self.bn3(self.conv3(x), training=training))
        x = tf.nn.relu(self.bn4(self.conv4(x), training=training))
        x = self.conv5(x)
        x = tf.nn.tanh(x)
        return x


class Dsicriminator(keras.Model):
    def __init__(self):
        super(Dsicriminator, self).__init__()
        filter = 64
        self.conv1 = layers.Conv2D(filter, 4, 2, 'valid', use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(filter * 2, 4, 2, 'valid', use_bias=False)
        self.bn2 = layers.BatchNormalization()
        self.conv3 = layers.Conv2D(filter * 4, 4, 2, 'valid', use_bias=False)
        self.bn3 = layers.BatchNormalization()
        self.conv4 = layers.Conv2D(filter * 8, 3, 1, 'valid', use_bias=False)
        self.bn4 = layers.BatchNormalization()
        self.conv5 = layers.Conv2D(filter * 16, 3, 1, 'valid', use_bias=False)
        self.bn5 = layers.BatchNormalization()
        self.pool = layers.GlobalAveragePooling2D()
        self.flatten = layers.Flatten()
        self.fc = layers.Dense(1)

    def call(self, inputs, training=None):
        x = tf.nn.leaky_relu(self.bn1(self.conv1(inputs), training=training))
        x = tf.nn.leaky_relu(self.bn2(self.conv2(x), training=training))
        x = tf.nn.leaky_relu(self.bn3(self.conv3(x), training=training))
        x = tf.nn.leaky_relu(self.bn4(self.conv4(x), training=training))
        x = tf.nn.leaky_relu(self.bn5(self.conv5(x), training=training))
        x = self.pool(x)
        x = self.flatten(x)
        logits = self.fc(x)
        return logits


def d_loss_fn(generator, discriminator, batch_z, batch_x, is_training):
    fake_image = generator(batch_z, is_training)
    d_fake_logits = discriminator(fake_image, is_training)
    d_real_logits = discriminator(batch_x, is_training)
    d_loss_real = celoss_ones(d_real_logits)
    d_loss_fake = celoss_zeros(d_fake_logits)
    loss = d_loss_fake + d_loss_real
    return loss


def g_loss_fn(generator, discriminator, batch_z, is_training):
    fake_image = generator(batch_z, is_training)
    d_fake_logits = discriminator(fake_image, is_training)
    loss = celoss_ones(d_fake_logits)
    return loss


def celoss_ones(logits):
    y = tf.ones_like(logits)
    loss = losses.binary_crossentropy(y, logits, from_logits=True)
    return tf.reduce_mean(loss)


def celoss_zeros(logits):
    y = tf.zeros_like(logits)
    loss = losses.binary_crossentropy(y, logits, from_logits=True)
    return tf.reduce_mean(loss)


def save_result(val_out, val_block_size, image_path):
    def preprocess(img):
        img = ((img + 1.0) * 127.5).astype(np.uint8)
        return img

    preprocessed = preprocess(val_out)
    final_image = np.array([])
    single_row = np.array([])
    for b in range(val_out.shape[0]):
        if single_row.size == 0:
            single_row = preprocessed[b, :, :, :]
        else:
            single_row = np.concatenate((single_row, preprocessed[b, :, :, :]), axis=1)
        if (b + 1) % val_block_size == 0:
            if final_image.size == 0:
                final_image = single_row
            else:
                final_image = np.concatenate((final_image, single_row), axis=0)
            single_row = np.array([])
    if final_image.shape[2] == 1:
        final_image = np.squeeze(final_image, axis=2)
    Image.fromarray(final_image).convert('RGB').save(image_path)


def main():
    batch_size = 64
    z_dim = 100
    is_training = True

    img_path = glob.glob(r'datasets/faces/*.jpg') + glob.glob(r'datasets/faces/*.png')
    dataset, img_shape, _ = make_anime_dataset(img_path, batch_size, resize=64)
    dataset = dataset.repeat(100)
    db_iter = iter(dataset)

    generator = Generator()
    discriminator = Dsicriminator()
    g_optimizer = optimizers.Adam(lr=2e-4, beta_1=0.5)
    d_optimizer = optimizers.Adam(lr=2e-4, beta_1=0.5)

    d_losses = []
    g_losses = []
    for epoch in range(3000):
        for _ in range(5):
            batch_z = tf.random.normal([batch_size, z_dim])
            batch_x = next(db_iter)
            with tf.GradientTape() as tape:
                d_loss = d_loss_fn(generator, discriminator, batch_z, batch_x, is_training)
            grads = tape.gradient(d_loss, discriminator.trainable_variables)
            d_optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))
        batch_z = tf.random.normal([batch_size, z_dim])
        with tf.GradientTape() as tape:
            g_loss = g_loss_fn(generator, discriminator, batch_z, is_training)
        grads = tape.gradient(g_loss, generator.trainable_variables)
        g_optimizer.apply_gradients(zip(grads, generator.trainable_variables))
        if epoch % 10 == 0:
            print(f'Epoch:{epoch},d-loss:{d_loss.numpy()}, g-loss:{g_loss.numpy()}')
            z = tf.random.normal([100, z_dim])
            fake_image = generator(z, training=False)
            img_path = os.path.join('gan_images', 'gan-%d.png' % epoch)
            save_result(fake_image.numpy(), 10, img_path)
            d_losses.append(d_loss.numpy())
            g_losses.append(g_loss.numpy())
        if (epoch + 1) % 3000 == 0:
            generator.save_weights('generator.ckpt')
            discriminator.save_weights('discriminator.ckpt')


if __name__ == '__main__':
    main()
