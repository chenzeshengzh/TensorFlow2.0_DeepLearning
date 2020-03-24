import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses, optimizers, datasets

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
gpu = tf.config.experimental.list_physical_devices('GPU')
assert len(gpu) > 0
tf.config.experimental.set_memory_growth(gpu[0], True)

(x, y), (x_test, y_test) = datasets.cifar10.load_data()
x = 2 * tf.convert_to_tensor(x, dtype=tf.float32) / 255. - 1
y = tf.convert_to_tensor(y, dtype=tf.int32)
train_db = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(50000).batch(128)
x_test = 2 * tf.convert_to_tensor(x_test, dtype=tf.float32) / 255. - 1
y_test = tf.convert_to_tensor(y_test, dtype=tf.int32)
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(128)


class VGG13(keras.Model):
    def __init__(self):
        super(VGG13, self).__init__()
        self.conv1_1 = layers.Conv2D(64, kernel_size=3, padding='same', activation='relu')
        self.conv1_2 = layers.Conv2D(64, kernel_size=3, padding='same', activation='relu')
        self.pool1 = layers.MaxPool2D(pool_size=2, strides=2, padding='same')
        self.conv2_1 = layers.Conv2D(128, kernel_size=3, padding='same', activation='relu')
        self.conv2_2 = layers.Conv2D(128, kernel_size=3, padding='same', activation='relu')
        self.pool2 = layers.MaxPool2D(pool_size=2, strides=2, padding='same')
        self.conv3_1 = layers.Conv2D(256, kernel_size=3, padding='same', activation='relu')
        self.conv3_2 = layers.Conv2D(256, kernel_size=3, padding='same', activation='relu')
        self.pool3 = layers.MaxPool2D(pool_size=2, strides=2, padding='same')
        # self.conv4_1 = layers.Conv2D(512, kernel_size=3, padding='same', activation='relu')
        # self.conv4_2 = layers.Conv2D(512, kernel_size=3, padding='same', activation='relu')
        # self.pool4 = layers.MaxPool2D(pool_size=2, strides=2)
        # self.conv5_1 = layers.Conv2D(512, kernel_size=3, padding='same', activation='relu')
        # self.conv5_2 = layers.Conv2D(512, kernel_size=3, padding='same', activation='relu')
        # self.pool5 = layers.MaxPool2D(pool_size=2, strides=2)
        self.fla = layers.Flatten()
        self.fc1 = layers.Dense(256, activation='relu')
        self.fc2 = layers.Dense(128, activation='relu')
        self.fc3 = layers.Dense(10)

    def call(self, inputs):
        x = self.pool1(self.conv1_2(self.conv1_1(inputs)))
        x = self.pool2(self.conv2_2(self.conv2_1(x)))
        x = self.pool3(self.conv3_2(self.conv3_1(x)))
        # x = self.pool4(self.conv4_2(self.conv4_1(x)))
        # x = self.pool5(self.conv5_2(self.conv5_1(x)))
        x = self.fla(x)
        x = self.fc3(self.fc2(self.fc1(x)))
        return x


def main():
    net = VGG13()
    optimizer = optimizers.Adam(lr=1e-4)

    for epoch in range(100):
        for step, (x, y) in enumerate(train_db):
            with tf.GradientTape() as tape:
                out = net(x)
                y = tf.one_hot(y, depth=10)
                loss = tf.reduce_mean(losses.categorical_crossentropy(y, out, from_logits=True), axis=0)
            grads = tape.gradient(loss, net.trainable_variables)
            optimizer.apply_gradients(zip(grads, net.trainable_variables))
        print(f'Epoch:{epoch}, Loss:{loss.numpy()}')
        total = 0
        correct = 0
        for x, y in test_db:
            out = net(x)
            out = tf.argmax(out, axis=1)
            out = tf.cast(out, dtype=tf.int32)
            correct += tf.reduce_sum(tf.cast(tf.equal(out, y), dtype=tf.int32))
            total += y.shape[0]
        acc = correct / total
        print(f'Epoch:{epoch}, Acc:{acc}')


if __name__ == '__main__':
    main()
