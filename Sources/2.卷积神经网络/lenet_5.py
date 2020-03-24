import os

import tensorflow as tf
from tensorflow.keras import layers, Sequential, losses, optimizers, datasets

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
gpu = tf.config.experimental.list_physical_devices('GPU')
assert len(gpu) > 0
tf.config.experimental.set_memory_growth(gpu[0], True)


def main():
    (x, y), (x_test, y_test) = datasets.mnist.load_data()
    x = 2 * tf.convert_to_tensor(x, dtype=tf.float32) / 255. - 1
    y = tf.convert_to_tensor(y, dtype=tf.int32)
    train_db = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(1000).batch(128)

    model = Sequential([
        layers.Conv2D(6, kernel_size=3, strides=1),
        layers.MaxPool2D(pool_size=2, strides=2),
        layers.ReLU(),
        layers.Conv2D(16, kernel_size=3, strides=1),
        layers.MaxPool2D(pool_size=2, strides=2),
        layers.ReLU(),
        layers.Flatten(),
        layers.Dense(120, activation='relu'),
        layers.Dense(84, activation='relu'),
        layers.Dense(10)
    ])
    model.build(input_shape=(4, 28, 28, 1))

    optimizer = optimizers.Adam(lr=1e-3)
    loss = losses.CategoricalCrossentropy(from_logits=True)

    for epoch in range(100):
        for step, (x, y) in enumerate(train_db):
            with tf.GradientTape() as tape:
                x = tf.reshape(x, (-1, 28, 28, 1))
                out = model(x)
                y = tf.one_hot(y, depth=10)
                loss = losses.categorical_crossentropy(y, out, from_logits=True)
                loss = tf.reduce_mean(loss, axis=0)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
        print(f'Epoch:{epoch}, Loss:{loss.numpy()}')

        x = 2 * tf.convert_to_tensor(x_test, dtype=tf.float32) / 255. - 1
        y = tf.convert_to_tensor(y_test, dtype=tf.int64)
        out = model(tf.reshape(x, (-1, 28, 28, 1)))
        correct = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(out, axis=1), y), dtype=tf.int32)) / y.shape[0]
        print(f'Epoch:{epoch}, Accuracy:{correct.numpy()}')


if __name__ == '__main__':
    main()