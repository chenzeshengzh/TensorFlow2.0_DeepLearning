import os
import sys

import tensorflow as tf
from tensorflow.keras import datasets, optimizers, losses
from resnet import resnet18

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = ""


# gpu = tf.config.experimental.list_physical_devices('GPU')
# assert len(gpu) > 0
# tf.config.experimental.set_memory_growth(gpu[0], True)


def preprocess(x, y):
    x = 2 * tf.cast(x, dtype=tf.float32) / 255. - 1
    y = tf.cast(y, dtype=tf.int32)
    return x, y


(x, y), (x_test, y_test) = datasets.cifar10.load_data()
y = tf.squeeze(y, axis=1)
y_test = tf.squeeze(y_test, axis=1)
train_db = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(1000).map(preprocess).batch(128)
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).map(preprocess).batch(128)


def main():
    model = resnet18()
    model.build(input_shape=(128, 32, 32, 3))
    model.summary()
    optimizer = optimizers.Adam(lr=1e-4)
    for epoch in range(20):
        for step, (x, y) in enumerate(train_db):
            with tf.GradientTape() as tape:
                logits = model(x)
                y_onehot = tf.one_hot(y, depth=10)
                loss = tf.reduce_mean(losses.categorical_crossentropy(y_onehot, logits, from_logits=True))
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
        print(f'Epoch:{epoch}, Loss:{loss.numpy()}')

        total = 0
        correct = 0
        for x, y in test_db:
            logits = model(x)
            preds = tf.cast(tf.argmax(logits, axis=1), dtype=tf.int32)
            correct += tf.reduce_sum(tf.cast(tf.equal(preds, y), dtype=tf.int32))
            total += y.shape[0]
        acc = correct / total
        print(f'Epoch:{epoch}, Acc:{acc}')


if __name__ == '__main__':
    main()
