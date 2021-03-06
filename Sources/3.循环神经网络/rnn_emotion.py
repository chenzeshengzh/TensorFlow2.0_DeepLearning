import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, losses, optimizers, Sequential, layers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
gpu = tf.config.experimental.list_physical_devices('GPU')
assert len(gpu) > 0
tf.config.experimental.set_memory_growth(gpu[0], True)

batch_size = 128
total_words = 10000
max_review_len = 80
embedding_len = 100

(x_train, y_train), (x_test, y_test) = datasets.imdb.load_data(num_words=total_words)
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_review_len)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_review_len)
train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1000).batch(batch_size, drop_remainder=True)
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size, drop_remainder=True)


class MyRNN(keras.Model):
    def __init__(self, units):
        super(MyRNN, self).__init__()
        self.state0 = [tf.zeros([batch_size, units])]
        self.state1 = [tf.zeros([batch_size, units])]
        self.embedding = layers.Embedding(total_words, embedding_len, input_length=max_review_len)
        self.rnn_cell0 = layers.SimpleRNNCell(units, dropout=0.5)
        self.rnn_cell1 = layers.SimpleRNNCell(units, dropout=0.5)
        self.out_layer = layers.Dense(1, activation='sigmoid')

    def call(self, inputs, training=None):
        x = self.embedding(inputs)
        state0 = self.state0
        state1 = self.state1
        for word in tf.unstack(x, axis=1):
            out0, state0 = self.rnn_cell0(word, state0, training)
            out1, state1 = self.rnn_cell1(out0, state1, training)
        x = self.out_layer(out1)
        return x


def main():
    units = 64
    epochs = 20
    model = MyRNN(units)
    model.compile(optimizer=optimizers.Adam(1e-3), loss=losses.BinaryCrossentropy(),
                  metrics=['accuracy'], experimental_run_tf_function=False)
    model.fit(train_db, epochs=epochs, validation_data=test_db)
    model.evaluate(test_db)


if __name__ == '__main__':
    main()
