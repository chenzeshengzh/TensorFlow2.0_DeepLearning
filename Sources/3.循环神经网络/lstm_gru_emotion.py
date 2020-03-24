import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential, losses, optimizers, datasets

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
train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1000).batch(batch_size)
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

word_index = datasets.imdb.get_word_index()

GLOVE_DIR = r'datasets/glove.6B'
embedding_index = {}
with open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'), encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = coefs

num_words = min(total_words, len(word_index))
embedding_matrix = np.zeros((num_words, embedding_len))
for word, i in word_index.items():
    if i >= num_words:
        continue
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


class LSTM(keras.Model):
    def __init__(self, units):
        super(LSTM, self).__init__()
        self.embedding = layers.Embedding(input_dim=num_words, output_dim=embedding_len,
                                          input_length=max_review_len, trainable=False)
        self.embedding.build(input_shape=(None, max_review_len))
        self.embedding.set_weights([embedding_matrix])
        self.lstm = Sequential([
            layers.LSTM(units, dropout=0.5, return_sequences=True),
            layers.LSTM(units, dropout=0.5)
        ])
        self.outlayer = Sequential([
            layers.Dense(32, activation='relu'),
            layers.Dropout(rate=0.5),
            layers.Dense(1, activation='sigmoid')
        ])

    def call(self, inputs, training=None):
        x = self.embedding(inputs)
        x = self.lstm(x)
        x = self.outlayer(x)
        return x


class GRU(keras.Model):
    def __init__(self, units):
        super(GRU, self).__init__()
        self.embedding = layers.Embedding(input_dim=num_words, output_dim=embedding_len,
                                          input_length=max_review_len, trainable=False)
        self.embedding.build(input_shape=(None, max_review_len))
        self.embedding.set_weights([embedding_matrix])
        self.gru = Sequential([
            layers.GRU(units, dropout=0.5, return_sequences=True),
            layers.GRU(units, dropout=0.5)
        ])
        self.outlayer = Sequential([
            layers.Dense(32, activation='relu'),
            layers.Dropout(rate=0.5),
            layers.Dense(1, activation='sigmoid')
        ])

    def call(self, inputs, training=None):
        x = self.embedding(inputs)
        x = self.gru(x)
        x = self.outlayer(x)
        return x


def main():
    units = 64
    epochs = 50
    lstm_model = LSTM(units)
    lstm_model.compile(optimizer=optimizers.Adam(1e-3),
                       loss=losses.BinaryCrossentropy(),
                       metrics=['accuracy'],
                       experimental_run_tf_function=False)
    lstm_model.fit(train_db, epochs=epochs, validation_data=test_db)
    lstm_model.evaluate(test_db)

    gru_model = GRU(units)
    gru_model.compile(optimizer=optimizers.Adam(1e-3),
                      loss=losses.BinaryCrossentropy(),
                      metrics=['accuracy'],
                      experimental_run_tf_function=False)
    gru_model.fit(train_db, epochs=epochs, validation_data=test_db)
    gru_model.evaluate(test_db)


if __name__ == '__main__':
    main()
