import tensorflow as tf
import numpy as np

from nnv_rs import Constellation, DNN


def test_stuff():
    network = tf.keras.models.Sequential()
    network.add(tf.keras.layers.Dense(4, input_dim=4, activation='relu'))
    network.add(tf.keras.layers.Dense(1))

    network.compile(loss='binary_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])

    dnn = DNN(network)
    constellation = Constellation(dnn, np.zeros((4, )), np.ones((4, 4)))
    sample = constellation.bounded_sample()
