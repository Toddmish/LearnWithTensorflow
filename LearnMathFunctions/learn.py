"""
@author Todd

"""


import tensorflow as tf
import numpy as np
from tensorflow import keras



class Learn:
    def learn_model(self, y_new, training_data, epochs_no=0):
        training_x, training_y= training_data[0], training_data[1]
        model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
        model.compile(optimizer='sgd', loss='mean_squared_error')
        model.fit(training_x, training_y, epochs=epochs_no)
        return model.predict(y_new)[0]

    def learnLinearFunc(self, epochs_no):
        """

        :param epochs_no:
        :return predicted output:
        """
        traing_data = [np.array([0, 1, 2, 3, 4, 5, 6, 7, 8.0], dtype=float),
                       np.array([2, 3, 4, 5.00, 6, 7, 8, 9, 10], dtype=float)]

        prediction = Learn.learn_model([9], traing_data, epochs_no)
        return prediction

    def learnLinearFunc(self, input_x, sequence_length, epochs_no):
        """

        :param input:
        :param sequence_length:
        :param epochs_no:
        :return output:
        """
        tr_x = []
        tr_y = []

        for i in range(sequence_length):
            tr_x.append(i)
            tr_y.append(i + 2)

        tr_x = np.array(tr_x, dtype=np.float)
        tr_y = np.array(tr_y, dtype=np.float)

        traing_data = np.array([tr_x, tr_y])

        print(traing_data)

        prediction = self.learn_model([input_x], traing_data, epochs_no)
        return prediction




