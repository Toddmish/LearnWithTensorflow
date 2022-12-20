import tensorflow as tf
import matplotlib.pyplot as plt
from os import path, getcwd, chdir


class MNIST:

    def train(self):

        accuracy_field_name='accuracy' # it could be 'acc' for other versions and would throw exception
        accuracy_level= 0.98


        #This is a dataset of 60,000 28x28 grayscale images of the 10 digits, along
        #with a test set of 10,000 images.More info can be found at the MNIST homepage.
        dataset_name='mnist.npz'

        epochs_no = 10

        class myCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs={}):
                if logs.get(accuracy_field_name) > accuracy_level:
                    print("\nAccuracy", accuracy_level, "reached.")
                    self.model.stop_training = True

        callbacks = myCallback()

        print(getcwd())

        path = f"{getcwd()}/" + dataset_name
        print(path)

        (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.mnist.load_data(path=path)

        # x_train[tr_example_index][i][j] respresents just a pixel of one-dimensional value and
        # not a three-dimensional (R,G,B) because images are greyscaled, i.e:  Pixel values range from 0 to 255.
        #print(x_train[59999][14][14])
        #print(y_train[59998])

        #Scaling for faster convergence
        self.x_train = self.x_train / 255.0

        self.model = tf.keras.models.Sequential([

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation=tf.nn.relu),
            tf.keras.layers.Dense(10, activation=tf.nn.softmax),
        ])

        self.model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=[accuracy_field_name])

        # model fitting
        history = self.model.fit(
            self.x_train, self.y_train, epochs=epochs_no, callbacks=[callbacks]
        )

        return history.epoch, history.history[accuracy_field_name][-1]
    def predict(self,selected):

        image_index = 5555
        plt.imshow(self.x_test[image_index].reshape(28, 28), cmap='Greys')
        pred = self.model.predict(self.x_test[image_index].reshape(1, 28, 28, 1))
        print(pred.argmax())




if __name__ == '__main__':
    print('Using tensorflow to learn from images')

    mnist=MNIST()
    mnist.train()
    selected=0
    mnist.predict(selected)

