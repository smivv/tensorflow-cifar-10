import tensorflow as tf
import numpy as np


class ModelCifar10:

    """--------------------------------------------------------------------------------------------------------------"""

    def __init__(self, input_size, hidden_size, output_size, std=1e-4):

        self._input_size = int(input_size)
        self._hidden_size = int(hidden_size)
        self._output_size = int(output_size)

        # Hyperparameters
        self.W1 = std * np.random.randn(self._input_size, self._hidden_size)
        self.W2 = std * np.random.randn(self._hidden_size, self._output_size)

        self.b1 = np.zeros(self._hidden_size)
        self.b2 = np.zeros(self._output_size)

    """--------------------------------------------------------------------------------------------------------------"""

    def read(self, file):
        """
        Read data.

        Read bunch of data for 10000 images from CIFAR-10 file.

        Parameters
        ----------
        file : string
            File name

        Returns
        -------
        dict
            Dictionary with the following elements:
            -- data - a 10000x3072 numpy array of uint8s. Each row of the array stores a 32x32 colour image.
            The first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue.
            The image is stored in row-major order, so that the first 32 entries of the array are the red channel values of the first row of the image.
            -- labels - a list of 10000 numbers in the range 0-9. The number at index i indicates the label of the ith image in the array data.

        """
        import pickle
        with open(file, 'rb') as fo:
            dictionary = pickle.load(fo, encoding='bytes')
        return dictionary

    def loss(self):
        pass

    def train(self):
        pass

    def predict(self):
        pass
