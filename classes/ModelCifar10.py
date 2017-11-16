import tensorflow as tf



class ModelCifar10:

    def __init__(self, input_size, hidden_size, output_size):

        self._input_size = input_size
        self._hidden_size = hidden_size
        self._output_size = output_size

    def read(self, file):
        """
        Read data.

        Read data from file.

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
