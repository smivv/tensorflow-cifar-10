import tensorflow as tf
import numpy as np
import logging


class ModelCifar10:
    """
    A two-layer fully-connected neural network. The net has an input dimension of
    N, a hidden layer dimension of H, and performs classification over C classes.
    We train the network with a softmax loss function and L2 regularization on the
    weight matrices. The network uses a ReLU nonlinearity after the first fully
    connected layer.

    In other words, the network has the following architecture:

    input - fully connected layer - ReLU - fully connected layer - softmax

    The outputs of the second fully-connected layer are the scores for each class.
    """

    """--------------------------------------------------------------------------------------------------------------"""

    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:

        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (H,)
        W2: Second layer weights; has shape (H, C)
        b2: Second layer biases; has shape (C,)

        Inputs:
        - input_size: The dimension D of the input data.
        - hidden_size: The number of neurons H in the hidden layer.
        - output_size: The number of classes C.
        """

        self._input_size = int(input_size)
        self._hidden_size = int(hidden_size)
        self._output_size = int(output_size)

        # Input
        self.x = tf.placeholder(tf.float32, [None, self._input_size])

        # Layer 1
        with tf.variable_scope('layer1'):
            # Hyperparameters
            self.W1 = tf.get_variable('W1', [self._input_size, self._hidden_size], initializer=tf.random_normal_initializer())
            self.b1 = tf.get_variable('b1', [self._hidden_size, ], initializer=tf.random_normal_initializer())
            # Activation
            self.y1 = tf.nn.relu(tf.matmul(self.x, self.W1) + self.b1)

        # Layer 2
        with tf.variable_scope('layer2'):
            # Hyperparameters
            self.W2 = tf.get_variable('W2', [self._hidden_size, self._output_size], initializer=tf.random_normal_initializer())
            self.b2 = tf.get_variable('b2', [self._output_size, ], initializer=tf.random_normal_initializer())
            # Activation
            self.y2 = tf.nn.relu(tf.matmul(self.y1, self.W2) + self.b2)

        # Add an op to initialize the variables.
        self.init_op = tf.global_variables_initializer()

        logging.info('Output layer size argument passed..')

    """--------------------------------------------------------------------------------------------------------------"""

    def launch(self):
        with tf.Session() as sess:
            # Run the init operation.
            sess.run(self.init_op)

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
