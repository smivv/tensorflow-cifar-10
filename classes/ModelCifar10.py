import tensorflow as tf
import logging

from classes.Utils import Utils


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

        if not Utils.download():
            logging.error("Dataset could not be downloaded..")
            return

        # Data loading from training dataset
        train_images, train_labels = Utils.load_training_data()

        # Data loading from dataset
        test_images, test_labels = Utils.load_testing_data()

        # Correct labels
        y_ = tf.placeholder(tf.float32, [None, self._output_size])

        # Input
        self.x = tf.placeholder(tf.float32, [None, self._input_size])

        # Layer 1
        with tf.variable_scope('layer1'):
            # Hyperparameters
            self.W1 = tf.get_variable('W1', [self._input_size, self._hidden_size], initializer=tf.random_normal_initializer())
            self.b1 = tf.get_variable('b1', [self._hidden_size, ], initializer=tf.random_normal_initializer())
            # Activation
            self.h1 = tf.nn.relu(tf.matmul(self.x, self.W1) + self.b1)

        logging.info('First layer build successfull..')

        # Layer 2
        with tf.variable_scope('layer2'):
            # Hyperparameters
            self.W2 = tf.get_variable('W2', [self._hidden_size, self._output_size], initializer=tf.random_normal_initializer())
            self.b2 = tf.get_variable('b2', [self._output_size, ], initializer=tf.random_normal_initializer())
            # Activation
            output = tf.nn.softmax(tf.matmul(self.h1, self.W2) + self.b2)

        logging.info('Second layer build successfull..')

        # define the loss function
        self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(output), reduction_indices=[1]))

        logging.info('Loss function initialized..')

        # define training step and accuracy
        train_step = tf.train.MomentumOptimizer(learning_rate=0.1, momentum=0.9).minimize(self.cross_entropy)
        correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        logging.info('Training step and accuracy defined..')

    """--------------------------------------------------------------------------------------------------------------"""

    def run(self):
        pass

    """--------------------------------------------------------------------------------------------------------------"""

