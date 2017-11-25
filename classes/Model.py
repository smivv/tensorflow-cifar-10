import tensorflow as tf
import logging
import os

from classes.Utils import Utils
from classes import Constants

from tensorflow.contrib.tensorboard.plugins import projector

class Model:
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

    def __init__(self):
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

        if not Utils.download():
            logging.error("Dataset could not be downloaded..")
            return

    """--------------------------------------------------------------------------------------------------------------"""

    # def generate_embeddings():
        # Import data
        # mnist = input_data.read_data_sets(FLAGS.data_dir,
        #                                   one_hot=True,
        #                                   fake_data=FLAGS.fake_data)
        # sess = tf.InteractiveSession()
        #
        # # Input set for Embedded TensorBoard visualization
        # # Performed with cpu to conserve memory and processing power
        # with tf.device("/cpu:0"):
        #     embedding = tf.Variable(tf.stack(mnist.test.images[:FLAGS.max_steps], axis=0), trainable=False,
        #                             name='embedding')
        #
        # tf.global_variables_initializer().run()
        #
        # saver = tf.train.Saver()
        # writer = tf.summary.FileWriter(FLAGS.log_dir + '/projector', sess.graph)
        #
        # # Add embedding tensorboard visualization. Need tensorflow version
        # # >= 0.12.0RC0
        # config = projector.ProjectorConfig()
        # embed = config.embeddings.add()
        # embed.tensor_name = 'embedding:0'
        # embed.metadata_path = os.path.join(FLAGS.log_dir + '/projector/metadata.tsv')
        # embed.sprite.image_path = os.path.join(FLAGS.data_dir + '/mnist_10k_sprite.png')
        #
        # # Specify the width and height of a single thumbnail.
        # embed.sprite.single_image_dim.extend([28, 28])
        # projector.visualize_embeddings(writer, config)
        #
        # saver.save(sess, os.path.join(
        #     FLAGS.log_dir, 'projector/a_model.ckpt'), global_step=FLAGS.max_steps)

    """--------------------------------------------------------------------------------------------------------------"""

    def train(self, features, labels, mode):
        """

        [INPUT] -> [CONV] -> [POOL] -> [CONV] -> [POOL] -> [FC] ->

        """

        try:
            images = tf.cast(features['x'], tf.float32)
            # Input Layer
            with tf.name_scope('Data'):
                input_layer = tf.reshape(images, [-1, Constants.img_width, Constants.img_height, Constants.num_channels])

            # Convolutional Layer 1
            with tf.variable_scope('ConvLayer1'):
                conv1 = tf.layers.conv2d(inputs=input_layer, filters=32, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
                pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

            logging.info('Convolutional Layer 1 build successfull..')

            # Convolutional Layer 1
            with tf.variable_scope('ConvLayer2'):
                conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
                pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

            logging.info('Convolutional Layer 2 build successfull..')

            # Fully Connected Layer
            with tf.variable_scope('FullyConnectedLayer'):
                pool2_flat = tf.reshape(pool2, [-1, 8 * 8 * 64])
                dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
                dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=(mode == tf.estimator.ModeKeys.TRAIN))

            logging.info('Fully Connected Layer build successfull..')

            # Logits Layer
            logits = tf.layers.dense(inputs=dropout, units=10)

            logging.info('Logits Layer build successfull..')

            predictions = {
                # Generate predictions (for PREDICT and EVAL mode)
                "classes": tf.argmax(input=logits, axis=1),
                # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
                # `logging_hook`.
                "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
            }

            if mode == tf.estimator.ModeKeys.PREDICT:
                return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

            # Calculate Loss (for both TRAIN and EVAL modes)
            onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
            loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

            logging.info('Losses build successfull..')

            # Configure the Training Op (for TRAIN mode)
            if mode == tf.estimator.ModeKeys.TRAIN:
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
                train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
                return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

            # Add evaluation metrics (for EVAL mode)
            eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])}

            logging.info('Accuracy metric build successfull..')

            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
        except Exception as e:
            print(e)


    """--------------------------------------------------------------------------------------------------------------"""


    """--------------------------------------------------------------------------------------------------------------"""

