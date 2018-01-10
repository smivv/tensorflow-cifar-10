import tensorflow as tf

from datasets.cifar10 import Cifar10 as Dataset

EMBEDDING_SIZE = 3


class Cifar10:

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    """--------------------------------------------------------------------------------------------------------------"""

    @staticmethod
    def architecture(features, params):
        with tf.variable_scope(params.scopes['net_scope']):
            images = tf.cast(features, tf.float32)
            # images = features
            # Input Layer
            with tf.name_scope('Data'):
                input_layer = tf.reshape(images, [-1, Dataset.IMG_SIZE, Dataset.IMG_SIZE, Dataset.NUM_CHANNELS])

            # Convolutional Layer 1
            with tf.variable_scope('Conv1'):
                conv1 = tf.layers.conv2d(inputs=input_layer, filters=32, kernel_size=[3, 3], padding="same", activation=tf.nn.elu)
                pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=1)

            tf.logging.info('Convolutional Layer 1 build successful..')

            # Convolutional Layer 2
            with tf.variable_scope('Conv2'):
                conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[3, 3], padding="same", activation=tf.nn.elu)
                pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=1)

            tf.logging.info('Convolutional Layer 2 build successful..')

            # Convolutional Layer 3
            with tf.variable_scope('Conv3'):
                conv3 = tf.layers.conv2d(inputs=pool2, filters=128, kernel_size=[3, 3], padding="same", activation=tf.nn.elu)
                pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

            tf.logging.info('Convolutional Layer 3 build successful..')

            # Convolutional Layer 4
            with tf.variable_scope('Conv4'):
                conv4 = tf.layers.conv2d(inputs=pool3, filters=256, kernel_size=[3, 3], padding="same",
                                         activation=tf.nn.elu)
                pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)

            tf.logging.info('Convolutional Layer 4 build successful..')

            # Fully Connected Layer
            with tf.variable_scope('FC'):
                pool2_flat = tf.reshape(pool4, [-1, 7 * 7 * 256])
                dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.elu)

            tf.logging.info('Fully Connected Layer build successful..')

            """ ---------------------------------------------------------------------------------------------------- """

            # Embeddings Layer
            with tf.variable_scope(params.scopes['emb_scope']):
                l2_norm = tf.nn.l2_normalize(dense, dim=1, name='l2norm')

                tf.summary.histogram("l2norm", l2_norm)

                embeddings = tf.layers.dense(inputs=l2_norm, kernel_initializer=tf.initializers.random_normal,
                                             kernel_regularizer=tf.contrib.layers.l2_regularizer,
                                             units=EMBEDDING_SIZE, name=params.scopes['emb_name'])

                tf.summary.histogram(params.scopes['emb_name'], embeddings)

                tf.logging.info('Embeddings Layer build successful..')

            """ ---------------------------------------------------------------------------------------------------- """

            # Logits Layer
            with tf.variable_scope('LogitsLayer'):
                logits = tf.layers.dense(inputs=embeddings, units=10)

                tf.summary.histogram('logits', logits)

                tf.logging.info('Logits Layer build successful..')

        return logits

    @staticmethod
    def get_estimator_spec(features, labels, mode, params):

        logits = Cifar10.architecture(features, params)

        # Metrics Layer
        with tf.variable_scope(params.scopes['metrics_scope']):
            # Generate predictions (for PREDICT and EVAL mode)
            predictions = {
                "classes": tf.argmax(input=logits, axis=1),
                # Add `softmax_tensor` to the graph. It is used for PREDICT.
                "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
            }

            tf.logging.info('Predictions build successful..')

            # Calculate Loss (for both TRAIN and EVAL modes)
            labels = tf.identity(labels, name=params.scopes['label_name'])
            one_hot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
            loss = tf.losses.softmax_cross_entropy(onehot_labels=one_hot_labels, logits=logits)

            tf.summary.histogram('loss', loss)

            tf.logging.info('Losses build successful..')

            learning_rate = tf.train.exponential_decay(
                params.learning_rate,  # Base learning rate.
                tf.train.get_global_step(),  # Current index into the dataset.
                1000,  # Decay step.
                0.75,  # Decay rate.
                staircase=True,
                name='learning_rate'
            )

            tf.summary.histogram('learning_rate', learning_rate)

            tf.logging.info('Learning rate build successful..')

            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            # optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
            train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

            # Add evaluation metrics (for EVAL mode)
            accuracy = tf.metrics.accuracy(labels=labels, predictions=predictions["classes"], name='accuracy')

            tf.summary.histogram('accuracy', accuracy)

            tf.logging.info('Accuracy metric build successful..')

            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op,
                                              predictions=predictions,
                                              eval_metric_ops={"accuracy": accuracy},
                                              scaffold=tf.train.Scaffold(
                                                  summary_op=tf.summary.merge_all()
                                              ))

    """ ------------------------------------------------------------------------------------------------------------ """

