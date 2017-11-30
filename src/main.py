import sys
import argparse
import logging
import numpy as np

import tensorflow as tf

from classes.Model import Model
from classes.Utils import Utils
from classes.hooks.EmbeddingSaverHook import EmbeddingSaverHook
from classes.Constants import LOG_DIR, num_test_images, num_train_images, batch_size

logging.basicConfig(level=logging.INFO)

FLAGS = None


def serve(args):
    # try:
        model = Model()

        config = tf.contrib.learn.RunConfig(save_checkpoints_secs=30)

        # Create the Estimator
        classifier = tf.estimator.Estimator(model_fn=model.inference, config=config, model_dir=LOG_DIR)

        # Set up logging for predictions
        tensors_to_log = {
            # "probabilities": "softmax_tensor"
        }

        if FLAGS.train:
            # Data loading from training dataset
            train_images, train_labels, train_labels_onehot = Utils.load_training_data()

            data = np.zeros([num_train_images])
            labels = np.zeros([num_train_images])

            hooks = [
                # logging hook
                tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50),
                # saver hook
                EmbeddingSaverHook(data, labels)
                # summary hook
                # tf.train.SummarySaverHook(save_secs=2, output_dir=Constants.LOG_DIR,
                #                           scaffold=tf.train.Scaffold(
                #                               summary_op=tf.summary.merge_all()
                #                           ))
            ]

            data_to_pass = {
                'x': train_images,
                'data': data,
                'labels': labels
            }

            train_input_fn = tf.estimator.inputs.numpy_input_fn(x=data_to_pass, y=train_labels,
                                                                batch_size=batch_size, num_epochs=None, shuffle=True)
            classifier.train(input_fn=train_input_fn, steps=FLAGS.steps, hooks=hooks)

        if FLAGS.evaluate:
            # Data loading from dataset
            test_images, test_labels, test_labels_onehot = Utils.load_testing_data()

            data = np.zeros([num_test_images], dtype=float)
            labels = np.zeros([num_test_images])

            hooks = [
                # logging hook
                tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50),
                # saver hook
                EmbeddingSaverHook(data, labels)
                # summary hook
                # tf.train.SummarySaverHook(save_secs=2, output_dir=Constants.LOG_DIR,
                #                           scaffold=tf.train.Scaffold(
                #                               summary_op=tf.summary.merge_all()
                #                           ))
            ]

            data_to_pass = {
                'x': test_images,
                'data': data,
                'labels': labels
            }

            test_input_fn = tf.estimator.inputs.numpy_input_fn(x=data_to_pass, y=test_labels,
                                                               batch_size=batch_size, num_epochs=None, shuffle=True)

            # classifier.predict(input_fn=test_input_fn)

            classifier.evaluate(input_fn=test_input_fn, steps=500, hooks=hooks)

    # except Exception as e:
    #     print(e)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--train', action='store_true', default=False, help='Train CNN.')

    logging.info('Train argument passed..')

    parser.add_argument('--evaluate', action='store_true', default=False, help='Evaluate CNN.')

    logging.info('Evaluate argument passed..')

    parser.add_argument('--steps', default=1000, help='Number of steps.')

    logging.info('Steps argument passed..')

    FLAGS, unparsed = parser.parse_known_args()

    logging.info('Processing started..')

    tf.app.run(main=serve, argv=[sys.argv[0]] + unparsed)
