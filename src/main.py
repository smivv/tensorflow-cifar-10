from classes.Model import Model
from classes.Utils import Utils
from classes import Constants

import tensorflow as tf
import argparse
import logging
import os

logging.basicConfig(level=logging.INFO)


def serve(config):
    try:
        model = Model()

        # Data loading from training dataset
        train_images, train_labels, train_labels_onehot = Utils.load_training_data()

        # Data loading from dataset
        test_images, test_labels, test_labels_onehot = Utils.load_testing_data()

        # Create the Estimator
        classifier = tf.estimator.Estimator(model_fn=model.train, model_dir=Constants.LOG_DIR)

        tensors_to_log = {}
        logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=100)

        # Train the model
        train_input_fn = tf.estimator.inputs.numpy_input_fn(x={'x': train_images}, y=train_labels, batch_size=100, num_epochs=None, shuffle=True)
        test_input_fn = tf.estimator.inputs.numpy_input_fn(x={'x': test_images}, y=test_labels, batch_size=100, num_epochs=None, shuffle=True)

        # classifier.train(input_fn=train_input_fn, steps=10000)

        # classifier.predict(input_fn=test_input_fn)
        classifier.evaluate(input_fn=test_input_fn, steps=100)
    except Exception as e:
        print(e)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--train', help='Output layer size of CNN.')

    logging.info('Output layer size argument passed..')

    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = parse_args()

    logging.info('Processing started..')

    # Run service
    serve(config=args)
