import logging
import os

import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector


from datasets.cifar10 import Cifar10 as Dataset
from models.cifar10 import Cifar10 as Model

from hooks.EmbeddingSaverHook import EmbeddingSaverHook

logging.basicConfig(level=logging.INFO)

""" ---------------------------------- Flags ---------------------------------- """

tf.app.flags.DEFINE_boolean('train', False, 'Flag for model training.')

tf.app.flags.DEFINE_boolean('evaluate', False, 'Flag for model evaluation.')

tf.app.flags.DEFINE_string('dataset_dir', "tmp/cifar-10-batches-py/", 'Flag for model evaluation.')

tf.app.flags.DEFINE_string('checkpoint_dir', "tmp/checkpoints/", 'Checkpoint dir where .ckpt file lies.')

tf.app.flags.DEFINE_integer('batch_size', 100, 'Butch size.')

tf.app.flags.DEFINE_integer('steps', 30000, 'Number of steps.')

tf.app.flags.DEFINE_integer('learning_rate', 0.0005, 'Starting learning rate.')

FLAGS = tf.app.flags.FLAGS


def main(_):
    model = Model(FLAGS.learning_rate)
    dataset = Dataset(FLAGS.dataset_dir)

    # config = tf.contrib.learn.RunConfig(save_checkpoints_steps=FLAGS.steps/10)
    config = tf.contrib.learn.RunConfig()

    # Create the Estimator
    classifier = tf.estimator.Estimator(model_fn=model.inference, config=config, model_dir=FLAGS.checkpoint_dir)

    # Set up logging for predictions
    tensors_to_log = {
        # "probabilities": "softmax_tensor"
        # "learning_rate": "learning_rate"
        # "accuracy": "accuracy"
    }

    # Init Logging Hook
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

    # Init Embeddings Saver Hook
    embeddings_hook = EmbeddingSaverHook()

    if FLAGS.train:

        # Data loading from training dataset
        train_images, train_labels, train_labels_onehot = dataset.load_training_data()

        hooks = [
            # Set Logging hook
            logging_hook,
            # Set Embeddings hook
            embeddings_hook
        ]

        data_to_pass = {
            'x': train_images
        }

        train_input_fn = tf.estimator.inputs.numpy_input_fn(x=data_to_pass,
                                                            y=train_labels,
                                                            batch_size=FLAGS.batch_size,
                                                            num_epochs=None,
                                                            shuffle=True)

        classifier.train(input_fn=train_input_fn, steps=FLAGS.steps, hooks=hooks)

    elif FLAGS.evaluate:

        # Data loading from dataset
        test_images, test_labels, test_labels_onehot = dataset.load_testing_data()

        hooks = [
            # logging hook
            logging_hook,
            # saver hook
            embeddings_hook
        ]

        data_to_pass = {
            'x': test_images
        }

        test_input_fn = tf.estimator.inputs.numpy_input_fn(x=data_to_pass,
                                                           y=test_labels,
                                                           batch_size=FLAGS.batch_size,
                                                           num_epochs=None,
                                                           shuffle=True)

        classifier.evaluate(input_fn=test_input_fn, steps=100, hooks=hooks)

        """ ------------------------------------------ Embeddings saving ------------------------------------------ """

        classes = dataset.get_classnames()
        embeddings = embeddings_hook.get_embeddings()

        values = embeddings['values']
        labels = embeddings['labels']
        captions = [classes[x] for x in labels]

        if not os.path.isdir(os.path.join(FLAGS.checkpoint_dir, 'projector')):
            os.makedirs(os.path.join(FLAGS.checkpoint_dir, 'projector'))

        with open(os.path.join(FLAGS.checkpoint_dir, 'projector/metadata.tsv'), 'w+') as f:
            f.write('Index\tCaption\tLabel\n')
            for idx in range(len(labels)):
                f.write('{:05d}\t{}\t{}\n'
                        .format(idx, captions[idx], labels[idx]))
            f.close()

        with tf.Session() as sess:
            # The embedding variable to be stored
            embedding_var = tf.Variable(np.array(values), name='emb_values')
            sess.run(embedding_var.initializer)

            config = projector.ProjectorConfig()
            embedding = config.embeddings.add()
            embedding.tensor_name = embedding_var.name

            # Add metadata to the log
            embedding.metadata_path = os.path.join(FLAGS.checkpoint_dir, "projector/metadata.tsv")

            summary_writer = tf.summary.FileWriter(os.path.join(FLAGS.checkpoint_dir, 'projector/'), sess.graph)
            projector.visualize_embeddings(summary_writer, config)

            saver = tf.train.Saver([embedding_var])
            saver.save(sess, os.path.join(FLAGS.checkpoint_dir, "projector/model_emb.ckpt"), 1)
    else:
        raise RuntimeError("You should provide --train or --evaluate flag.")


if __name__ == '__main__':

    logging.info('Processing started..')

    tf.app.run()
