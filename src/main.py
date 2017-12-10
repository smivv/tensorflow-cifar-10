import sys
import argparse
import logging
import os
import numpy as np

from tensorflow.contrib.tensorboard.plugins import projector
import tensorflow as tf

from classes.Model import Model
from classes.Utils import Utils
from classes.hooks.EmbeddingSaverHook import EmbeddingSaverHook
from classes.Constants import LOG_DIR, num_test_images, num_train_images, batch_size

logging.basicConfig(level=logging.INFO)

FLAGS = None

emb_values = []
emb_labels = []
emb_captions = []


def serve(args):
    # try:
        model = Model()

        # config = tf.contrib.learn.RunConfig(save_checkpoints_steps=FLAGS.steps/10)
        config = tf.contrib.learn.RunConfig()

        # Create the Estimator
        classifier = tf.estimator.Estimator(model_fn=model.inference, config=config, model_dir=LOG_DIR)

        # Set up logging for predictions
        tensors_to_log = {
            # "probabilities": "softmax_tensor"
            # "learning_rate": "learning_rate"
            # "accuracy": "accuracy"
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
                EmbeddingSaverHook(emb_values, emb_labels, emb_captions)
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
                EmbeddingSaverHook(emb_values, emb_labels, emb_captions)
            ]

            data_to_pass = {
                'x': test_images,
                'data': data,
                'labels': labels
            }

            test_input_fn = tf.estimator.inputs.numpy_input_fn(x=data_to_pass, y=test_labels,
                                                               batch_size=batch_size, num_epochs=None, shuffle=True)

            # classifier.predict(input_fn=test_input_fn)

            classifier.evaluate(input_fn=test_input_fn, steps=100, hooks=hooks)

            with open(os.path.join(LOG_DIR, 'projector/metadata.tsv'), 'w+') as f:
                f.write('Index\tCaption\tLabel\n')
                for idx in range(len(emb_labels)):
                    f.write('{:05d}\t{}\t{}\n'
                            .format(idx, emb_captions[idx], emb_labels[idx]))
                f.close()

            with tf.Session() as sess:
                # The embedding variable to be stored
                embedding_var = tf.Variable(np.array(emb_values), name='emb_values')
                sess.run(embedding_var.initializer)

                config = projector.ProjectorConfig()
                embedding = config.embeddings.add()
                embedding.tensor_name = embedding_var.name

                # Add metadata to the log
                embedding.metadata_path = os.path.join(LOG_DIR, "projector/metadata.tsv")

                summary_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'projector/'), sess.graph)
                projector.visualize_embeddings(summary_writer, config)

                saver = tf.train.Saver([embedding_var])
                saver.save(sess, os.path.join(LOG_DIR, "projector/model_emb.ckpt"), 1)

    # except Exception as e:
    #     print(e)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--train', action='store_true', default=False, help='Train CNN.')

    logging.info('Train argument passed..')

    parser.add_argument('--evaluate', action='store_true', default=False, help='Evaluate CNN.')

    logging.info('Evaluate argument passed..')

    # parser.add_argument('--steps', default=10000, help='Number of steps.')
    parser.add_argument('--steps', default=20000, help='Number of steps.')

    logging.info('Steps argument passed..')

    FLAGS, unparsed = parser.parse_known_args()

    logging.info('Processing started..')

    tf.app.run(main=serve, argv=[sys.argv[0]] + unparsed)
