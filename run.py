import os

import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.contrib.learn import learn_runner

from datasets.cifar10 import Cifar10

from classes.Experiment import Experiment
from classes.SessionHook import SessionHook

# Show debugging output
tf.logging.set_verbosity(tf.logging.DEBUG)

""" ---------------------------------- Flags ---------------------------------- """

tf.app.flags.DEFINE_boolean('train', False, 'Flag for model training.')

tf.app.flags.DEFINE_boolean('evaluate', False, 'Flag for model evaluation.')

tf.app.flags.DEFINE_integer('batch_size', 100, 'Butch size.')

tf.app.flags.DEFINE_integer('n_classes', 10, 'Number of classes.')

tf.app.flags.DEFINE_integer('steps', 20000, 'Number of steps.')

tf.app.flags.DEFINE_integer('min_eval_frequency', 100, 'Number of steps.')

tf.app.flags.DEFINE_integer('learning_rate', 0.0005, 'Starting learning rate.')

tf.app.flags.DEFINE_string('model', "cifar-10", 'Dataset provider.')

tf.app.flags.DEFINE_string('dataset_dir', "tmp/cifar-10-batches-py/", 'Flag for model evaluation.')

tf.app.flags.DEFINE_string('checkpoint_dir', "tmp/2/", 'Checkpoint dir where .ckpt file lies.')

FLAGS = tf.app.flags.FLAGS


def main(_):
    """Run the training experiment."""

    scopes = {
        'net_scope': 'CifarConvNet',
        'emb_scope': 'EmbLayer',
        'metrics_scope': 'MetricsLayer',
        'emb_name': 'Embeddings',
        'label_name': 'labels'
    }

    # Define hook
    hook = SessionHook(scopes)

    # Define model parameters
    params = tf.contrib.training.HParams(
        learning_rate=FLAGS.learning_rate,
        n_classes=FLAGS.n_classes,
        batch_size=FLAGS.batch_size,
        train_steps=FLAGS.steps,
        # min_eval_frequency=100,
        model=FLAGS.model,
        dataset_dir=FLAGS.dataset_dir,
        checkpoint_dir=FLAGS.checkpoint_dir,
        hooks={
            # 'iterator_init_hook': iterator_init_hook,
            # 'embeddings_hook': embeddings_hook,
            'hook': hook
        },
        scopes=scopes
    )

    if FLAGS.train and FLAGS.evaluate:
        what_to_run = "train_and_evaluate"
    elif FLAGS.train:
        what_to_run = "train"
    elif FLAGS.evaluate:
        what_to_run = "evaluate"
    else:
        raise EnvironmentError("You have to init --train or --evaluate flag.")

    # Set the run_config and the directory to save the model and stats
    run_config = tf.contrib.learn.RunConfig()
    run_config = run_config.replace(model_dir=FLAGS.checkpoint_dir)

    learn_runner.run(
        experiment_fn=Experiment(params).get_experiment_fn,  # First-class function
        run_config=run_config,  # RunConfig
        schedule=what_to_run,  # What to run
        hparams=params  # HParams
    )

    """ ------------------------------------------ Embeddings saving ------------------------------------------ """

    classes = Cifar10(params.dataset_dir).get_classnames()
    embeddings = hook.get_embeddings()

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

        writer = tf.summary.FileWriter(os.path.join(FLAGS.checkpoint_dir, 'projector/'), sess.graph)
        projector.visualize_embeddings(writer, config)

        saver = tf.train.Saver([embedding_var])
        saver.save(sess, os.path.join(FLAGS.checkpoint_dir, "projector/model_emb.ckpt"), 1)


if __name__ == '__main__':

    tf.logging.info('Processing started..')

    tf.app.run()
