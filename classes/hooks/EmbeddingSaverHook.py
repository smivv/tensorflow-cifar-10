import os
import tensorflow as tf

from tensorflow.contrib.tensorboard.plugins import projector

from classes.Constants import DATA_DIR, LOG_DIR, batch_size, EMBEDDING_SIZE
from classes.Utils import Utils


class EmbeddingSaverHook(tf.train.SessionRunHook):
    _data = None
    _labels = None

    def __init__(self, data, labels):
        self._data = data
        self._labels = labels

        self._saver = None

        self._dropout = None
        self._l2_normalized = None
        self._dense3 = None
        self._labels = None

        self._metadata_file = open(os.path.join(LOG_DIR, 'projector/metadata.tsv'), 'w+')

    def begin(self):
        self._dropout = tf.get_variable("dropout", [batch_size, 1024])
        self._l2_normalized = tf.get_variable("l2_normalized", [batch_size, 1024])
        self._dense3 = tf.get_variable("dense3", [batch_size, EMBEDDING_SIZE])
        self._labels = tf.get_variable("labels", [batch_size])
        self._saver = tf.train.Saver([self._dropout, self._l2_normalized, self._dense3, self._labels])
        pass

    def before_run(self, run_context):
        return tf.train.SessionRunArgs([self._dropout, self._l2_normalized, self._dense3, self._labels])

    def after_run(self, run_context, run_values):

        # metadata_file.write('Name\tClass\n')
        self._metadata_file.write("".join([('%s\n' % x) for x in self._labels.eval(session=run_context.session)]))

    def end(self, session):

        self._metadata_file.close()

        writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'projector'), session.graph)

        config = projector.ProjectorConfig()
        embed = config.embeddings.add()
        embed.tensor_name = 'dropout'
        embed.metadata_path = os.path.join(LOG_DIR + 'projector/metadata.tsv')

        embed = config.embeddings.add()
        embed.tensor_name = 'l2_normalized'
        embed.metadata_path = os.path.join(LOG_DIR + 'projector/metadata.tsv')

        embed = config.embeddings.add()
        embed.tensor_name = '_dense3'
        embed.metadata_path = os.path.join(LOG_DIR + 'projector/metadata.tsv')

        # embed.sprite.image_path = os.path.join(DATA_DIR + '/cifar_10k_sprite.png')
        # embed.sprite.single_image_dim.extend([img_width, img_height])

        projector.visualize_embeddings(writer, config)

        self._saver.save(session, os.path.join(LOG_DIR, "projector/model.ckpt"))

        pass

    def generate_metadata_file(self):
        classnames = Utils.get_classnames()

        metadata_file = open(os.path.join(LOG_DIR, '/metadata.tsv'), 'w')
        # metadata_file.write('Name\tClass\n')

        for i in range(len(self._labels)):
            metadata_file.write('%s\n' % (classnames[self._labels[i]]))

        metadata_file.close()