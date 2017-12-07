import os
import numpy
import tensorflow as tf

from tensorflow.contrib.tensorboard.plugins import projector

from classes.Constants import DATA_DIR, LOG_DIR, batch_size, EMBEDDING_SIZE
from classes.Utils import Utils


class EmbeddingSaverHook(tf.train.SessionRunHook):

    def __init__(self, values, labels, captions):
        self._saver = None

        self._classes = Utils.get_classnames()

        self._dropout = None
        self._l2_normalized = None
        self._dense3 = None
        self._labels = None

        self._emb_values = values
        self._emb_labels = labels
        self._emb_captions = captions

    def begin(self):
        # self._dropout = tf.get_variable("dropout", [batch_size, 1024])
        # self._l2_normalized = tf.get_variable("l2_normalized", [batch_size, 1024])
        self._dense3 = tf.get_variable("dense3", [batch_size, EMBEDDING_SIZE])

        self._tdense3 = tf.get_default_graph().get_tensor_by_name("dense3/Tanh:0")

        self._labels = tf.get_default_graph().get_tensor_by_name("labels:0")

    def before_run(self, run_context):
        return tf.train.SessionRunArgs([self._tdense3, self._labels])

    def after_run(self, run_context, run_values):
        # metadata_file.write('Name\tClass\n')
        # self._metadata_file.write("".join([('%s\n' % self._classes[x]) for x in run_values[0][0]]))
        self._emb_values.extend(run_values[0][0])
        self._emb_labels.extend(run_values[0][1])
        self._emb_captions.extend([self._classes[x] for x in run_values[0][1]])

    def end(self, session):
        pass

        # writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'projector/'), session.graph)
        #
        # config = projector.ProjectorConfig()
        #
        # embed = config.embeddings.add()
        # embed.tensor_name = self._dense3.name
        # embed.metadata_path = os.path.join(LOG_DIR, 'projector/metadata.tsv')
        #
        # # embed.sprite.image_path = os.path.join(DATA_DIR + '/cifar_10k_sprite.png')
        # # embed.sprite.single_image_dim.extend([img_width, img_height])
        #
        # projector.visualize_embeddings(writer, config)
        #
        # self._saver.save(session, os.path.join(LOG_DIR, "projector/model.ckpt"))
        #
        # writer.close()
