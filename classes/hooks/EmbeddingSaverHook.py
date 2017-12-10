import tensorflow as tf

from classes.Utils import Utils


class EmbeddingSaverHook(tf.train.SessionRunHook):

    def __init__(self, values, labels, captions):
        self._saver = None

        self._classes = Utils.get_classnames()

        self._dense3 = None
        self._labels = None

        self._emb_values = values
        self._emb_labels = labels
        self._emb_captions = captions

    def begin(self):
        self._dense3 = tf.get_default_graph().get_tensor_by_name("dense3/BiasAdd:0")

        self._labels = tf.get_default_graph().get_tensor_by_name("labels:0")

    def before_run(self, run_context):
        return tf.train.SessionRunArgs([self._dense3, self._labels])

    def after_run(self, run_context, run_values):
        self._emb_values.extend(run_values[0][0])
        self._emb_labels.extend(run_values[0][1])
        self._emb_captions.extend([self._classes[x] for x in run_values[0][1]])

    def end(self, session):
        pass
