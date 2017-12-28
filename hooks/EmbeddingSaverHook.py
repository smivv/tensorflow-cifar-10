import tensorflow as tf


class EmbeddingSaverHook(tf.train.SessionRunHook):

    def __init__(self):
        self._saver = None

        self._tensors = None

        self.dense = "dense3/BiasAdd:0"
        self.labels = "labels:0"

        self._tensor_names = [self.dense, self.labels]
        self._embeddings = [[], []]

    def begin(self):
        self._tensors = [tf.get_default_graph().get_tensor_by_name(x) for x in self._tensor_names]

    def before_run(self, run_context):
        return tf.train.SessionRunArgs(self._tensors)

    def after_run(self, run_context, run_values):
        self._embeddings[0].extend(run_values[0][0])
        self._embeddings[1].extend(run_values[0][1])

    def end(self, session):
        pass

    def get_embeddings(self):
        return {
            'values': self._embeddings[0],
            'labels': self._embeddings[1],
        }
