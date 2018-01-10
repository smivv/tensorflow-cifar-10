import tensorflow as tf


class SessionHook(tf.train.SessionRunHook):

    def __init__(self, scopes):
        super(SessionHook, self).__init__()

        self.iterator_initializer_func = None

        self._tensors = None

        values = scopes['net_scope'] + '/' + scopes['emb_scope'] + '/' + scopes['emb_name'] + '/BiasAdd:0'
        labels = scopes['metrics_scope'] + '/' + scopes['label_name'] + ':0'

        self._tensor_names = [values, labels]
        self._embeddings = [[], []]

    def begin(self):
        self._tensors = [tf.get_default_graph().get_tensor_by_name(x) for x in self._tensor_names]

    def after_create_session(self, session, coord):
        """ Initialise the iterator after the session has been created."""
        self.iterator_initializer_func(session)

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

    def set_iterator_initializer(self, fun):
        self.iterator_initializer_func = fun
