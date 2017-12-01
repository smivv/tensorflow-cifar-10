import os
import tensorflow as tf

from tensorflow.contrib.tensorboard.plugins import projector

from classes.Constants import DATA_DIR, LOG_DIR
from classes.Utils import Utils


class EmbeddingSaverHook(tf.train.SessionRunHook):
    _data = None
    _labels = None

    def __init__(self, data, labels):
        self._data = data
        self._labels = labels
        self._saver = None

    def begin(self):
        self._saver = tf.train.Saver([tf.get_variable("dropout", [100, 1024])])
        pass

    def before_run(self, run_context):
        pass

    def after_run(self, run_context, run_values):
        pass

    def end(self, session):

        writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'projector'), session.graph)

        config = projector.ProjectorConfig()
        embed = config.embeddings.add()
        embed.tensor_name = 'dropout'

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

    """--------------------------------------------------------------------------------------------------------------"""

    def generate_embeddings(self, images):
        # Import data
        # mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True, fake_data=FLAGS.fake_data)
        sess = tf.InteractiveSession()

        # Input set for Embedded TensorBoard visualization
        # Performed with cpu to conserve memory and processing power
        with tf.device("/cpu:0"):
            embedding = tf.Variable(self._data, trainable=False, name='embedding')

        sess.run(embedding.initializer)

        writer = tf.summary.FileWriter(LOG_DIR + '/projector', sess.graph)

        config = projector.ProjectorConfig()
        embed = config.embeddings.add()
        embed.tensor_name = embedding.name

        embed.metadata_path = os.path.join(LOG_DIR + '/projector/metadata.tsv')

        embed.sprite.image_path = os.path.join(DATA_DIR + '/cifar_10k_sprite.png')
        embed.sprite.single_image_dim.extend([img_width, img_height])

        projector.visualize_embeddings(writer, config)

        saver = tf.train.Saver([embedding])
        saver.save(sess, os.path.join(LOG_DIR, 'projector/a_model.ckpt'))


    def create_sprite_image(images):
        """Returns a sprite image consisting of images passed as argument. Images should be count x width x height"""
        if isinstance(images, list):
            images = np.array(images)
        img_h = images.shape[1]
        img_w = images.shape[2]
        n_plots = int(np.ceil(np.sqrt(images.shape[0])))

        spriteimage = np.ones((img_h * n_plots, img_w * n_plots))

        for i in range(n_plots):
            for j in range(n_plots):
                this_filter = i * n_plots + j
                if this_filter < images.shape[0]:
                    this_img = images[this_filter]
                    spriteimage[i * img_h:(i + 1) * img_h,
                    j * img_w:(j + 1) * img_w] = this_img

        return spriteimage


    def vector_to_matrix_mnist(mnist_digits):
        """Reshapes normal mnist digit (batch,28*28) to matrix (batch,28,28)"""
        return np.reshape(mnist_digits, (-1, 28, 28))


    def invert_grayscale(mnist_digits):
        """ Makes black white, and white black """
        return 1 - mnist_digits