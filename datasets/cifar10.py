import tarfile
import logging
import urllib
import pickle
import numpy
import errno
import tqdm
import os

IMG_SIZE = 32
NUM_CHANNELS = 3

NUM_CLASSES = 10

NUM_TRAIN_FILES = 5
NUM_TEST_FILES = 1

IMAGES_PER_FILE = 10000

NUM_TRAIN_IMAGES = NUM_TRAIN_FILES * IMAGES_PER_FILE
NUM_TEST_IMAGES = NUM_TEST_FILES * IMAGES_PER_FILE


class Cifar10:

    def __init__(self, data_dir):
        
        self.DATA_DIR = data_dir
        
        self._download_if_needed(data_dir)

    def load_training_data(self):
        """
        Load all the training-data for the CIFAR-10 data-set.
        The data-set is split into 5 data-files which are merged here.
        Returns the images, class-numbers and one-hot encoded class-labels.
        """

        # Pre-allocate the arrays for the images and class-numbers for efficiency.
        images = numpy.zeros(shape=[NUM_TRAIN_IMAGES, IMG_SIZE, IMG_SIZE, NUM_CHANNELS], dtype=float)
        labels = numpy.zeros(shape=[NUM_TRAIN_IMAGES], dtype=int)

        # Begin-index for the current batch.
        begin = 0

        # For each data-file.
        for i in range(NUM_TRAIN_FILES):
            # Load the images and class-numbers from the data-file.
            images_batch, labels_batch = self._load_data(self.DATA_DIR, filename="data_batch_" + str(i + 1))

            # End-index for the current batch.
            end = begin + len(images_batch)

            # Store the images into the array.
            images[begin:end, :] = images_batch

            # Store the class-numbers into the array.
            labels[begin:end] = labels_batch

            # The begin-index for the next batch is the current end-index.
            begin = end

        return images, labels, Cifar10._dense_to_one_hot(labels)

    def load_testing_data(self):
        """
        Load all the testing-data for the CIFAR-10 data-set.
        """
        images, labels = self._load_data(self.DATA_DIR, filename="test_batch")

        return images, labels, Cifar10._dense_to_one_hot(labels)

    def get_classnames(self):
        """
        Load the names for the classes in the CIFAR-10 data-set.
        Returns a list with the names. Example: names[3] is the name
        associated with class-number 3.
        """
        return [x.decode('utf-8') for x in Cifar10._unpickle(self.DATA_DIR, filename="batches.meta")[b'label_names']]

    @staticmethod
    def _dense_to_one_hot(labels):
        num_labels = labels.shape[0]
        index_offset = numpy.arange(num_labels) * NUM_CLASSES
        labels_one_hot = numpy.zeros((num_labels, NUM_CLASSES))
        labels_one_hot.flat[index_offset + labels.ravel()] = 1
        return labels_one_hot

    @staticmethod
    def _load_data(data_dir, filename):
        """
        Load a pickled data-file from the CIFAR-10 data-set
        and return the converted images (see above) and the class-number
        for each image.
        """

        # Load the pickled data-file.
        data = Cifar10._unpickle(data_dir, filename)

        # Get the raw images.
        raw = data[b'data']

        # Get the class-numbers for each image. Convert to numpy-array.
        labels = numpy.array(data[b'labels'])

        # Convert the raw images from the data-files to floating-points.
        raw_float = numpy.array(raw, dtype=float) / 255.0

        # Reshape the array to 4-dimensions.
        images = raw_float.reshape([-1, NUM_CHANNELS, IMG_SIZE, IMG_SIZE])

        # Reorder the indices of the array.
        images = images.transpose([0, 2, 3, 1])

        return images, labels

    @staticmethod
    def _unpickle(data_dir, filename):
        """
        Unpickle the given file and return the data.
        Note that the appropriate dir-name is prepended the filename.
        """

        # Create full path for the file.
        filepath = os.path.join(data_dir, filename)

        logging.info("Loading data from: %s" % filepath)

        with open(filepath, mode='rb') as file:
            # In Python 3.X it is important to set the encoding,
            # otherwise an exception is raised here.
            data = pickle.load(file, encoding='bytes')

        logging.info("Data loaded from: %s" % filepath)

        return data

    @staticmethod
    def _download_if_needed(data_dir):
        """
        Uploads and extracts Cifar-10 data to default folder
        """

        if os.path.isdir(data_dir):
            logging.info("Found cifar-10 data in {} folder.".format(data_dir))
        else:
            data_url = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'

            try:
                os.makedirs(data_dir)
                logging.info("DATA_DIR {} created.".format(data_dir))
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise RuntimeError("Dataset error: " + e.strerror)

            filename = data_url.split('/')[-1]

            filepath = os.path.join(data_dir, filename)

            def hook(t):
                last_b = [0]

                def inner(b, bsize, tsize=None):
                    if tsize is not None:
                        t.total = tsize
                    t.update((b - last_b[0]) * bsize)
                    last_b[0] = b

                return inner

            try:
                logging.info("Dataset uploading to {}".format(data_dir))

                with tqdm.tqdm(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
                    filepath, _ = urllib.request.urlretrieve(data_url, filepath, reporthook=hook(t))

                logging.info("Dataset uploaded to {}".format(data_dir))

                size = os.stat(filepath).st_size

                if size == 0:
                    raise RuntimeError("Empty file downloaded!")

                logging.info("Dataset filesize = {}".format(size))

            except IOError:
                raise RuntimeError("Failed to download {}".format(data_url))

            tarfile.open(filepath, 'r:gz').extractall(os.path.abspath(os.path.join(data_dir, os.pardir)))
            os.remove(filepath)

            logging.info('Temporary archive deleted..')
