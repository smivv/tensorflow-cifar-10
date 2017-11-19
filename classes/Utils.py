import tarfile
import logging
import urllib
import pickle
import numpy
import errno
import tqdm
import os

MODEL = 10

img_width = 32
img_height = 32
num_channels = 3

img_size_flat = img_width * img_height * num_channels

num_classes = 10

_num_files_train = 5

_images_per_file = 10000

_num_images_train = _num_files_train * _images_per_file

DIRECTORY = "/Users/vladimirsmirnov/PycharmProjects/python-tensorflow-cifar-10"

if MODEL == 10:
    DIRECTORY = os.path.join(DIRECTORY, 'cifar-10-batches-py')
else:
    DIRECTORY = os.path.join(DIRECTORY, 'cifar-100-python')


DATA_URL_CIFAR_10 = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
DATA_URL_CIFAR_100 = 'http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'


class Utils:

    @staticmethod
    def load_training_data():
        """
        Load all the training-data for the CIFAR-10 data-set.
        The data-set is split into 5 data-files which are merged here.
        Returns the images, class-numbers and one-hot encoded class-labels.
        """

        # Pre-allocate the arrays for the images and class-numbers for efficiency.
        images = numpy.zeros(shape=[_num_images_train, img_width, img_height, num_channels], dtype=float)
        cls = numpy.zeros(shape=[_num_images_train], dtype=int)

        # Begin-index for the current batch.
        begin = 0

        # For each data-file.
        for i in range(_num_files_train):
            # Load the images and class-numbers from the data-file.
            images_batch, cls_batch = Utils._load_data(filename="data_batch_" + str(i + 1))

            # Number of images in this batch.
            num_images = len(images_batch)

            # End-index for the current batch.
            end = begin + num_images

            # Store the images into the array.
            images[begin:end, :] = images_batch

            # Store the class-numbers into the array.
            cls[begin:end] = cls_batch

            # The begin-index for the next batch is the current end-index.
            begin = end

        return images, cls

    @staticmethod
    def _load_data(filename):
        """
        Load a pickled data-file from the CIFAR-10 data-set
        and return the converted images (see above) and the class-number
        for each image.
        """

        # Load the pickled data-file.
        data = Utils._unpickle(filename)

        # Get the raw images.
        raw = data[b'data']

        # Get the class-numbers for each image. Convert to numpy-array.
        cls = numpy.array(data[b'labels'])

        # Convert the raw images from the data-files to floating-points.
        raw_float = numpy.array(raw, dtype=float) / 255.0

        # Reshape the array to 4-dimensions.
        images = raw_float.reshape([-1, num_channels, img_width, img_height])

        # Reorder the indices of the array.
        images = images.transpose([0, 2, 3, 1])

        return images, cls

    @staticmethod
    def _unpickle(filename):
        """
        Unpickle the given file and return the data.
        Note that the appropriate dir-name is prepended the filename.
        """

        # Create full path for the file.
        filepath = os.path.join(DIRECTORY, filename)

        logging.info("Loading data from: " + filepath)

        with open(filepath, mode='rb') as file:
            # In Python 3.X it is important to set the encoding,
            # otherwise an exception is raised here.
            data = pickle.load(file, encoding='bytes')

        logging.info("Data loaded.")

        return data

    @staticmethod
    def _get_filenames():
        """
        Load the names for the classes in the CIFAR-10 data-set.
        Returns a list with the names. Example: names[3] is the name
        associated with class-number 3.
        """
        return [x.decode('utf-8') for x in Utils._unpickle(filename="batches.meta")[b'label_names']]

    @staticmethod
    def download():
        """
        Uploads and extracts Cifar-10 data to default folder
        """

        if os.path.isdir(DIRECTORY):
            logging.info("Found cifar-{} data in {} folder.".format(MODEL, DIRECTORY))
            return True
        else:
            data_url = DATA_URL_CIFAR_10 if MODEL == 10 else DATA_URL_CIFAR_100

            try:
                os.makedirs(DIRECTORY)
                logging.info("Directory {} created.".format(DIRECTORY))
            except OSError as e:
                if e.errno != errno.EEXIST:
                    return False

            filename = data_url.split('/')[-1]

            filepath = os.path.join(DIRECTORY, filename)

            def hook(t):
                last_b = [0]

                def inner(b, bsize, tsize=None):
                    if tsize is not None:
                        t.total = tsize
                    t.update((b - last_b[0]) * bsize)
                    last_b[0] = b

                return inner

            try:
                logging.info("Dataset uploading to {}".format(DIRECTORY))

                with tqdm.tqdm(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
                    filepath, _ = urllib.request.urlretrieve(data_url, filepath, reporthook=hook(t))

                logging.info("Dataset uploaded to {}".format(DIRECTORY))

                size = os.stat(filepath).st_size

                if size == 0:
                    logging.error("Empty file downloaded..")
                    return False

                logging.info("Dataset filesize = {}".format(size))

            except IOError:
                logging.error("Failed to download {}".format(data_url))
                return False

            tarfile.open(filepath, 'r:gz').extractall(os.path.abspath(os.path.join(DIRECTORY, os.pardir)))
            os.remove(filepath)

            logging.info('Temporary archive deleted..')

            return True
