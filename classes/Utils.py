import tarfile
import logging
import urllib
import pickle
import numpy
import errno
import tqdm
import os

from classes import Constants


class Utils:

    @staticmethod
    def load_training_data():
        """
        Load all the training-data for the CIFAR-10 data-set.
        The data-set is split into 5 data-files which are merged here.
        Returns the images, class-numbers and one-hot encoded class-labels.
        """

        # Pre-allocate the arrays for the images and class-numbers for efficiency.
        images = numpy.zeros(shape=[Constants.num_images_train, Constants.img_width, Constants.img_height, Constants.num_channels], dtype=float)
        cls = numpy.zeros(shape=[Constants.num_images_train], dtype=int)

        # Begin-index for the current batch.
        begin = 0

        # For each data-file.
        for i in range(Constants.num_train_files):
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
    def load_testing_data():
        """
        Load all the testing-data for the CIFAR-10 data-set.
        """

        return Utils._load_data(filename="test_batch")

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
        images = raw_float.reshape([-1, Constants.num_channels, Constants.img_width, Constants.img_height])

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
        filepath = os.path.join(Constants.DIRECTORY, filename)

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

        if os.path.isdir(Constants.DIRECTORY):
            logging.info("Found cifar-{} data in {} folder.".format(Constants.MODEL, Constants.DIRECTORY))
            return True
        else:
            data_url = Constants.DATA_URL_CIFAR_10 if Constants.MODEL == 10 else Constants.DATA_URL_CIFAR_100

            try:
                os.makedirs(Constants.DIRECTORY)
                logging.info("Directory {} created.".format(Constants.DIRECTORY))
            except OSError as e:
                if e.errno != errno.EEXIST:
                    return False

            filename = data_url.split('/')[-1]

            filepath = os.path.join(Constants.DIRECTORY, filename)

            def hook(t):
                last_b = [0]

                def inner(b, bsize, tsize=None):
                    if tsize is not None:
                        t.total = tsize
                    t.update((b - last_b[0]) * bsize)
                    last_b[0] = b

                return inner

            try:
                logging.info("Dataset uploading to {}".format(Constants.DIRECTORY))

                with tqdm.tqdm(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
                    filepath, _ = urllib.request.urlretrieve(data_url, filepath, reporthook=hook(t))

                logging.info("Dataset uploaded to {}".format(Constants.DIRECTORY))

                size = os.stat(filepath).st_size

                if size == 0:
                    logging.error("Empty file downloaded..")
                    return False

                logging.info("Dataset filesize = {}".format(size))

            except IOError:
                logging.error("Failed to download {}".format(data_url))
                return False

            tarfile.open(filepath, 'r:gz').extractall(os.path.abspath(os.path.join(Constants.DIRECTORY, os.pardir)))
            os.remove(filepath)

            logging.info('Temporary archive deleted..')

            return True
