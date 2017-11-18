import tarfile
import logging
import urllib
import errno
import tqdm
import os

MODEL = 10

# TODO: To think about that approach and find more appropriate way
DIRECTORY = "/Users/vladimirsmirnov/PycharmProjects/python-tensorflow-cifar-10/"

DATA_URL_CIFAR_10 = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
DATA_URL_CIFAR_100 = 'http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'


class Utils:

    @staticmethod
    def download():
        """
        Uploads and extracts Cifar-10 data to default folder
        """

        if MODEL == 10:
            cifar_foldername = 'cifar-10-batches-py'
        else:
            cifar_foldername = 'cifar-100-python'

        cifar_foldername = os.path.join(DIRECTORY, cifar_foldername)

        if os.path.isdir(cifar_foldername):
            logging.info("Found cifar-{} data in {} folder.".format(MODEL, cifar_foldername))
            return True
        else:
            data_url = DATA_URL_CIFAR_10 if MODEL == 10 else DATA_URL_CIFAR_100

            try:
                os.makedirs(cifar_foldername)
                logging.info("Directory {} created.".format(cifar_foldername))
            except OSError as e:
                if e.errno != errno.EEXIST:
                    return False

            filename = data_url.split('/')[-1]

            filepath = os.path.join(cifar_foldername, filename)

            def hook(t):
                last_b = [0]

                def inner(b, bsize, tsize=None):
                    if tsize is not None:
                        t.total = tsize
                    t.update((b - last_b[0]) * bsize)
                    last_b[0] = b

                return inner

            try:
                with tqdm.tqdm(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
                    logging.info("Dataset uploading to {}".format(cifar_foldername))
                    filepath, _ = urllib.request.urlretrieve(data_url, filepath, reporthook=hook(t))
                    logging.info("Dataset uploaded to {}".format(cifar_foldername))

                size = os.stat(filepath).st_size

                if size == 0:
                    logging.error("Empty file downloaded..")
                    return False

                logging.info("Dataset filesize = {}".format(size))

            except IOError:
                logging.error("Failed to download {}".format(data_url))
                return False

            tarfile.open(filepath, 'r:gz').extractall(DIRECTORY)
            os.remove(filepath)

            logging.info('Temporary archive deleted..')

            return True
