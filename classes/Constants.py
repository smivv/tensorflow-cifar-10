import os


MODEL = 10

EMBEDDING_SIZE = 3

start_learning_rate = 0.001

img_width = 32
img_height = 32
num_channels = 3

img_size_flat = img_width * img_height * num_channels

num_classes = 10

max_steps = 100
batch_size = 100

num_train_files = 5
num_test_files = 1

images_per_file = 10000

num_train_images = num_train_files * images_per_file
num_test_images = num_test_files * images_per_file

DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if MODEL == 10:
    DATA_DIR = os.path.join(DIR, 'cifar-10-batches-py')
else:
    DATA_DIR = os.path.join(DIR, 'cifar-100-python')

LOG_DIR = os.path.join(DIR, 'log/1/')

DATA_URL_CIFAR_10 = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
DATA_URL_CIFAR_100 = 'http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'