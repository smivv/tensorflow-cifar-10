import os


MODEL = 10

img_width = 32
img_height = 32
num_channels = 3

img_size_flat = img_width * img_height * num_channels

num_classes = 10

num_train_files = 5
num_test_files = 1

images_per_file = 10000

num_images_train = num_train_files * images_per_file
num_images_test = num_test_files * images_per_file

DIRECTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if MODEL == 10:
    DIRECTORY = os.path.join(DIRECTORY, 'cifar-10-batches-py')
else:
    DIRECTORY = os.path.join(DIRECTORY, 'cifar-100-python')

DATA_URL_CIFAR_10 = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
DATA_URL_CIFAR_100 = 'http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'