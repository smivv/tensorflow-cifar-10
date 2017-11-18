#!/usr/bin/env bash

# virtual environment installation
virtualenv -p python3 .env
source .env/bin/activate

# installation dependencies
pip install tensorflow
pip install tqdm
deactivate

# get CIFAR10 dataset, now it's ambiguous
wget http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xzvf cifar-10-python.tar.gz
rm cifar-10-python.tar.gz