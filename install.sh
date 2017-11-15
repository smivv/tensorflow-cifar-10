#!/usr/bin/env bash

# virtual environment installation
virtualenv -p python3 .env
source .env/bin/activate

# installation requirements
pip install numpy
pip install tensorflow
deactivate

# get CIFAR10 dataset
wget http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xzvf cifasr-10-python.tar.gz
rm cifar-10-python.tar.gz

