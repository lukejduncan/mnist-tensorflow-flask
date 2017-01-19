#!/bin/sh

# This script is only intended to be run once.

# Download the mnist data set
mkdir data

wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz -O data/train-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz -O data/train-labels-idx1-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz -O data/t10k-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz -O data/t10k-labels-idx1-ubyte.gz 

# Then unzip them
gunzip data/*.gz
