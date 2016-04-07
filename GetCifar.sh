#!/usr/bin/env sh
# This scripts downloads the CIFAR10 (binary version) data and unzips it.
# Modified from https://github.com/BVLC/caffe/blob/master/data/cifar10/get_cifar10.sh

cd ./datasets/cifar/
rm *

echo "Contains cifar dataset" > readme

echo "Downloading..."

wget --no-check-certificate http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

echo "Unzipping..."

tar -xf cifar-10-python.tar.gz

echo "Done."

cd ../../