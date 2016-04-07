#!/usr/bin/env sh
# This scripts downloads the CIFAR10 (binary version) data and unzips it.
# Modified from https://github.com/BVLC/caffe/blob/master/data/cifar10/get_cifar10.sh

cd ./datasets/
mkdir cifar
cd cifar

echo "Downloading..."

wget --no-check-certificate http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz

echo "Unzipping..."

tar -xf cifar-10-binary.tar.gz && rm -f cifar-10-binary.tar.gz
mv cifar-10-batches-bin/* . && rm -rf cifar-10-batches-bin

echo "Done."

cd ../../