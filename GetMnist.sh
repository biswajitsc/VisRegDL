#!/usr/bin/env sh
# This scripts downloads the mnist data and unzips it.
# Modified from https://github.com/BVLC/caffe/blob/master/data/mnist/get_mnist.sh

cd datasets
mkdir mnist
cd mnist

echo "Downloading..."

for fname in train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte
do
    if [ ! -e $fname ]; then
        wget --no-check-certificate http://yann.lecun.com/exdb/mnist/${fname}.gz
        gunzip ${fname}.gz
    fi
done

cd ../../