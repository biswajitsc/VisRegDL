# VisRegDL
Implementation of the visualization regularizer for neural networks trained on image tasks.

Instructions
============

Pre-requisites
--------------
The following python packages are required to be installed:
* Theano: https://github.com/Theano/Theano
* Lasagne: https://lasagne.readthedocs.org/en/latest/

Downloading and pre-processing the datasets
-------------------------------------------
* Run the scripts ```GetMnist.sh``` and ```GetCifar.sh``` to download the MNIST and CIFAR-10 datasets.
* Run ```ConvertMnist.py``` to pre-process the MNIST dataset into an usable format.

Training
--------
Scripts ```RunMnist.py``` and ```RunCifarCnn.py``` train neural networks with various parameter combinations and report the accuracies and loss values per epoch in their respective log files in the ```runs``` folder.
