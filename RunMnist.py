import theano.sandbox.cuda
theano.sandbox.cuda.use('gpu0')

import numpy as np
import cPickle as cP

import theano as TH
import theano.tensor as T

import scipy.misc as sm
import nnet.lasagnenets as LN
import lasagne as L

import datetime
import pandas as pd


# Load training data
train = pd.read_csv('./datasets/mnist/mnist_train.csv')

trainX = train.values[:, 1:]
trainY = train.values[:, 0]
num_examples = trainX.shape[0]
temp = np.zeros((num_examples, 10))
for i in xrange(num_examples):
    temp[i][trainY[i]] = 1

trainY = np.asarray(temp, dtype='float32')
trainX = np.asarray(trainX, dtype='float32') / 255.0

# Load testing data
test = pd.read_csv('./datasets/mnist/mnist_test.csv')

testX = test.values[:, 1:]
testY = test.values[:, 0]
num_examples = testX.shape[0]
temp = np.zeros((num_examples, 10))
for i in xrange(num_examples):
    temp[i][testY[i]] = 1

testY = np.asarray(temp, dtype='float32')
testX = np.asarray(testX, dtype='float32') / 255.0

trainX = trainX.reshape(-1, 784)
testX = testX.reshape(-1, 784)


def main():
    '''
    Creates neural networks with various parameters and trains them.
    '''

    '''
    n_in: input size
    n_out: output size
    h_layer: hidden layer sizes
    l_drops: dropout rates of hidden layers.
        Set as None if dropout not to be used.
    nonlinearity: activation function to be used.
    lam: weight of the L2 regularizer.
        Set as None if L2 regualizer not to be used.
    clam: weight of VR regularizer.
    '''
    ####################################################
    # VR + L2
    nnet = LN.nnet(
        n_in=784,
        n_out=10,
        h_layers=[1000, 1000],
        lam=20,
        nonlinearity=L.nonlinearities.rectify,
        clam=1
    )

    nnet.train(x=trainX, y=trainY, testx=testX, testy=testY,
               lrate=0.1, gamma=0.9, batch_size=100, iters=200,
               thresh=100, filename='runs/Mnist_Vr_L2')

    ####################################################
    # L2
    nnet = LN.nnet(
        n_in=784,
        n_out=10,
        h_layers=[1000, 1000],
        lam=20,
        nonlinearity=L.nonlinearities.rectify,
        clam=None
    )

    nnet.train(x=trainX, y=trainY, testx=testX, testy=testY,
               lrate=0.1, gamma=0.9, batch_size=100, iters=200,
               thresh=100, filename='runs/Mnist_L2')

    ####################################################
    # Vr + L2 + Dr
    nnet = LN.nnet(
        n_in=784,
        n_out=10,
        h_layers=[1000, 1000],
        l_drops=[0.3, 0.3],
        lam=10,
        nonlinearity=L.nonlinearities.rectify,
        clam=1
    )

    nnet.train(x=trainX, y=trainY, testx=testX, testy=testY,
               lrate=0.01, gamma=0.9, batch_size=100, iters=200,
               thresh=100, filename='runs/Mnist_Vr_L2_Dr')

    ####################################################
    # L2 + Dr
    nnet = LN.nnet(
        n_in=784,
        n_out=10,
        h_layers=[1000, 1000],
        l_drops=[0.3, 0.3],
        lam=1,
        nonlinearity=L.nonlinearities.rectify,
        clam=None
    )

    nnet.train(x=trainX, y=trainY, testx=testX, testy=testY,
               lrate=0.01, gamma=0.9, batch_size=100, iters=200,
               thresh=100, filename='runs/Mnist_L2_Dr')

    ####################################################
    # Dr
    nnet = LN.nnet(
        n_in=784,
        n_out=10,
        h_layers=[1000, 1000],
        l_drops=[0.3, 0.3],
        lam=None,
        nonlinearity=L.nonlinearities.rectify,
        clam=None
    )

    nnet.train(x=trainX, y=trainY, testx=testX, testy=testY,
               lrate=0.005, gamma=0.9, batch_size=100, iters=200,
               thresh=100, filename='runs/Mnist_Dr')

    ####################################################
    # Vr + Dr
    nnet = LN.nnet(
        n_in=784,
        n_out=10,
        h_layers=[1000, 1000],
        l_drops=[0.3, 0.3],
        lam=None,
        nonlinearity=L.nonlinearities.rectify,
        clam=1
    )

    nnet.train(x=trainX, y=trainY, testx=testX, testy=testY,
               lrate=0.01, gamma=0.9, batch_size=100, iters=200,
               thresh=100, filename='runs/Mnist_Vr_Dr')
    ####################################################

if __name__ == '__main__':
    main()
