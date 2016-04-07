import theano.sandbox.cuda
theano.sandbox.cuda.use('gpu1')

import numpy as np
import cPickle as cP

import theano as TH
import theano.tensor as T

import scipy.misc as sm
import nnet.lasagnenets as LN
import lasagne as L

import datetime
import pandas as pd


train = pd.read_csv('../Datasets/mnist/mnist_train.csv')

trainX = train.values[:, 1:]
trainY = train.values[:, 0]
num_examples = trainX.shape[0]
temp = np.zeros((num_examples, 10))
for i in xrange(num_examples):
    temp[i][trainY[i]] = 1

trainY = np.asarray(temp, dtype='float32')
trainX = np.asarray(trainX, dtype='float32') / 255.0


test = pd.read_csv('../Datasets/mnist/mnist_test.csv')

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
    # nnet = LN.nnet(
    #     n_in=784,
    #     n_out=10,
    #     h_layers=[1000, 1000],
    #     #     i_drop=0.3,
    #     # l_drops=[0.3, 0.3],
    #     lam=20,
    #     Temp=1,
    #     nonlinearity=L.nonlinearities.rectify,
    #     clam=1
    # )

    # nnet.train(x=trainX, y=trainY, testx=testX, testy=testY,
    #            lrate=0.1, gamma=0.9, batch_size=100, iters=200,
    #            thresh=100, filename='runs/Mnist_Vl_L2.npz')

    # nnet = LN.nnet(
    #     n_in=784,
    #     n_out=10,
    #     h_layers=[1000, 1000],
    #     #     i_drop=0.3,
    #     # l_drops=[0.3, 0.3],
    #     lam=20,
    #     Temp=1,
    #     nonlinearity=L.nonlinearities.rectify,
    #     clam=None
    # )

    # nnet.train(x=trainX, y=trainY, testx=testX, testy=testY,
    #            lrate=0.1, gamma=0.9, batch_size=100, iters=200,
    #            thresh=100, filename='runs/Mnist_L2.npz')

    # nnet = LN.nnet(
    #     n_in=784,
    #     n_out=10,
    #     h_layers=[1000, 1000],
    #     #     i_drop=0.3,
    #     l_drops=[0.3, 0.3],
    #     lam=10,
    #     Temp=1,
    #     nonlinearity=L.nonlinearities.rectify,
    #     clam=1
    # )

    # nnet.train(x=trainX, y=trainY, testx=testX, testy=testY,
    #            lrate=0.01, gamma=0.9, batch_size=100, iters=200,
    #            thresh=100, filename='runs/Mnist_Vl_L2_Dr.npz')

    # nnet = LN.nnet(
    #     n_in=784,
    #     n_out=10,
    #     h_layers=[1000, 1000],
    #     #     i_drop=0.3,
    #     l_drops=[0.3, 0.3],
    #     lam=1,
    #     Temp=1,
    #     nonlinearity=L.nonlinearities.rectify,
    #     clam=None
    # )

    # nnet.train(x=trainX, y=trainY, testx=testX, testy=testY,
    #            lrate=0.01, gamma=0.9, batch_size=100, iters=200,
    #            thresh=100, filename='runs/Mnist_L2_Dr.npz')

    nnet = LN.nnet(
        n_in=784,
        n_out=10,
        h_layers=[1000, 1000],
        #     i_drop=0.3,
        l_drops=[0.3, 0.3],
        lam=None,
        Temp=1,
        nonlinearity=L.nonlinearities.rectify,
        clam=None
    )

    nnet.train(x=trainX, y=trainY, testx=testX, testy=testY,
               lrate=0.005, gamma=0.9, batch_size=100, iters=200,
               thresh=100, filename='runs/Mnist_Dr.npz')

    # nnet = LN.nnet(
    #     n_in=784,
    #     n_out=10,
    #     h_layers=[1000, 1000],
    #     #     i_drop=0.3,
    #     l_drops=[0.3, 0.3],
    #     lam=None,
    #     Temp=1,
    #     nonlinearity=L.nonlinearities.rectify,
    #     clam=1
    # )

    # nnet.train(x=trainX, y=trainY, testx=testX, testy=testY,
    #            lrate=0.01, gamma=0.9, batch_size=100, iters=200,
    #            thresh=100, filename='runs/Mnist_Vl_Dr.npz')


if __name__ == '__main__':
    main()