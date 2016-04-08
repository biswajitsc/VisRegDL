# import theano.sandbox.cuda
# theano.sandbox.cuda.use('gpu0')

import numpy as np
import cPickle as cP

import theano as TH
import theano.tensor as T

import scipy.misc as sm
import nnet.lasagnenetsCFCNN as LN
import lasagne as L

import datetime


def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict


trainX1 = unpickle('./datasets/cifar/cifar-10-batches-py/data_batch_1')
trainX2 = unpickle('./datasets/cifar/cifar-10-batches-py/data_batch_2')
trainX3 = unpickle('./datasets/cifar/cifar-10-batches-py/data_batch_3')
trainX4 = unpickle('./datasets/cifar/cifar-10-batches-py/data_batch_4')
trainX5 = unpickle('./datasets/cifar/cifar-10-batches-py/data_batch_5')
testX = unpickle('./datasets/cifar/cifar-10-batches-py/test_batch')

labelname = unpickle('./datasets/cifar/cifar-10-batches-py/batches.meta')

trainX = np.vstack((trainX1['data'], trainX2['data'], trainX3[
                   'data'], trainX4['data'], trainX5['data']))
trainY = np.hstack((trainX1['labels'], trainX2['labels'], trainX3[
                   'labels'], trainX4['labels'], trainX5['labels']))

testX, testY = testX['data'], testX['labels']
testY = np.asarray(testY)

temp = np.zeros((trainY.shape[0], 10), dtype=np.float32)
for i in xrange(trainY.shape[0]):
    temp[i][trainY[i]] = 1

trainY = temp

temp = np.zeros((testY.shape[0], 10), dtype=np.float32)
for i in xrange(testY.shape[0]):
    temp[i][testY[i]] = 1

testY = temp

trainX = np.asarray(trainX / 256.0, dtype=np.float32)
testX = np.asarray(testX / 256.0, dtype=np.float32)

trainX = trainX.reshape(-1, 3, 32, 32)
testX = testX.reshape(-1, 3, 32, 32)


def main():
    '''
    Creates neural networks with various parameters and trains them.
    '''

    '''
    n_out: output size
    h_layer: hidden layer sizes
    l_drops: dropout rates of hidden layers.
        Set as None if dropout not to be used.
    nonlinearity: activation function to be used.
    lam: weight of the L2 regularizer.
        Set as None if L2 regualizer not to be used.
    clam: weight of VR regularizer.

    Input size has been hardcoded to (3, 32, 32).
    '''
    ####################################################
    # VR + L2
    nnet = LN.nnet(
        n_out=10,
        h_layers=[1000, 1000],
        lam=500,
        Temp=1,
        nonlinearity=L.nonlinearities.rectify,
        clam=20
    )

    nnet.train(x=trainX, y=trainY, testx=testX, testy=testY,
               lrate=0.01, gamma=0.9, batch_size=100, iters=200,
               thresh=70, filename='runs/Cifar_Vr_L2')

    ####################################################
    # L2
    nnet = LN.nnet(
        n_out=10,
        h_layers=[1000, 1000],
        lam=500,
        Temp=1,
        nonlinearity=L.nonlinearities.rectify,
        clam=None
    )

    nnet.train(x=trainX, y=trainY, testx=testX, testy=testY,
               lrate=0.01, gamma=0.9, batch_size=100, iters=200,
               thresh=70, filename='runs/Cifar_L2')

    ####################################################
    # Vr + L2 + Dr
    nnet = LN.nnet(
        n_out=10,
        h_layers=[1000, 1000],
        l_drops=[0.3, 0.3],
        lam=10,
        Temp=1,
        nonlinearity=L.nonlinearities.rectify,
        clam=300
    )

    nnet.train(x=trainX, y=trainY, testx=testX, testy=testY,
               lrate=0.005, gamma=0.9, batch_size=100, iters=200,
               thresh=70, filename='runs/Cifar_Vr_L2_Dr')

    ####################################################
    # L2 + Dr
    nnet = LN.nnet(
        n_out=10,
        h_layers=[1000, 1000],
        l_drops=[0.3, 0.3],
        lam=10,
        Temp=1,
        nonlinearity=L.nonlinearities.rectify,
        clam=None
    )

    nnet.train(x=trainX, y=trainY, testx=testX, testy=testY,
               lrate=0.001, gamma=0.9, batch_size=100, iters=200,
               thresh=70, filename='runs/Cifar_L2_Dr')

    ####################################################
    # Dr
    nnet = LN.nnet(
        n_out=10,
        h_layers=[1000, 1000],
        l_drops=[0.3, 0.3],
        lam=None,
        Temp=1,
        nonlinearity=L.nonlinearities.rectify,
        clam=None
    )

    nnet.train(x=trainX, y=trainY, testx=testX, testy=testY,
               lrate=0.001, gamma=0.9, batch_size=100, iters=200,
               thresh=70, filename='runs/Cifar_Dr')

    ####################################################
    # Vr + Dr
    nnet = LN.nnet(
        n_out=10,
        h_layers=[1000, 1000],
        l_drops=[0.3, 0.3],
        lam=None,
        Temp=1,
        nonlinearity=L.nonlinearities.rectify,
        clam=100
    )

    nnet.train(x=trainX, y=trainY, testx=testX, testy=testY,
               lrate=0.005, gamma=0.9, batch_size=100, iters=200,
               thresh=70, filename='runs/Cifar_Vr_Dr')
    ####################################################

if __name__ == '__main__':
    main()
