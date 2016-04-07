'''
        CHANGE ALL SUMS TO Mean
        .
        .
        .
        .
        .
        .
        '''

import lasagne as L
import theano as TH
import theano.tensor as T
import theano.printing as PP
import numpy as np
from theano.tensor.nnet import conv
import sys


def batch_iterable(
    x, y, batch_size
):
    size = x.shape[0]
    num_batches = size / batch_size
    for i in xrange(num_batches):
        yield (x[i * batch_size:(i + 1) * batch_size],
               y[i * batch_size:(i + 1) * batch_size])

    if size % batch_size != 0:
        yield (x[num_batches * batch_size:],
               y[num_batches * batch_size:])


class scaled_softmax:

    def __init__(self, temp):
        self.t = temp

    def __call__(self, x):
        return L.nonlinearities.softmax(x / self.t)


class nnet:

    def __init__(
        self, n_in, n_out, h_layers,
        i_drop=None,
        l_drops=None,
        lam=0, Temp=1,
        inp_max=False,
        nonlinearity=L.nonlinearities.sigmoid,
        clam=None
    ):

        if not inp_max:
            self.input = T.fmatrix('input')
        else:
            self.maxinput = TH.shared(np.zeros((1, n_in)), 'input')
            self.input = self.maxinput * TH.shared(np.ones((1, n_in)), 'ones')

        self.layers = []

        l_in = L.layers.InputLayer(shape=(None, n_in), input_var=self.input)

        if i_drop is not None and not inp_max:
            curr = L.layers.DropoutLayer(l_in, p=i_drop, rescale=True)
        else:
            curr = l_in

        for i, j in enumerate(h_layers):
            curr = L.layers.DenseLayer(
                curr, num_units=j,
                nonlinearity=nonlinearity,
                W=L.init.GlorotUniform(
                    gain=(1 if nonlinearity == L.nonlinearities.sigmoid
                          else 'relu')
                ),
                b=L.init.Constant(0.0)
            )
            self.layers.append(curr)
            if not inp_max:
                if l_drops is not None and l_drops[i] is not None:
                    curr = L.layers.DropoutLayer(
                        curr, p=l_drops[i], rescale=True
                    )

        final_nonlinearity = L.nonlinearities.softmax
        if Temp != 1:
            final_nonlinearity = scaled_softmax(Temp)

        self.output_layer = L.layers.DenseLayer(
            curr, num_units=n_out,
            nonlinearity=L.nonlinearities.linear
        )
        self.layers.append(self.output_layer)

        self.output_layer = L.layers.NonlinearityLayer(
            self.output_layer, nonlinearity=final_nonlinearity)

        self.output = L.layers.get_output(self.output_layer)
        self.test_output = L.layers.get_output(
            self.output_layer, deterministic=True
        )
        self.target = T.fmatrix('target')

        regs = L.layers.get_all_params(self.output_layer, regularizable=True)
        self.reg = T.sum(regs[0] * regs[0])
        for par in regs[1:]:
            self.reg += T.sum(par * par)

        self.loss = L.objectives.categorical_crossentropy(
            self.output, self.target
        )

        l1wts = regs[0].dimshuffle(1, 0).reshape((-1, 3, 32, 32))

        convlen = 3
        cW = [-1 for i in xrange(convlen * convlen)]
        cW[(convlen * convlen - 1) / 2] = convlen * convlen - 1

        cW = np.asarray((cW, cW, cW))

        cW = TH.shared(
            np.asarray(cW, dtype=T.config.floatX).reshape(
                1, 3, convlen, convlen),
            name='cW'
        )

        clarity = conv.conv2d(l1wts, cW, border_mode='full')  # DO Mode = full

        self.loss = T.mean(self.loss) + lam * self.reg
        if clam is not None:
            self.closs = clam * T.sum(clarity * clarity)
            self.loss += self.closs

    def savemodel(self, filename):
        vals = L.layers.get_all_param_values(self.output_layer)
        np.savez(filename, vals)

    def loadmodel(self, filename):
        vals = np.load(filename)['arr_0']
        L.layers.set_all_param_values(self.output_layer, vals)

    def train(
        self, x, y, lrate, gamma, batch_size, iters,
        test_batch=None,
        testx=None, testy=None,
        filename='model.npz',
        lrate_iters=None, lrate_factor=None,
        thresh=500
    ):
        print "Training ... "
        outputs = [self.loss, self.closs]
        inputs = [self.input, self.target]

        params = L.layers.get_all_params(self.output_layer, trainable=True)
        updates = L.updates.nesterov_momentum(
            self.loss, params, learning_rate=lrate, momentum=gamma)

        self.trainer = TH.function(
            inputs=inputs, outputs=outputs, updates=updates)

        last_acc = 0.10

        for i in xrange(iters):
            tot_loss = 0.0
            tot_closs = 0.0
            cnt = 0
            for bx, by in batch_iterable(x, y, batch_size):
                c_loss, tc_loss = self.trainer(bx, by)
                tot_loss += c_loss
                tot_closs += tc_loss
                cnt += 1
            print "Iteration {0}, Loss = {1}, {2}"\
                .format(i, tot_loss / cnt, tot_closs / cnt)

            if np.isnan(tot_loss) or np.isnan(tot_closs):
                return last_acc

            if testx is None or testy is None:
                testx = x
                testy = y

            if i % 10 == 0:
                last_acc = self.test(testx, testy, batch_size)
                self.savemodel(filename)

            if i % thresh == 0 and i > 0:
                lrate = np.float32(lrate / 10.0)
                updates = L.updates.nesterov_momentum(
                    self.loss, params, learning_rate=lrate, momentum=gamma)

                self.trainer = TH.function(
                    inputs=inputs, outputs=outputs, updates=updates)

            sys.stdout.flush()

        return self.test(testx, testy, batch_size)

    def test(
        self, x, y, batch_size
    ):
        print 'Testing ...'

        self.tester = TH.function(
            inputs=[self.input],
            outputs=[self.test_output],
            updates=None
        )

        acc = 0.0
        cnt = 0
        for bx, by in batch_iterable(x, y, batch_size):
            c_out, = self.tester(bx)
            c_acc = np.mean(np.argmax(c_out, axis=1) == np.argmax(by, axis=1))
            acc += c_acc
            cnt += 1

        acc /= cnt
        print 'Mean accuracy = {0}'.format(acc)
        return acc

    def getclass(self, x):
        self.tester = TH.function(
            inputs=[self.input],
            outputs=[self.test_output],
            updates=None
        )

        return np.argmax(self.tester(x.reshape(1, -1))[0][0])

    def setx(self, xval):
        self.maxinput.set_value(xval)

    def getx(self):
        return self.maxinput.get_value()

    def max_inp(self, layer_num, node_num, l_rate, gamma, iters, energy):
        node = L.layers.get_output(
            self.layers[layer_num], deterministic=True
        )[0][node_num]

        grad = T.grad(node, self.maxinput)
        updates = [(self.maxinput, self.maxinput + grad)]

        maxer = TH.function(
            inputs=[],
            outputs=[node],
            updates=updates
        )

        print 'Maximizing ...'
        for i in xrange(iters):
            cval = maxer()
            currx = self.getx()
            currx = currx - (currx < 0) * currx
            assert(np.all(currx >= 0))
            curre = np.sqrt(np.sum(currx * currx))
            if curre > 0:
                self.setx(currx / curre * energy)

        print 'Node value =', cval
        return cval

    def get_layer(self, layer_num, input):
        layer = L.layers.get_output(
            self.layers[layer_num], deterministic=True
        )

        layer_out = TH.function(
            inputs=[self.input],
            outputs=layer,
            updates=[]
        )

        return layer_out(input)
