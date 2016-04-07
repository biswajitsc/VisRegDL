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
        self, n_out, h_layers,
        l_drops=None,
        lam=0, Temp=1,
        nonlinearity=L.nonlinearities.sigmoid,
        clam=None
    ):

        np.random.seed(0)

        self.input = T.ftensor4('input')

        self.layers = []

        l_in = L.layers.InputLayer(shape=(None, 3, 32, 32),
                                   input_var=self.input)

        feat_maps = 10
        curr = L.layers.Conv2DLayer(
            l_in, num_filters=feat_maps, filter_size=(3, 3),
            stride=2, pad='same',
            nonlinearity=L.nonlinearities.rectify,
            W=L.init.GlorotUniform())

        curr = L.layers.MaxPool2DLayer(curr, pool_size=(2, 2))

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

        self.reg = T.mean(regs[2] * regs[2])
        for par in regs[3:]:
            self.reg += T.mean(par * par)

        self.loss = L.objectives.categorical_crossentropy(
            self.output, self.target
        )

        l1wts = regs[1].dimshuffle(1, 0).reshape((-1, 1, 8, 8))

        convlen = 3
        cW = [-1 for i in xrange(convlen * convlen)]
        cW[(convlen * convlen - 1) / 2] = convlen * convlen - 1

        cW = np.asarray(cW, dtype=T.config.floatX).reshape(
            1, 1, convlen, convlen)

        cW = TH.shared(cW, name='c')

        clarity = conv.conv2d(l1wts, cW, border_mode='full')
        self.closs = T.mean(clarity * clarity)

        self.loss = T.mean(self.loss)
        if clam is not None:
            self.closs = clam * self.closs
            self.loss += self.closs
        else:
            self.reg += T.mean(regs[1] * regs[1])

        if lam is not None:
            self.reg = self.reg * lam
            self.loss += self.reg

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
        logfile = open(filename + '.txt', 'w')
        logfile.write('tot_loss, c_loss, reg_loss\n\n')
        logfile.flush()

        print "Training ... "

        params = L.layers.get_all_params(self.output_layer, trainable=True)
        # grad = T.grad(self.loss, params)
        # grads = []

        outputs = [self.loss, self.closs, self.reg]
        # outputs.extend(grad)
        inputs = [self.input, self.target]

        updates = L.updates.nesterov_momentum(
            self.loss, params, learning_rate=lrate, momentum=gamma)

        self.trainer = TH.function(
            inputs=inputs, outputs=outputs, updates=updates)

        last_acc = 0.10

        for i in xrange(iters):
            tot_loss = 0.0
            tot_closs = 0.0
            tot_regloss = 0.0
            cnt = 0
            for bx, by in batch_iterable(x, y, batch_size):
                train_outs = self.trainer(bx, by)
                [c_loss, tc_loss, reg_loss] = train_outs[0:3]
                # grad_out = train_outs[3:]

                # if np.random.rand() < 0.001:
                #     for g in grad_out:
                #         grads.extend(np.asarray(g).flatten())

                tot_loss += c_loss
                tot_closs += tc_loss
                tot_regloss += reg_loss
                cnt += 1
            print "Iteration {0}, Loss = {1}, {2}, {3}"\
                .format(i, tot_loss / cnt, tot_closs / cnt, tot_regloss / cnt)
            logfile.write(
                '{} {} {}\n'.format(
                    tot_loss / cnt, tot_closs / cnt, tot_regloss / cnt
                )
            )
            logfile.flush()

            if np.isnan(tot_loss) or np.isnan(tot_closs):
                return last_acc

            if testx is None or testy is None:
                testx = x
                testy = y

            if i % 10 == 0:
                curr_acc = self.test(testx, testy, batch_size)
                last_acc = max(last_acc, curr_acc)
                logfile.write('acc = {}\n'.format(curr_acc))
                logfile.flush()
                self.savemodel(filename)

            if i >= 40 and last_acc <= 0.3:
                return last_acc

            if i % thresh == 0 and i > 0:
                lrate = np.float32(lrate / 10.0)
                updates = L.updates.nesterov_momentum(
                    self.loss, params, learning_rate=lrate, momentum=gamma)

                self.trainer = TH.function(
                    inputs=inputs, outputs=outputs, updates=updates)

            sys.stdout.flush()

        logfile.close()
        # grads = np.histogram(grads, bins=1000)
        # np.savez(filename + '_grads.npz', grads)

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
