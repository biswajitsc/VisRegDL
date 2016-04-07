import theano as TH
from theano import tensor as T

import numpy as np

floattype = TH.config.floatX


class default_filler:

    def getarray(self, *args):
        return np.random.uniform(0, 0.001, args).astype(floattype)


def relu(x):
    return T.switch(x < 0, 0, x)


def threshup(x):
    return T.switch(x > 1, 1, x)


def renormalize(x):
    min_val = np.min(x)
    max_val = np.max(x)
    if min_val < 0 or max_val > 1:
        x = (x - min_val) / (max_val - min_val)
    return x


def threshold(x):
    return [min(max(0, i), 1) for i in x]

epsilon = np.float32(1e-6)


class mlp:

    def __init__(
        self, n_in, n_out, h_layers,
        filler=None,
        lam=0, t=1,
        nlinear=T.nnet.sigmoid,
        inp_max=False
    ):

        if filler is None:
            filler = default_filler()

        self.y = T.fmatrix('y')
        self.lrate = T.scalar('lrate')
        self.gamma = T.scalar('gamma')

        if inp_max:
            self.inp_max = True
            self.x = TH.shared(np.zeros(n_in, dtype=floattype), 'x')
        else:
            self.inp_max = False
            self.x = T.fmatrix('x')

        self.params = []
        self.vs = []

        self.maxers = []
        self.layerfuncs = []

        if len(h_layers) <= 0:
            raise TypeError('h_layers must be non-empty')

        for i in xrange(len(h_layers)):
            if i == 0:
                num_inp = n_in
            else:
                num_inp = h_layers[i - 1]

            num_out = h_layers[i]

            w = TH.shared(filler.getarray(num_inp, num_out), 'w')
            b = TH.shared(np.ones(num_out, dtype=floattype), 'b')

            if i == 0:
                layer = nlinear(T.dot(self.x, w) + b)
            else:
                layer = nlinear(T.dot(layer, w) + b)

            v1 = TH.shared(np.zeros((num_inp, num_out), dtype=floattype), 'v1')
            v2 = TH.shared(np.zeros(num_out, dtype=floattype), 'v2')

            self.params.extend([w, b])
            self.vs.extend([v1, v2])

            if inp_max:
                self.maxers.append(self.getmaxer(layer))
            else:
                self.layerfuncs.append(self.getlayerfunc(layer))

        w = TH.shared(filler.getarray(h_layers[-1], n_out), 'w')
        b = TH.shared(np.ones(n_out, dtype=floattype), 'b')
        self.final = nlinear(T.dot(layer, w) + b)

        v1 = TH.shared(np.zeros((h_layers[-1], n_out), dtype=floattype), 'v1')
        v2 = TH.shared(np.zeros(n_out, dtype=floattype), 'v2')

        self.params.extend([w, b])
        self.vs.extend([v1, v2])

        if inp_max:
            self.maxers.append(self.getmaxer(self.final))
        else:
            self.layerfuncs.append(self.getlayerfunc(self.final))

        self.t = TH.shared(np.float32(t))
        self.out = T.nnet.softmax(self.final / self.t)

        if inp_max:
            self.maxers.append(self.getmaxer(self.out))
        else:
            self.layerfuncs.append(self.getlayerfunc(self.out))

        if not inp_max:
            self.cost = -T.mean(
                self.y * (T.log(self.out + epsilon)) +
                (1 - self.y) * (T.log(1 - self.out + epsilon))
            )
            self.reg = T.sum([T.sum(p * p) for p in self.params])
            self.cost = self.cost + lam * self.reg
            self.delta = T.grad(self.cost, self.params)

            updates = []
            for param, delta, v in zip(self.params, self.delta, self.vs):
                updates.append(
                    (param, param - self.lrate * delta - self.gamma * v)
                )
            for v, delta in zip(self.vs, self.delta):
                updates.append(
                    (v, self.gamma * v + self.lrate * delta)
                )

            inputs = [self.x, self.y, self.lrate, self.gamma]
            outputs = self.cost

            self.trainer = TH.function(
                inputs=inputs, outputs=outputs, updates=updates
            )

            self.tester = TH.function(inputs=[self.x], outputs=self.out)

        if inp_max:
            self.cost = self.out - self.y
            self.cost = T.mean(self.cost * self.cost)
            self.idelta = T.grad(self.cost, self.x)
            self.maximize_input = TH.function(
                inputs=[self.y, self.lrate], outputs=[self.cost],
                updates=[(self.x, self.x - self.lrate * self.idelta)]
            )

    def getmaxer(self, layer):
        mlayer = T.fmatrix('ml')
        mcost = T.sum(layer * mlayer)
        mdelta = T.grad(mcost, self.x)
        mfunc = TH.function(
            inputs=[mlayer, self.lrate], outputs=[mcost, mdelta],
            updates=[(self.x, self.x + self.lrate * mdelta)]
        )
        return mfunc

    def getlayerfunc(self, layer):
        return TH.function(
            inputs=[self.x], outputs=layer
        )

    def train(self, x, y, lrate, gamma, batch_size, iters,
              test_batch=None,
              testx=None, testy=None,
              filename=None,
              lrate_iters=None, lrate_factor=None,
              ):

        if self.inp_max:
            raise TypeError('This is an input maximizing network')

        if test_batch is None:
            test_batch = batch_size

        if filename is None:
            filename = 'model.npz'

        self.accuracy = self.test(testx, testy, test_batch)

        print "Training ..."
        num_batches = x.shape[0] / batch_size
        for i in xrange(iters):
            cost = 0.0

            for j in xrange(num_batches):
                batchx = x[j * batch_size:(j + 1) * batch_size]
                batchy = y[j * batch_size:(j + 1) * batch_size]
                o1 = self.trainer(batchx, batchy, lrate, gamma)
                cost += o1

            if num_batches * batch_size < x.shape[0]:
                batchx = x[num_batches * batch_size:-1]
                batchy = y[num_batches * batch_size:-1]
                o1 = self.trainer(batchx, batchy, lrate, gamma)
                cost += o1

            print "Iteration ", i, " Cost = ", cost / num_batches

            if testx is None and testy is None:
                testx = x
                testy = y

            acc = self.test(testx, testy, test_batch)
            if acc > self.accuracy:
                print "Saving model ..."
                self.accuracy = acc
                self.savemodel(filename)

            if lrate_iters is not None:
                if (i + 1) % lrate_iters == 0:
                    lrate /= lrate_factor

    def test(self, x, y, batch_size):

        if self.inp_max:
            raise TypeError('This is an input maximizing network')

        print "Testing ... "
        num_batches = x.shape[0] / batch_size
        acc = 0

        for j in xrange(num_batches):
            batchx = x[j * batch_size:(j + 1) * batch_size]
            batchy = y[j * batch_size:(j + 1) * batch_size]
            pred = np.argmax(self.tester(batchx), axis=1)
            gold = np.argmax(batchy, axis=1)
            acc += np.sum(pred == gold)

        if num_batches * batch_size < x.shape[0]:
            batchx = x[num_batches * batch_size:-1]
            batchy = y[num_batches * batch_size:-1]
            pred = np.argmax(self.tester(batchx), axis=1)
            gold = np.argmax(batchy, axis=1)
            acc += np.sum(pred == gold)

        acc = acc * 1.0 / x.shape[0]
        print "Accuracy ", acc

        return acc

    def savemodel(self, filename):

        if self.inp_max:
            raise TypeError('This is an input maximizing network')

        saves = []
        for param in self.params:
            saves.append(param.get_value())
        np.savez(filename, tuple(saves))

    def loadmodel(self, filename):
        weights = np.load(filename)
        weights = weights['arr_0']
        for w, p in zip(weights, self.params):
            p.set_value(w)

    def setx(self, val):

        if not self.inp_max:
            raise TypeError('This is not an input maximizing network')

        self.x.set_value(val)

    def getx(self):

        if not self.inp_max:
            raise TypeError('This is not an input maximizing network')

        return self.x.get_value()

    def maximize(self, layernum, layerv, lrate, iters, energy):

        if not self.inp_max:
            raise TypeError('This is not an input maximizing network')

        print "Maximizing ..."
        cost = 0.0

        for i in xrange(iters):
            cost, delt = self.maxers[layernum](layerv, lrate)
            currx = self.x.get_value()
            curre = np.sqrt(np.sum(currx * currx))
            if curre > energy:
                self.x.set_value(currx / curre * energy)
        print cost
        return cost
