"""
3D ConvNet layers using Theano, Pylearn and Numpy

ConvLayer: convolutions, filter bank
NormLayer: normalization (LCN, GCN, local mean subtraction)
PoolLayer: pooling, subsampling
RectLayer: rectification (absolute value)

"""

from conv3d2d import conv3d
from maxpool3d import max_pool_3d
from activations import relu, softplus
from logreg import LogisticRegression
from mlp import HiddenLayer

from numpy import sqrt, prod, ones, floor, repeat, pi, exp, zeros, sum
from numpy.random import RandomState

from theano.tensor.nnet import conv2d
from theano import shared, config, _asarray, theano
import theano.tensor as T
import numpy
import sys

floatX = config.floatX


class ConvNet(object):
    def __init__(self, x, y, batch_size, videos, kernels, pools, n_input, n_output, hidden_input, params=None):
        learning_rate = 0.1
        rng = numpy.random.RandomState(1234)

        print '... building the model'
        sys.stdout.flush()

        if not params:
            # Construct the first convolutional pooling layer:
            # filtering reduces the image size to (28-5+1,28-5+1)=(24,24)
            # maxpooling reduces this further to (24/2,24/2) = (12,12)
            # 4D output tensor is thus of shape (batch_size,nkerns[0],12,12)
            layer0 = ConvLayer(x, n_input[0], n_output[0], kernels[0], videos[0], pools[0],
                               batch_size, 'L0', rng)

            layer1 = ConvLayer(layer0.output, n_input[1], n_output[1], kernels[1], videos[1], pools[1],
                               batch_size, 'L1', rng)

            layer2_input = layer1.output.flatten(2)

            # construct a fully-connected sigmoidal layer
            layer2 = HiddenLayer(rng, input=layer2_input, n_in=hidden_input,
                                 n_out=batch_size, activation=T.tanh)

            # classify the values of the fully-connected sigmoidal layer
            layer3 = LogisticRegression(input=layer2.output, n_in=batch_size, n_out=2)
        else:

            layer0 = ConvLayer(x, n_input[0], n_output[0], kernels[0], videos[0], pools[0],
                               batch_size, 'L0', rng, True, params[6], params[7])

            layer1 = ConvLayer(layer0.output, n_input[1], n_output[1], kernels[1], videos[1], pools[1],
                               batch_size, 'L1', rng, True, params[4], params[5])

            layer2_input = layer1.output.flatten(2)

            # construct a fully-connected sigmoidal layer
            layer2 = HiddenLayer(rng, input=layer2_input, n_in=hidden_input,
                                 n_out=batch_size, activation=T.tanh, W=params[2], b=params[3])

            # classify the values of the fully-connected sigmoidal layer
            layer3 = LogisticRegression(input=layer2.output, n_in=batch_size, n_out=2, W=params[0], b=params[1])

        # the cost we minimize during training is the NLL of the model
        cost = layer3.negative_log_likelihood(y)

        # create a list of all model parameters to be fit by gradient descent
        self.params = layer3.params + layer2.params + layer1.params + layer0.params

        # create a list of gradients for all model parameters
        grads = T.grad(cost, self.params)

        # train_model is a function that updates the model parameters by
        # SGD Since this model has many parameters, it would be tedious to
        # manually create an update rule for each model parameter. We thus
        # create the updates list by automatically looping over all
        # (params[i],grads[i]) pairs.
        updates = []
        for param_i, grad_i in zip(self.params, grads):
            updates.append((param_i, param_i - learning_rate * grad_i))

        self.train_model = theano.function([x, y], cost, updates=updates)
        self.validate_model = theano.function(inputs=[x, y], outputs=layer3.errors(y))
        self.predict = theano.function(inputs=[x], outputs=layer3.y_pred)

        print '... building done'
        sys.stdout.flush()

class ConvLayer(object):
    """ Convolutional layer, Filter Bank Layer """

    def __init__(self, input, n_in_maps, n_out_maps, kernel_shape, video_shape, pool_shape,
                 batch_size, layer_name="Conv", rng=RandomState(1234),
                 borrow=True, W=None, b=None):

        """
        video_shape: (frames, height, width)
        kernel_shape: (frames, height, width)
        W_shape: (out, in, kern_frames, kern_height, kern_width)
        """

        # init W
        if W is not None:
            self.W = W
        else:
            # fan in: filter time x filter height x filter width x input maps
            fan_in = prod(kernel_shape) * n_in_maps
            norm_scale = 2. * sqrt(1. / fan_in)
            W_shape = (n_out_maps, n_in_maps) + kernel_shape
            W_val = _asarray(rng.normal(loc=0, scale=norm_scale, size=W_shape), dtype=floatX)
            self.W = shared(value=W_val, borrow=borrow, name=layer_name + '_W')

        # init bias
        if b is not None:
            self.b = b
        else:
            b_val = zeros((n_out_maps,), dtype=floatX)
            self.b = shared(b_val, name=layer_name + "_b", borrow=borrow)

        self.params = [self.W, self.b]

        # 3D convolution; dimshuffle: last 3 dimensions must be (in, h, w)
        n_fr, h, w = video_shape
        n_fr_k, h_k, w_k = kernel_shape
        signals = input.dimshuffle([0, 2, 1, 3, 4])
        out = conv3d(
            signals=signals,
            filters=self.W,
            signals_shape=(batch_size, n_fr, n_in_maps, h, w),
            filters_shape=(n_out_maps, n_fr_k, n_in_maps, h_k, w_k),
            border_mode='valid').dimshuffle([0, 2, 1, 3, 4])

        pooled_out = max_pool_3d(out, pool_shape, ignore_border=True)
        pooled_out += self.b.dimshuffle('x', 0, 'x', 'x', 'x')

        self.output = T.tanh(pooled_out)


class NormLayer(object):
    """ Normalization layer """

    def __init__(self, input, method="lcn", **kwargs):
        """
        method: "lcn", "gcn", "mean"

        LCN: local contrast normalization
            kwargs: 
                kernel_size=9, threshold=1e-4, use_divisor=True

        GCN: global contrast normalization
            kwargs:
                scale=1., subtract_mean=True, use_std=False, sqrt_bias=0., 
                min_divisor=1e-8

        MEAN: local mean subtraction
            kwargs:
                kernel_size=5
        """

        input_shape = input.shape

        # make 4D tensor out of 5D tensor -> (n_images, 1, height, width)
        input_shape_4D = (input_shape[0] * input_shape[1] * input_shape[2], 1,
                          input_shape[3], input_shape[4])
        input_4D = input.reshape(input_shape_4D, ndim=4)
        if method == "lcn":
            out = self.lecun_lcn(input_4D, **kwargs)
        elif method == "gcn":
            out = self.global_contrast_normalize(input_4D, **kwargs)
        elif method == "mean":
            out = self.local_mean_subtraction(input_4D, **kwargs)
        else:
            raise NotImplementedError()

        self.output = out.reshape(input_shape)

    def lecun_lcn(self, X, kernel_size=7, threshold=1e-4, use_divisor=False):
        """
        Yann LeCun's local contrast normalization
        Orginal code in Theano by: Guillaume Desjardins
        """

        filter_shape = (1, 1, kernel_size, kernel_size)
        filters = gaussian_filter(kernel_size).reshape(filter_shape)
        filters = shared(_asarray(filters, dtype=floatX), borrow=True)

        convout = conv2d(X, filters=filters, filter_shape=filter_shape,
                         border_mode='full')

        # For each pixel, remove mean of kernel_sizexkernel_size neighborhood
        mid = int(floor(kernel_size / 2.))
        new_X = X - convout[:, :, mid:-mid, mid:-mid]

        if use_divisor:
            # Scale down norm of kernel_sizexkernel_size patch
            sum_sqr_XX = conv2d(T.sqr(T.abs_(X)), filters=filters,
                                filter_shape=filter_shape, border_mode='full')

            denom = T.sqrt(sum_sqr_XX[:, :, mid:-mid, mid:-mid])
            per_img_mean = denom.mean(axis=[2, 3])
            divisor = T.largest(per_img_mean.dimshuffle(0, 1, 'x', 'x'), denom)
            divisor = T.maximum(divisor, threshold)

            new_X /= divisor

        return new_X  #T.cast(new_X, floatX)

    def local_mean_subtraction(self, X, kernel_size=5):

        filter_shape = (1, 1, kernel_size, kernel_size)
        filters = mean_filter(kernel_size).reshape(filter_shape)
        filters = shared(_asarray(filters, dtype=floatX), borrow=True)

        mean = conv2d(X, filters=filters, filter_shape=filter_shape,
                      border_mode='full')
        mid = int(floor(kernel_size / 2.))

        return X - mean[:, :, mid:-mid, mid:-mid]

    def global_contrast_normalize(self, X, scale=1., subtract_mean=True,
                                  use_std=False, sqrt_bias=0., min_divisor=1e-8):

        ndim = X.ndim
        if not ndim in [3, 4]: raise NotImplementedError("X.dim>4 or X.ndim<3")

        scale = float(scale)
        mean = X.mean(axis=ndim - 1)
        new_X = X.copy()

        if subtract_mean:
            if ndim == 3:
                new_X = X - mean[:, :, None]
            else:
                new_X = X - mean[:, :, :, None]

        if use_std:
            normalizers = T.sqrt(sqrt_bias + X.var(axis=ndim - 1)) / scale
        else:
            normalizers = T.sqrt(sqrt_bias + (new_X ** 2).sum(axis=ndim - 1)) / scale

        # Don't normalize by anything too small.
        T.set_subtensor(normalizers[(normalizers < min_divisor).nonzero()], 1.)

        if ndim == 3:
            new_X /= normalizers[:, :, None]
        else:
            new_X /= normalizers[:, :, :, None]

        return new_X


class PoolLayer(object):
    """ Subsampling and pooling layer """

    def __init__(self, input, pool_shape, method="max"):
        """
        method: "max", "avg", "L2", "L4", ...
        """

        self.__dict__.update(locals())
        del self.self

        if method == "max":
            out = max_pool_3d(input, pool_shape)
        else:
            raise NotImplementedError()

        self.output = out


class RectLayer(object):
    """  Rectification layer """

    def __init__(self, input):
        self.output = T.abs_(input)


def gaussian_filter(kernel_shape):
    x = zeros((kernel_shape, kernel_shape), dtype='float32')

    def gauss(x, y, sigma=2.0):
        Z = 2 * pi * sigma ** 2
        return 1. / Z * exp(-(x ** 2 + y ** 2) / (2. * sigma ** 2))

    mid = floor(kernel_shape / 2.)
    for i in xrange(0, kernel_shape):
        for j in xrange(0, kernel_shape):
            x[i, j] = gauss(i - mid, j - mid)

    return x / sum(x)


def mean_filter(kernel_size):
    s = kernel_size ** 2
    x = repeat(1. / s, s).reshape((kernel_size, kernel_size))
    return x