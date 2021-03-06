# -*- coding: utf-8 -*-
from __future__ import absolute_import
import theano
import theano.tensor as T
import numpy as np

from .. import activations, initializations
from ..utils.theano_utils import shared_scalar, shared_zeros, alloc_zeros_matrix
from ..utils.theano_utils import sharedX, shared_ones
from ..layers.core import Layer, MaskedLayer
from six.moves import range

#from keras.models_xd import Sequential
from keras.layers.core import Dense
from keras.regularizers import l2

class Recurrent(MaskedLayer):
    input_ndim = 3

    def get_output_mask(self, train=None):
        if self.return_sequences:
            return super(Recurrent, self).get_output_mask(train)
        else:
            return None

    def get_padded_shuffled_mask(self, train, X, pad=0):
        mask = self.get_input_mask(train)
        if mask is None:
            mask = T.ones_like(X.sum(axis=-1))  # is there a better way to do this without a sum?

        # mask is (nb_samples, time)
        mask = T.shape_padright(mask)  # (nb_samples, time, 1)
        mask = T.addbroadcast(mask, -1)  # the new dimension (the '1') is made broadcastable
        # see http://deeplearning.net/software/theano/library/tensor/basic.html#broadcasting-in-theano-vs-numpy
        mask = mask.dimshuffle(1, 0, 2)  # (time, nb_samples, 1)

        if pad > 0:
            # left-pad in time with 0
            padding = alloc_zeros_matrix(pad, mask.shape[1], 1)
            mask = T.concatenate([padding, mask], axis=0)
        return mask.astype('int8')

    @property
    def output_shape(self):
        input_shape = self.input_shape
        if self.return_sequences:
            return (input_shape[0], input_shape[1], self.output_dim)
        else:
            return (input_shape[0], self.output_dim)


class SimpleRNN(Recurrent):
    '''
        Fully connected RNN where output is to fed back to input.

        Not a particularly useful model,
        included for demonstration purposes
        (demonstrates how to use theano.scan to build a basic RNN).
    '''
    def __init__(self, output_dim,
                 init='glorot_uniform', inner_init='orthogonal', activation='sigmoid', weights=None,
                 truncate_gradient=-1, return_sequences=False, input_dim=None, input_length=None, **kwargs):
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.truncate_gradient = truncate_gradient
        self.activation = activations.get(activation)
        self.return_sequences = return_sequences
        self.initial_weights = weights

        self.input_dim = input_dim
        self.input_length = input_length
        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim)
        super(SimpleRNN, self).__init__(**kwargs)

    def build(self):
        input_dim = self.input_shape[2]
        self.input = T.tensor3()

        self.W = self.init((input_dim, self.output_dim))
        self.U = self.inner_init((self.output_dim, self.output_dim))
        self.b = shared_zeros((self.output_dim))
        self.params = [self.W, self.U, self.b]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def _step(self, x_t, mask_tm1, h_tm1, u):
        '''
            Variable names follow the conventions from:
            http://deeplearning.net/software/theano/library/scan.html

        '''
        return self.activation(x_t + mask_tm1 * T.dot(h_tm1, u))

    def get_output(self, train=False):
        X = self.get_input(train)  # shape: (nb_samples, time (padded with zeros), input_dim)
        # new shape: (time, nb_samples, input_dim) -> because theano.scan iterates over main dimension
        padded_mask = self.get_padded_shuffled_mask(train, X, pad=1)
        X = X.dimshuffle((1, 0, 2))
        x = T.dot(X, self.W) + self.b

        # scan = theano symbolic loop.
        # See: http://deeplearning.net/software/theano/library/scan.html
        # Iterate over the first dimension of the x array (=time).
        outputs, updates = theano.scan(
            self._step,  # this will be called with arguments (sequences[i], outputs[i-1], non_sequences[i])
            sequences=[x, dict(input=padded_mask, taps=[-1])],  # tensors to iterate over, inputs to _step
            # initialization of the output. Input to _step with default tap=-1.
            outputs_info=T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1),
            non_sequences=self.U,  # static inputs to _step
            truncate_gradient=self.truncate_gradient)

        if self.return_sequences:
            return outputs.dimshuffle((1, 0, 2))
        return outputs[-1]

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "output_dim": self.output_dim,
                  "init": self.init.__name__,
                  "inner_init": self.inner_init.__name__,
                  "activation": self.activation.__name__,
                  "truncate_gradient": self.truncate_gradient,
                  "return_sequences": self.return_sequences,
                  "input_dim": self.input_dim,
                  "input_length": self.input_length}
        base_config = super(SimpleRNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SimpleDeepRNN(Recurrent):
    '''
        Fully connected RNN where the output of multiple timesteps
        (up to "depth" steps in the past) is fed back to the input:

        output = activation( W.x_t + b + inner_activation(U_1.h_tm1) + inner_activation(U_2.h_tm2) + ... )

        This demonstrates how to build RNNs with arbitrary lookback.
        Also (probably) not a super useful model.
    '''
    def __init__(self, output_dim, depth=3,
                 init='glorot_uniform', inner_init='orthogonal',
                 activation='sigmoid', inner_activation='hard_sigmoid',
                 weights=None, truncate_gradient=-1, return_sequences=False,
                 input_dim=None, input_length=None, **kwargs):
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.truncate_gradient = truncate_gradient
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        self.depth = depth
        self.return_sequences = return_sequences
        self.initial_weights = weights

        self.input_dim = input_dim
        self.input_length = input_length
        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim)
        super(SimpleDeepRNN, self).__init__(**kwargs)

    def build(self):
        input_dim = self.input_shape[2]
        self.input = T.tensor3()
        self.W = self.init((input_dim, self.output_dim))
        self.Us = [self.inner_init((self.output_dim, self.output_dim)) for _ in range(self.depth)]
        self.b = shared_zeros((self.output_dim))
        self.params = [self.W] + self.Us + [self.b]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def _step(self, x_t, *args):
        o = x_t
        for i in range(self.depth):
            mask_tmi = args[i]
            h_tmi = args[i + self.depth]
            U_tmi = args[i + 2*self.depth]
            o += mask_tmi*self.inner_activation(T.dot(h_tmi, U_tmi))
        return self.activation(o)

    def get_output(self, train=False):
        X = self.get_input(train)
        padded_mask = self.get_padded_shuffled_mask(train, X, pad=self.depth)
        X = X.dimshuffle((1, 0, 2))

        x = T.dot(X, self.W) + self.b

        if self.depth == 1:
            initial = T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1)
        else:
            initial = T.unbroadcast(T.unbroadcast(alloc_zeros_matrix(self.depth, X.shape[1], self.output_dim), 0), 2)

        outputs, updates = theano.scan(
            self._step,
            sequences=[x, dict(
                input=padded_mask,
                taps=[(-i) for i in range(self.depth)]
            )],
            outputs_info=[dict(
                initial=initial,
                taps=[(-i-1) for i in range(self.depth)]
            )],
            non_sequences=self.Us,
            truncate_gradient=self.truncate_gradient
        )

        if self.return_sequences:
            return outputs.dimshuffle((1, 0, 2))
        return outputs[-1]

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "output_dim": self.output_dim,
                  "depth": self.depth,
                  "init": self.init.__name__,
                  "inner_init": self.inner_init.__name__,
                  "activation": self.activation.__name__,
                  "inner_activation": self.inner_activation.__name__,
                  "truncate_gradient": self.truncate_gradient,
                  "return_sequences": self.return_sequences,
                  "input_dim": self.input_dim,
                  "input_length": self.input_length}
        base_config = super(SimpleDeepRNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class GRU(Recurrent):
    '''
        Gated Recurrent Unit - Cho et al. 2014

        Acts as a spatiotemporal projection,
        turning a sequence of vectors into a single vector.

        Eats inputs with shape:
        (nb_samples, max_sample_length (samples shorter than this are padded with zeros at the end), input_dim)

        and returns outputs with shape:
        if not return_sequences:
            (nb_samples, output_dim)
        if return_sequences:
            (nb_samples, max_sample_length, output_dim)

        References:
            On the Properties of Neural Machine Translation: Encoder–Decoder Approaches
                http://www.aclweb.org/anthology/W14-4012
            Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling
                http://arxiv.org/pdf/1412.3555v1.pdf
    '''
    def __init__(self, output_dim,
                 init='glorot_uniform', inner_init='orthogonal',
                 activation='sigmoid', inner_activation='hard_sigmoid',
                 weights=None, truncate_gradient=-1, return_sequences=False,
                 input_dim=None, input_length=None, **kwargs):
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        self.truncate_gradient = truncate_gradient
        self.return_sequences = return_sequences
        self.initial_weights = weights

        self.input_dim = input_dim
        self.input_length = input_length
        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim)
        super(GRU, self).__init__(**kwargs)

    def build(self):
        input_dim = self.input_shape[2]
        self.input = T.tensor3()

        self.W_z = self.init((input_dim, self.output_dim))
        self.U_z = self.inner_init((self.output_dim, self.output_dim))
        self.b_z = shared_zeros((self.output_dim))

        self.W_r = self.init((input_dim, self.output_dim))
        self.U_r = self.inner_init((self.output_dim, self.output_dim))
        self.b_r = shared_zeros((self.output_dim))

        self.W_h = self.init((input_dim, self.output_dim))
        self.U_h = self.inner_init((self.output_dim, self.output_dim))
        self.b_h = shared_zeros((self.output_dim))

        self.params = [
            self.W_z, self.U_z, self.b_z,
            self.W_r, self.U_r, self.b_r,
            self.W_h, self.U_h, self.b_h,
        ]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def _step(self,
              xz_t, xr_t, xh_t, mask_tm1,
              h_tm1,
              u_z, u_r, u_h):
        h_mask_tm1 = mask_tm1 * h_tm1
        z = self.inner_activation(xz_t + T.dot(h_mask_tm1, u_z))
        r = self.inner_activation(xr_t + T.dot(h_mask_tm1, u_r))
        hh_t = self.activation(xh_t + T.dot(r * h_mask_tm1, u_h))
        h_t = z * h_mask_tm1 + (1 - z) * hh_t
        return h_t

    def get_output(self, train=False):
        X = self.get_input(train)
        padded_mask = self.get_padded_shuffled_mask(train, X, pad=1)
        X = X.dimshuffle((1, 0, 2))

        x_z = T.dot(X, self.W_z) + self.b_z
        x_r = T.dot(X, self.W_r) + self.b_r
        x_h = T.dot(X, self.W_h) + self.b_h
        outputs, updates = theano.scan(
            self._step,
            sequences=[x_z, x_r, x_h, padded_mask],
            outputs_info=T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1),
            non_sequences=[self.U_z, self.U_r, self.U_h],
            truncate_gradient=self.truncate_gradient)

        if self.return_sequences:
            return outputs.dimshuffle((1, 0, 2))
        return outputs[-1]

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "output_dim": self.output_dim,
                  "init": self.init.__name__,
                  "inner_init": self.inner_init.__name__,
                  "activation": self.activation.__name__,
                  "inner_activation": self.inner_activation.__name__,
                  "truncate_gradient": self.truncate_gradient,
                  "return_sequences": self.return_sequences,
                  "input_dim": self.input_dim,
                  "input_length": self.input_length}
        base_config = super(GRU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class LSTM(Recurrent):
    '''
        Acts as a spatiotemporal projection,
        turning a sequence of vectors into a single vector.

        Eats inputs with shape:
        (nb_samples, max_sample_length (samples shorter than this are padded with zeros at the end), input_dim)

        and returns outputs with shape:
        if not return_sequences:
            (nb_samples, output_dim)
        if return_sequences:
            (nb_samples, max_sample_length, output_dim)

        For a step-by-step description of the algorithm, see:
        http://deeplearning.net/tutorial/lstm.html

        References:
            Long short-term memory (original 97 paper)
                http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf
            Learning to forget: Continual prediction with LSTM
                http://www.mitpressjournals.org/doi/pdf/10.1162/089976600300015015
            Supervised sequence labelling with recurrent neural networks
                http://www.cs.toronto.edu/~graves/preprint.pdf
    '''
    def __init__(self, output_dim,
                 init='glorot_uniform', inner_init='orthogonal', forget_bias_init='one',
                 activation='tanh', inner_activation='hard_sigmoid',
                 weights=None, truncate_gradient=-1, return_sequences=True,
                 input_dim=None, input_length=None, **kwargs):
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.forget_bias_init = initializations.get(forget_bias_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        self.truncate_gradient = truncate_gradient
        self.return_sequences = return_sequences
        self.initial_weights = weights

        self.input_dim = input_dim
        self.input_length = input_length
        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim)
        super(LSTM, self).__init__(**kwargs)

    def build(self):
        input_dim = self.input_shape[2]
        self.input = T.tensor3()

        self.W_i = self.init((input_dim, self.output_dim))
        self.U_i = self.inner_init((self.output_dim, self.output_dim))
        self.b_i = shared_zeros((self.output_dim))

        self.W_f = self.init((input_dim, self.output_dim))
        self.U_f = self.inner_init((self.output_dim, self.output_dim))
        self.b_f = self.forget_bias_init((self.output_dim))

        self.W_c = self.init((input_dim, self.output_dim))
        self.U_c = self.inner_init((self.output_dim, self.output_dim))
        self.b_c = shared_zeros((self.output_dim))

        self.W_o = self.init((input_dim, self.output_dim))
        self.U_o = self.inner_init((self.output_dim, self.output_dim))
        self.b_o = shared_zeros((self.output_dim))

        self.params = [
            self.W_i, self.U_i, self.b_i,
            self.W_c, self.U_c, self.b_c,
            self.W_f, self.U_f, self.b_f,
            self.W_o, self.U_o, self.b_o,
        ]

    def _step(self,
              xi_t, xf_t, xo_t, xc_t, mask_tm1,
              h_tm1, c_tm1,
              u_i, u_f, u_o, u_c):
        h_mask_tm1 = mask_tm1 * h_tm1
        c_mask_tm1 = mask_tm1 * c_tm1

        i_t = self.inner_activation(xi_t + T.dot(h_mask_tm1, u_i))
        f_t = self.inner_activation(xf_t + T.dot(h_mask_tm1, u_f))
        c_t = f_t * c_mask_tm1 + i_t * self.activation(xc_t + T.dot(h_mask_tm1, u_c))
        o_t = self.inner_activation(xo_t + T.dot(h_mask_tm1, u_o))
        h_t = o_t * self.activation(c_t)
        return h_t, c_t

    def get_output(self, train=False):
        X = self.get_input(train)
        padded_mask = self.get_padded_shuffled_mask(train, X, pad=1)
        X = X.dimshuffle((1, 0, 2))

        xi = T.dot(X, self.W_i) + self.b_i
        xf = T.dot(X, self.W_f) + self.b_f
        xc = T.dot(X, self.W_c) + self.b_c
        xo = T.dot(X, self.W_o) + self.b_o
            
        [outputs, memories], updates = theano.scan(
            self._step,
            sequences=[xi, xf, xo, xc, padded_mask],
            outputs_info=[
                T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1),
                T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1)
            ],
            non_sequences=[self.U_i, self.U_f, self.U_o, self.U_c],
            truncate_gradient=self.truncate_gradient)
        
        if self.return_sequences:
            return outputs.dimshuffle((1, 0, 2))
        return outputs[-1]

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "output_dim": self.output_dim,
                  "init": self.init.__name__,
                  "inner_init": self.inner_init.__name__,
                  "forget_bias_init": self.forget_bias_init.__name__,
                  "activation": self.activation.__name__,
                  "inner_activation": self.inner_activation.__name__,
                  "truncate_gradient": self.truncate_gradient,
                  "return_sequences": self.return_sequences,
                  "input_dim": self.input_dim,
                  "input_length": self.input_length}
        base_config = super(LSTM, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
class ReducedLSTMOld(Recurrent):
    '''
        Acts as a spatiotemporal projection,
        turning a sequence of vectors into a single vector.

        Eats inputs with shape:
        (nb_samples, max_sample_length (samples shorter than this are padded with zeros at the end), input_dim)

        and returns outputs with shape:
        if not return_sequences:
            (nb_samples, output_dim)
        if return_sequences:
            (nb_samples, max_sample_length, output_dim)

        For a step-by-step description of the algorithm, see:
        http://deeplearning.net/tutorial/lstm.html

        References:
            Long short-term memory (original 97 paper)
                http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf
            Learning to forget: Continual prediction with LSTM
                http://www.mitpressjournals.org/doi/pdf/10.1162/089976600300015015
            Supervised sequence labelling with recurrent neural networks
                http://www.cs.toronto.edu/~graves/preprint.pdf
    '''
    def __init__(self, output_dim,
                 init='glorot_uniform', inner_init='orthogonal', forget_bias_init='one',
                 activation='tanh', inner_activation='hard_sigmoid',
                 weights=None, truncate_gradient=-1, return_sequences=False,
                 input_dim=None, input_length=None, **kwargs):
        self.output_dim = output_dim
        init = 'zero'  #XD
#        inner_init = 'uniform'  #XD
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.forget_bias_init = initializations.get(forget_bias_init)
#        self.activation = activations.get(activation)
        self.activation = activations.get('linear') #XD
#        self.inner_activation = activations.get(inner_activation)
        self.inner_activation = activations.get('sigmoid') #XD
        self.truncate_gradient = truncate_gradient
        self.return_sequences = return_sequences
        self.initial_weights = weights

        self.input_dim = input_dim
        self.input_length = input_length
        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim)
            
        self.set_init_input()  #XD
        super(ReducedLSTMOld, self).__init__(**kwargs)
        
    #XD
#    def add_init_models(self, in_dim):
#        self.hidden_init_model = Sequential()
#        self.hidden_init_model.add(Dense(self.output_dim, input_shape=(in_dim,), 
#                        init='uniform'))
#        self.cell_init_model = Sequential()
#        self.cell_init_model.add(Dense(self.output_dim, input_shape=(in_dim,), 
#                        init='uniform'))
    def set_init_input(self):
        self.hidden0 = T.matrix()
        self.cell0 = T.matrix()
        
    def get_init_input(self):
        return [self.hidden0, self.cell0]

    def build(self):
        input_dim = self.input_shape[2]
        self.input = T.tensor3()

        self.W_i = self.init((input_dim, self.output_dim))
        self.U_i = self.inner_init((self.output_dim, self.output_dim))
        self.b_i = shared_zeros((self.output_dim))

        self.W_f = self.init((input_dim, self.output_dim))
#        self.U_f = self.inner_init((self.output_dim, self.output_dim))
        self.U_f = sharedX(.5 * np.ones((self.output_dim, self.output_dim)))  #XD
#        self.b_f = self.forget_bias_init((self.output_dim))
        self.b_f = sharedX(1. * np.ones((self.output_dim,)))  #XD

        self.W_c = self.init((input_dim, self.output_dim))
#        self.U_c = self.inner_init((self.output_dim, self.output_dim))
        self.U_c = sharedX(-.5 * np.ones((self.output_dim, self.output_dim)))  #XD
        self.b_c = shared_zeros((self.output_dim))

        self.W_o = self.init((input_dim, self.output_dim))
        self.U_o = self.inner_init((self.output_dim, self.output_dim))
        self.b_o = shared_zeros((self.output_dim))

        self.params = [
#            self.W_i, self.U_i, self.b_i,
            self.W_c, self.U_c, self.b_c,
            self.W_f, self.U_f, self.b_f,
#            self.W_o, self.U_o, self.b_o,
        ]
        #XD
#        if hasattr(self, 'hidden_init_model'):
#            self.params += self.hidden_init_model.params
#        if hasattr(self, 'cell_init_model'):
#            self.params += self.cell_init_model.params

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def _step(self,
              xi_t, xf_t, xo_t, xc_t, mask_tm1,
              h_tm1, c_tm1,
              u_i, u_f, u_o, u_c):
#        h_mask_tm1 = mask_tm1 * h_tm1
#        c_mask_tm1 = mask_tm1 * c_tm1
        h_mask_tm1 = h_tm1  #XD
        c_mask_tm1 = c_tm1  #XD

        i_t = self.inner_activation(xi_t + T.dot(h_mask_tm1, u_i))
        i_t = i_t * 0. + 1. # XD: disable input gate
        f_t = self.inner_activation(xf_t + T.dot(h_mask_tm1, u_f))
        b_t = i_t * self.activation(xc_t + T.dot(h_mask_tm1, u_c))
        c_t = f_t * c_mask_tm1 + b_t
        o_t = self.inner_activation(xo_t + T.dot(h_mask_tm1, u_o))
        o_t = o_t * 0. + 1.  # XD: disable output gate
        h_t = o_t * self.activation(c_t)
#        h_t = h_t * (h_t > 0.) #XD
        return h_t, c_t, f_t, b_t

    def get_output(self, train=False):
        X = self.get_input(train)
        padded_mask = self.get_padded_shuffled_mask(train, X, pad=1)
        X = X.dimshuffle((1, 0, 2))

        xi = T.dot(X, self.W_i) + self.b_i
        xf = T.dot(X, self.W_f) + self.b_f
        xc = T.dot(X, self.W_c) + self.b_c
        xo = T.dot(X, self.W_o) + self.b_o

        #XD
        if hasattr(self, 'hidden_init_model'):
            init_hidden = self.hidden_init_model.get_output(train)
        else:
            init_hidden = T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1)
        if hasattr(self, 'cell_init_model'):
            init_cell = self.cell_init_model.get_output(train)
        else:
            init_cell = T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1)
            
        [outputs, memories, forgets, increments], updates = theano.scan(
            self._step,
            sequences=[xi, xf, xo, xc, padded_mask],
#            outputs_info=[
#                T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1),
#                T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1)
#            ],
            outputs_info=[
                self.hidden0,
                self.cell0,
                None,
                None
            ],
            non_sequences=[self.U_i, self.U_f, self.U_o, self.U_c],
            truncate_gradient=self.truncate_gradient)

        #XD
        if train == False:
            self.forgets = forgets.dimshuffle((1, 0, 2))
            self.increments = increments.dimshuffle((1, 0, 2))

        if self.return_sequences:
            return outputs.dimshuffle((1, 0, 2))
        return outputs[-1]

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "output_dim": self.output_dim,
                  "init": self.init.__name__,
                  "inner_init": self.inner_init.__name__,
                  "forget_bias_init": self.forget_bias_init.__name__,
                  "activation": self.activation.__name__,
                  "inner_activation": self.inner_activation.__name__,
                  "truncate_gradient": self.truncate_gradient,
                  "return_sequences": self.return_sequences,
                  "input_dim": self.input_dim,
                  "input_length": self.input_length}
        base_config = super(ReducedLSTMOld, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
 
class RLSTM(Recurrent):
    def __init__(self, input_dim, h0_dim, h1_dim, output_dim, step_type='replace',
                 init='glorot_uniform', inner_init='orthogonal', forget_bias_init='one',
                 activation='tanh', inner_activation='hard_sigmoid', 
                 W_h0_regularizer=None, W_h1_regularizer=None, 
                 weights=None, truncate_gradient=-1, return_sequences=False,
                 input_length=None, **kwargs):
        self.output_dim = output_dim
        self.h0_dim = h0_dim
        self.h1_dim = h1_dim
        self.step_type = step_type
        init = 'zero'  #XD
#        inner_init = 'uniform'  #XD
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.forget_bias_init = initializations.get(forget_bias_init)
#        self.activation = activations.get(activation)
        self.activation = activations.get('linear') #XD
#        self.inner_activation = activations.get(inner_activation)
        self.inner_activation = activations.get('sigmoid') #XD
        self.truncate_gradient = truncate_gradient
        self.return_sequences = return_sequences
        
        self.W_h0_regularizer = W_h0_regularizer
        self.W_h1_regularizer = W_h1_regularizer
        self.initial_weights = weights

        self.input_dim = input_dim
        self.input_length = input_length
        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim)
            
        #XD
        self.set_init_input()  
        
        super(RLSTM, self).__init__(**kwargs)

    def set_init_input(self):
        self.hidden0 = T.tensor3()
        self.cell0 = T.tensor3()
        self.cell_mean = T.tensor3()
        
    def get_init_input(self):
        return [self.hidden0, self.cell0, self.cell_mean]

    def build(self):
        input_dim = self.input_shape[2]
        self.input = T.tensor3()

        self.W_h0 = initializations.get('uniform_small')((input_dim, self.h0_dim))
        self.b_h0 = shared_zeros((self.h0_dim))
        self.W_h1 = initializations.get('uniform_small')((self.h0_dim, self.h1_dim))
        self.b_h1 = shared_zeros((self.h1_dim))
        
        self.W_f = self.init((self.h1_dim, self.output_dim))
        self.b_f = sharedX(1. * np.ones((self.output_dim,)))  #XD
        self.W_c = self.init((self.h1_dim, self.output_dim))
        self.b_c = sharedX(0. * np.ones((self.output_dim,)))  #XD
        
        self.U2_h0 = initializations.get('zero')((self.output_dim, self.h0_dim))
        self.U1_h0 = initializations.get('zero')((self.output_dim, self.h0_dim))
        
        self.U_c = sharedX(1. * np.ones((self.output_dim, self.output_dim)))  #XD
        self.U_f = sharedX(0. * np.ones((self.output_dim, self.output_dim)))

        self.params = [
            self.W_h0, self.b_h0, 
            self.W_h1, self.b_h1,
            self.W_c, self.b_c,
            self.W_f, self.b_f,
            self.U2_h0, self.U1_h0, 
            self.U_c, self.U_f
        ]

        self.regularizers = []
        if self.W_h0_regularizer:
            self.W_h0_regularizer.set_param(self.W_h0)
            self.regularizers.append(self.W_h0_regularizer)
        if self.W_h1_regularizer:
            self.W_h1_regularizer.set_param(self.W_h1)
            self.regularizers.append(self.W_h1_regularizer)
            
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def _step(self,
              x_t, cm_t, 
              h_tm2, h_tm1, c_tm2, c_tm1,
              w_h0, u2_h0, u1_h0, b_h0, w_h1, b_h1, w_f, b_f, w_c, b_c, u_c, u_f):
        h0_t = activations.get('relu')(T.dot(x_t, w_h0) + 0.*T.dot(h_tm2/100., u2_h0) + 0.*T.dot(h_tm1/100., u1_h0) + b_h0)
        h1_t = activations.get('relu')(T.dot(h0_t, w_h1) + b_h1)
        xf_t = T.dot(h1_t, w_f) + b_f + 0.*T.dot(h_tm1/100., u_f)
        xc_t = T.dot(h1_t, w_c) + b_c
        f_t = self.inner_activation(xf_t)
#        f_t = (1 - .5) * f_t + .5
        b_t = activations.get('linear')(xc_t)
        dh_t = T.dot(h_tm1, u_c) * 1.
        c_t = f_t * activations.get('relu')(c_tm1 + b_t + dh_t)
        h_t = self.activation(c_t) - cm_t
        d_t = c_t - c_tm1
        dx_t = d_t - dh_t
        return h_t, c_t, f_t, T.dot(h1_t, w_c), T.dot(h1_t, w_f), dx_t, dh_t

    def _step_inc(self,
              x_t, cm_t, 
              h_tm2, h_tm1, c_tm2, c_tm1,
              w_h0, u2_h0, u1_h0, b_h0, w_h1, b_h1, w_f, b_f, w_c, b_c, u_c, u_f):
        h0_t = activations.get('relu')(T.dot(x_t, w_h0) + 1.*T.dot(c_tm2/100., u2_h0) + 1.*T.dot(c_tm1/100., u1_h0) + b_h0)
        h1_t = activations.get('relu')(T.dot(h0_t, w_h1) + b_h1)
        xf_t = T.dot(h1_t, w_f) + b_f + 0.*T.dot(h_tm1/100., u_f)
        xc_t = T.dot(h1_t, w_c) + b_c
        f_t = self.inner_activation(xf_t)
        b_t = self.activation(xc_t)
        dh_t = T.dot(c_tm1, u_c) * 1.
#        c_t = f_t * activations.get('relu')(c_tm1 + b_t + dh_t)
        c_t = (f_t * 0. + 1.) * activations.get('relu')(b_t + dh_t)
        h_t = self.activation(c_t) - cm_t
        d_t = c_t - c_tm1
        dx_t = d_t - dh_t
        return h_t, c_t, f_t, T.dot(h1_t, w_c), T.dot(h1_t, w_f), dx_t, dh_t
    
    def _step_replace(self,
              x_t, cm_t, 
              h_tm2, h_tm1, c_tm2, c_tm1,
              w_h0, u2_h0, u1_h0, b_h0, w_h1, b_h1, w_f, b_f, w_c, b_c, u_c, u_f):
        h0_t = activations.get('tanh')(T.dot(x_t, w_h0) + 1.*T.dot(h_tm2/100., u2_h0) + 1.*T.dot(h_tm1/100., u1_h0) + b_h0)
        h1_t = activations.get('tanh')(T.dot(h0_t, w_h1) + b_h1)
        xf_t = T.dot(h1_t, w_f) + b_f + 0.*T.dot(h_tm1/100., u_f)
        xc_t = T.dot(h1_t, w_c) + b_c
        f_t = self.inner_activation(xf_t)
        b_t = self.activation(xc_t)
        dh_t = T.dot(h_tm1, u_c) * 1.
#        c_t = f_t * activations.get('relu')(c_tm1 + b_t + dh_t)
#        c_t = (f_t * 0. + 1.) * activations.get('relu')(b_t + dh_t)
#        h_t = self.activation(c_t) - cm_t
        h_t = (f_t * 0. + 1.) * (b_t + dh_t)
        c_t = activations.get('relu')(h_t + cm_t)
        d_t = c_t - c_tm1
        dx_t = d_t - dh_t
        return h_t, c_t, f_t, T.dot(h1_t, w_c), T.dot(h1_t, w_f), dx_t, dh_t
    
    def get_output(self, train=False):
        X = self.get_input(train)
        padded_mask = self.get_padded_shuffled_mask(train, X, pad=1)
        X = X.dimshuffle((1, 0, 2))
        cm = self.cell_mean.dimshuffle((1, 0, 2))
        hidden0 = self.hidden0.dimshuffle((1, 0, 2))
        cell0 = self.cell0.dimshuffle((1, 0, 2))

        step_fns = {
                    'forget+inc' : self._step,
                    'inc' : self._step_inc,
                    'replace' : self._step_replace,
                    }
        
        [outputs, memories, forgets, increments, delta, delta_x, delta_h], updates = theano.scan(
#            self._step,
            step_fns[self.step_type],
            sequences=[X, cm],
            outputs_info=[
                dict(initial=hidden0, taps=[-2,-1]),
                dict(initial=cell0, taps=[-2,-1]),
                None,
                None,
                None,
                None,
                None
            ],
            non_sequences=[self.W_h0, self.U2_h0, self.U1_h0, self.b_h0, self.W_h1, self.b_h1, 
                           self.W_f, self.b_f, self.W_c, self.b_c, self.U_c, self.U_f],
            truncate_gradient=self.truncate_gradient)

        #XD
        if train == False:
            self.forgets = forgets.dimshuffle((1, 0, 2))
            self.increments = increments.dimshuffle((1, 0, 2))
            self.delta = delta.dimshuffle((1, 0, 2))
            self.delta_x = delta_x.dimshuffle((1, 0, 2))
            self.delta_h = delta_h.dimshuffle((1, 0, 2))

        if self.return_sequences:
            return outputs.dimshuffle((1, 0, 2))
        return outputs[-1]

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "output_dim": self.output_dim,
                  "init": self.init.__name__,
                  "inner_init": self.inner_init.__name__,
                  "forget_bias_init": self.forget_bias_init.__name__,
                  "activation": self.activation.__name__,
                  "inner_activation": self.inner_activation.__name__,
                  "truncate_gradient": self.truncate_gradient,
                  "return_sequences": self.return_sequences,
                  "input_dim": self.input_dim,
                  "input_length": self.input_length}
        base_config = super(RLSTM, self).get_config()
        
class ReducedLSTMA(Recurrent):
    def __init__(self, output_dim,
                 init='glorot_uniform', inner_init='orthogonal', forget_bias_init='one',
                 activation='tanh', inner_activation='hard_sigmoid', 
                 fix_b_f=False,  #XD
                 weights=None, truncate_gradient=-1, return_sequences=False,
                 input_dim=None, input_length=None, **kwargs):
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.forget_bias_init = initializations.get(forget_bias_init)
#        self.activation = activations.get(activation)
        self.activation = activations.get('linear') #XD
        self.inner_activation = activations.get(inner_activation)
        self.truncate_gradient = truncate_gradient
        self.return_sequences = return_sequences
        self.initial_weights = weights

        self.input_dim = input_dim
        self.input_length = input_length
        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim)
            
        #XD
        self.fix_b_f = fix_b_f
        self.set_init_input()  
        
        super(ReducedLSTMA, self).__init__(**kwargs)

    def set_init_input(self):
        self.hidden0 = T.matrix()
        self.cell0 = T.matrix()
        self.cell_mean = T.tensor3()
        
    def get_init_input(self):
        return [self.hidden0, self.cell0, self.cell_mean]

    def build(self):
        input_dim = self.input_shape[2]
        self.input = T.tensor3()

        self.W_f = self.init((input_dim, self.output_dim))
#        self.U_f = self.inner_init((self.output_dim, self.output_dim))
        self.U_f = sharedX(.0 * np.ones((self.output_dim, self.output_dim)))  #XD
#        self.b_f = self.forget_bias_init((self.output_dim))
        self.b_f = sharedX(1. * np.ones((self.output_dim,)))  #XD

        self.W_c = self.init((input_dim, self.output_dim))
#        self.U_c = self.inner_init((self.output_dim, self.output_dim))
        self.U_c = sharedX(-.05 * np.ones((self.output_dim, self.output_dim)))  #XD
        self.b_c = shared_zeros((self.output_dim))

        self.params = [
#            self.W_i, self.U_i, self.b_i,
            self.W_c, self.U_c, self.b_c,
            self.W_f, self.U_f, #self.b_f,
#            self.W_o, self.U_o, self.b_o,
        ]
        if not self.fix_b_f:
#            print 'add b_f'
            self.params.append(self.b_f)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def _step(self,
              xf_t, xc_t, cm_t, mask_tm1,
              h_tm1, c_tm1,  # f_tm1 and d_tm1 added by XD
              u_f, u_c):
#        h_mask_tm1 = mask_tm1 * h_tm1
#        c_mask_tm1 = mask_tm1 * c_tm1
        h_mask_tm1 = h_tm1  #XD
        c_mask_tm1 = c_tm1  #XD

        f_t = self.inner_activation(xf_t + 0. * T.dot(c_mask_tm1, u_f))
        b_t = activations.get('linear')(xc_t)
        a_t = f_t
        c_t = a_t * activations.get('relu')(c_mask_tm1 + b_t + T.dot(h_mask_tm1, u_c))
        h_t = self.activation(c_t) - cm_t
        d_t = c_t - c_mask_tm1
        dh_t = T.dot(h_mask_tm1, u_c)
        dx_t = d_t - dh_t
        return h_t, c_t, a_t, b_t, xf_t - self.b_f, dx_t, dh_t     # f_t and b_t added by XD

    def get_output(self, train=False):
        X = self.get_input(train)
        padded_mask = self.get_padded_shuffled_mask(train, X, pad=1)
        X = X.dimshuffle((1, 0, 2))
        cell_mean = self.cell_mean.dimshuffle((1, 0, 2))

#        xi = T.dot(X, self.W_i) + self.b_i
        xf = T.dot(X, self.W_f) + self.b_f
        xc = T.dot(X, self.W_c) + self.b_c
#        xo = T.dot(X, self.W_o) + self.b_o

        [outputs, memories, forgets, increments, delta, delta_x, delta_h], updates = theano.scan(
            self._step,
            sequences=[xf, xc, cell_mean, padded_mask],
#            outputs_info=[
#                T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1),
#                T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1)
#            ],
            outputs_info=[
                self.hidden0,
                self.cell0,
                None,
                None,
                None,
                None,
                None
            ],
            non_sequences=[self.U_f, self.U_c],
            truncate_gradient=self.truncate_gradient)

        #XD
        if train == False:
            self.forgets = forgets.dimshuffle((1, 0, 2))
            self.increments = increments.dimshuffle((1, 0, 2))
            self.delta = delta.dimshuffle((1, 0, 2))
            self.delta_x = delta_x.dimshuffle((1, 0, 2))
            self.delta_h = delta_h.dimshuffle((1, 0, 2))

        if self.return_sequences:
            return outputs.dimshuffle((1, 0, 2))
        return outputs[-1]

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "output_dim": self.output_dim,
                  "init": self.init.__name__,
                  "inner_init": self.inner_init.__name__,
                  "forget_bias_init": self.forget_bias_init.__name__,
                  "activation": self.activation.__name__,
                  "inner_activation": self.inner_activation.__name__,
                  "fix_b_f": self.fix_b_f,
                  "truncate_gradient": self.truncate_gradient,
                  "return_sequences": self.return_sequences,
                  "input_dim": self.input_dim,
                  "input_length": self.input_length}
        base_config = super(ReducedLSTMA, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
       
class ReducedLSTMB(Recurrent):
    def __init__(self, output_dim,
                 init='glorot_uniform', inner_init='orthogonal', forget_bias_init='one',
                 activation='tanh', inner_activation='hard_sigmoid', 
                 forget_type='new',  #XD
                 weights=None, truncate_gradient=-1, return_sequences=False,
                 input_dim=None, input_length=None, **kwargs):
        self.output_dim = output_dim
        init = 'zero'  #XD
#        inner_init = 'uniform'  #XD
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.forget_bias_init = initializations.get(forget_bias_init)
#        self.activation = activations.get(activation)
        self.activation = activations.get('linear') #XD
#        self.inner_activation = activations.get(inner_activation)
        self.inner_activation = activations.get('sigmoid') #XD
        self.truncate_gradient = truncate_gradient
        self.return_sequences = return_sequences
        self.initial_weights = weights

        self.input_dim = input_dim
        self.input_length = input_length
        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim)
            
        #XD
        self.set_init_input()  
        assert forget_type in ['no', 'old', 'new']
        self.forget_type = forget_type
        
        super(ReducedLSTMB, self).__init__(**kwargs)

    def set_init_input(self):
        self.hidden0 = T.matrix()
        self.cell0 = T.matrix()
        self.cell_mean = T.tensor3()
        
    def get_init_input(self):
        return [self.hidden0, self.cell0, self.cell_mean]

    def build(self):
        input_dim = self.input_shape[2]
        self.input = T.tensor3()

        self.W_f = self.init((input_dim, self.output_dim))
#        self.U_f = self.inner_init((self.output_dim, self.output_dim))
        self.U_f = sharedX(.0 * np.ones((self.output_dim, self.output_dim)))  #XD
#        self.b_f = self.forget_bias_init((self.output_dim))
        self.b_f = sharedX(1. * np.ones((self.output_dim,)))  #XD

        self.W_c = self.init((input_dim, self.output_dim))
#        self.U_c = self.inner_init((self.output_dim, self.output_dim))
        self.U_c = sharedX(-.5 * np.ones((self.output_dim, self.output_dim)))  #XD
        self.b_c = shared_zeros((self.output_dim))

        self.params = [
#            self.W_i, self.U_i, self.b_i,
            self.W_c, self.U_c, self.b_c,
            self.W_f, self.U_f, self.b_f,
#            self.W_o, self.U_o, self.b_o,
        ]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def _step(self,
              xf_t, xc_t, cm_t, mask_tm1,
              h_tm1, c_tm1,  # f_tm1 and d_tm1 added by XD
              u_f, u_c):
#        h_mask_tm1 = mask_tm1 * h_tm1
#        c_mask_tm1 = mask_tm1 * c_tm1
        h_mask_tm1 = h_tm1  #XD
        c_mask_tm1 = c_tm1  #XD

        f_t = self.inner_activation(xf_t + 0. * T.dot(h_mask_tm1, u_f))
        b_t = activations.get('linear')(xc_t) + T.dot(h_mask_tm1, u_c)
        a_t = f_t
        c_t = a_t * c_mask_tm1 + b_t
        h_t = self.activation(c_t) - cm_t
        d_t = c_t - c_mask_tm1
        dh_t = T.dot(h_mask_tm1, u_c)
        dx_t = d_t - dh_t
        return h_t, c_t, a_t, b_t, d_t, dx_t, dh_t     # f_t and b_t added by XD

    def get_output(self, train=False):
        X = self.get_input(train)
        padded_mask = self.get_padded_shuffled_mask(train, X, pad=1)
        X = X.dimshuffle((1, 0, 2))
        cell_mean = self.cell_mean.dimshuffle((1, 0, 2))

#        xi = T.dot(X, self.W_i) + self.b_i
        xf = T.dot(X, self.W_f) + self.b_f
        xc = T.dot(X, self.W_c) + self.b_c
#        xo = T.dot(X, self.W_o) + self.b_o

        [outputs, memories, forgets, increments, delta, delta_x, delta_h], updates = theano.scan(
            self._step,
            sequences=[xf, xc, cell_mean, padded_mask],
#            outputs_info=[
#                T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1),
#                T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1)
#            ],
            outputs_info=[
                self.hidden0,
                self.cell0,
                None,
                None,
                None,
                None,
                None
            ],
            non_sequences=[self.U_f, self.U_c],
            truncate_gradient=self.truncate_gradient)

        #XD
        if train == False:
            self.forgets = forgets.dimshuffle((1, 0, 2))
            self.increments = increments.dimshuffle((1, 0, 2))
            self.delta = delta.dimshuffle((1, 0, 2))
            self.delta_x = delta_x.dimshuffle((1, 0, 2))
            self.delta_h = delta_h.dimshuffle((1, 0, 2))

        if self.return_sequences:
            return outputs.dimshuffle((1, 0, 2))
        return outputs[-1]

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "output_dim": self.output_dim,
                  "init": self.init.__name__,
                  "inner_init": self.inner_init.__name__,
                  "forget_bias_init": self.forget_bias_init.__name__,
                  "activation": self.activation.__name__,
                  "inner_activation": self.inner_activation.__name__,
                  "truncate_gradient": self.truncate_gradient,
                  "return_sequences": self.return_sequences,
                  "input_dim": self.input_dim,
                  "input_length": self.input_length}
        base_config = super(ReducedLSTMB, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
class ReducedLSTM(Recurrent):
    def __init__(self, output_dim,
                 init='glorot_uniform', inner_init='orthogonal', forget_bias_init='one',
                 activation='tanh', inner_activation='hard_sigmoid', 
                 forget_type='new',  #XD
                 weights=None, truncate_gradient=-1, return_sequences=False,
                 input_dim=None, input_length=None, **kwargs):
        self.output_dim = output_dim
        init = 'zero'  #XD
#        inner_init = 'uniform'  #XD
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.forget_bias_init = initializations.get(forget_bias_init)
#        self.activation = activations.get(activation)
        self.activation = activations.get('linear') #XD
#        self.inner_activation = activations.get(inner_activation)
        self.inner_activation = activations.get('sigmoid') #XD
        self.truncate_gradient = truncate_gradient
        self.return_sequences = return_sequences
        self.initial_weights = weights

        self.input_dim = input_dim
        self.input_length = input_length
        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim)
            
        #XD
        self.set_init_input()  
        assert forget_type in ['no', 'old', 'new']
        self.forget_type = forget_type
        
        super(ReducedLSTM, self).__init__(**kwargs)

    def set_init_input(self):
        self.hidden0 = T.matrix()
        self.cell0 = T.matrix()
        
    def get_init_input(self):
        return [self.hidden0, self.cell0]

    def build(self):
        input_dim = self.input_shape[2]
        self.input = T.tensor3()

        self.W_f = self.init((input_dim, self.output_dim))
#        self.U_f = self.inner_init((self.output_dim, self.output_dim))
        self.U_f = sharedX(.0 * np.ones((self.output_dim, self.output_dim)))  #XD
#        self.b_f = self.forget_bias_init((self.output_dim))
        self.b_f = sharedX(1. * np.ones((self.output_dim,)))  #XD

        self.W_c = self.init((input_dim, self.output_dim))
#        self.U_c = self.inner_init((self.output_dim, self.output_dim))
        self.U_c = sharedX(-.5 * np.ones((self.output_dim, self.output_dim)))  #XD
        self.b_c = shared_zeros((self.output_dim))

        self.params = [
#            self.W_i, self.U_i, self.b_i,
            self.W_c, self.U_c, self.b_c,
            self.W_f, self.U_f, self.b_f,
#            self.W_o, self.U_o, self.b_o,
        ]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def _step(self,
              xf_t, xc_t, mask_tm1,
              h_tm1, c_tm1,  # f_tm1 and d_tm1 added by XD
              u_f, u_c):
#        h_mask_tm1 = mask_tm1 * h_tm1
#        c_mask_tm1 = mask_tm1 * c_tm1
        h_mask_tm1 = h_tm1  #XD
        c_mask_tm1 = c_tm1  #XD

        f_t = self.inner_activation(xf_t + 0. * T.dot(h_mask_tm1, u_f))
        b_t = self.activation(xc_t + T.dot(h_mask_tm1, u_c))
        a_t = T.maximum(f_t, T.cast(T.le(c_mask_tm1, 0), theano.config.floatX)) # disable forget gate when below mean
        c_t = a_t * (c_mask_tm1 + b_t)
        h_t = self.activation(c_t)
        d_t = c_t - c_mask_tm1
        dh_t = T.dot(h_mask_tm1, u_c)
        dx_t = d_t - dh_t
        return h_t, c_t, a_t, b_t, d_t, dx_t, dh_t     # f_t and b_t added by XD

    def get_output(self, train=False):
        X = self.get_input(train)
        padded_mask = self.get_padded_shuffled_mask(train, X, pad=1)
        X = X.dimshuffle((1, 0, 2))

#        xi = T.dot(X, self.W_i) + self.b_i
        xf = T.dot(X, self.W_f) + self.b_f
        xc = T.dot(X, self.W_c) + self.b_c
#        xo = T.dot(X, self.W_o) + self.b_o

        [outputs, memories, forgets, increments, delta, delta_x, delta_h], updates = theano.scan(
            self._step,
            sequences=[xf, xc, padded_mask],
#            outputs_info=[
#                T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1),
#                T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1)
#            ],
            outputs_info=[
                self.hidden0,
                self.cell0,
                None,
                None,
                None,
                None,
                None
            ],
            non_sequences=[self.U_f, self.U_c],
            truncate_gradient=self.truncate_gradient)

        #XD
        if train == False:
            self.forgets = forgets.dimshuffle((1, 0, 2))
            self.increments = increments.dimshuffle((1, 0, 2))
            self.delta = delta.dimshuffle((1, 0, 2))
            self.delta_x = delta_x.dimshuffle((1, 0, 2))
            self.delta_h = delta_h.dimshuffle((1, 0, 2))

        if self.return_sequences:
            return outputs.dimshuffle((1, 0, 2))
        return outputs[-1]

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "output_dim": self.output_dim,
                  "init": self.init.__name__,
                  "inner_init": self.inner_init.__name__,
                  "forget_bias_init": self.forget_bias_init.__name__,
                  "activation": self.activation.__name__,
                  "inner_activation": self.inner_activation.__name__,
                  "truncate_gradient": self.truncate_gradient,
                  "return_sequences": self.return_sequences,
                  "input_dim": self.input_dim,
                  "input_length": self.input_length}
        base_config = super(ReducedLSTM, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class ReducedLSTM2(Recurrent):
    def __init__(self, output_dim,
                 init='glorot_uniform', inner_init='orthogonal', forget_bias_init='one',
                 activation='tanh', inner_activation='hard_sigmoid', 
                 forget_type='new',  #XD
                 weights=None, truncate_gradient=-1, return_sequences=False,
                 input_dim=None, input_length=None, **kwargs):
        self.output_dim = output_dim
        init = 'zero'  #XD
#        inner_init = 'uniform'  #XD
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.forget_bias_init = initializations.get(forget_bias_init)
#        self.activation = activations.get(activation)
        self.activation = activations.get('linear') #XD
#        self.inner_activation = activations.get(inner_activation)
        self.inner_activation = activations.get('sigmoid') #XD
        self.truncate_gradient = truncate_gradient
        self.return_sequences = return_sequences
        self.initial_weights = weights

        self.input_dim = input_dim
        self.input_length = input_length
        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim)
            
        #XD
        self.set_init_input()  
        assert forget_type in ['no', 'old', 'new']
        self.forget_type = forget_type
        
        super(ReducedLSTM2, self).__init__(**kwargs)

    def set_init_input(self):
        self.hidden0 = T.matrix()
        self.cell0 = T.matrix()
        
    def get_init_input(self):
        return [self.hidden0, self.cell0]

    def build(self):
        input_dim = self.input_shape[2]
        self.input = T.tensor3()

        self.W_f = self.init((input_dim, self.output_dim))
#        self.U_f = self.inner_init((self.output_dim, self.output_dim))
        self.U_f = sharedX(.0 * np.ones((self.output_dim, self.output_dim)))  #XD
#        self.b_f = self.forget_bias_init((self.output_dim))
        self.b_f = sharedX(1. * np.ones((self.output_dim,)))  #XD

        self.W_c = self.init((input_dim, self.output_dim))
#        self.U_c = self.inner_init((self.output_dim, self.output_dim))
        self.U_c = sharedX(-.5 * np.ones((self.output_dim, self.output_dim)))  #XD
        self.b_c = shared_zeros((self.output_dim))

        self.params = [
#            self.W_i, self.U_i, self.b_i,
            self.W_c, self.U_c, self.b_c,
            self.W_f, self.U_f, self.b_f,
#            self.W_o, self.U_o, self.b_o,
        ]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def _step(self,
              xf_t, xc_t, mask_tm1,
              h_tm1, c_tm1,  # f_tm1 and d_tm1 added by XD
              u_f, u_c):
#        h_mask_tm1 = mask_tm1 * h_tm1
#        c_mask_tm1 = mask_tm1 * c_tm1
        h_mask_tm1 = h_tm1  #XD
        c_mask_tm1 = c_tm1  #XD

        f_t = self.inner_activation(xf_t + 0. * T.dot(h_mask_tm1, u_f))
        b_t = self.activation(xc_t + T.dot(h_mask_tm1, u_c))
        a_t = T.maximum(f_t, T.cast(T.le(c_mask_tm1, 0), theano.config.floatX)) # disable forget gate when below mean
        c_t = a_t * c_mask_tm1 + b_t
        h_t = self.activation(c_t)
        d_t = c_t - c_mask_tm1
        dh_t = T.dot(h_mask_tm1, u_c)
        dx_t = d_t - dh_t
        return h_t, c_t, a_t, b_t, d_t, dx_t, dh_t     # f_t and b_t added by XD

    def get_output(self, train=False):
        X = self.get_input(train)
        padded_mask = self.get_padded_shuffled_mask(train, X, pad=1)
        X = X.dimshuffle((1, 0, 2))

#        xi = T.dot(X, self.W_i) + self.b_i
        xf = T.dot(X, self.W_f) + self.b_f
        xc = T.dot(X, self.W_c) + self.b_c
#        xo = T.dot(X, self.W_o) + self.b_o

        [outputs, memories, forgets, increments, delta, delta_x, delta_h], updates = theano.scan(
            self._step,
            sequences=[xf, xc, padded_mask],
#            outputs_info=[
#                T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1),
#                T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1)
#            ],
            outputs_info=[
                self.hidden0,
                self.cell0,
                None,
                None,
                None,
                None,
                None
            ],
            non_sequences=[self.U_f, self.U_c],
            truncate_gradient=self.truncate_gradient)

        #XD
        if train == False:
            self.forgets = forgets.dimshuffle((1, 0, 2))
            self.increments = increments.dimshuffle((1, 0, 2))
            self.delta = delta.dimshuffle((1, 0, 2))
            self.delta_x = delta_x.dimshuffle((1, 0, 2))
            self.delta_h = delta_h.dimshuffle((1, 0, 2))

        if self.return_sequences:
            return outputs.dimshuffle((1, 0, 2))
        return outputs[-1]

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "output_dim": self.output_dim,
                  "init": self.init.__name__,
                  "inner_init": self.inner_init.__name__,
                  "forget_bias_init": self.forget_bias_init.__name__,
                  "activation": self.activation.__name__,
                  "inner_activation": self.inner_activation.__name__,
                  "truncate_gradient": self.truncate_gradient,
                  "return_sequences": self.return_sequences,
                  "input_dim": self.input_dim,
                  "input_length": self.input_length}
        base_config = super(ReducedLSTM2, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class ReducedLSTM3(Recurrent):
    def __init__(self, output_dim,
                 init='glorot_uniform', inner_init='orthogonal', forget_bias_init='one',
                 activation='tanh', inner_activation='hard_sigmoid', 
                 forget_type='new',  #XD
                 weights=None, truncate_gradient=-1, return_sequences=False,
                 input_dim=None, input_length=None, **kwargs):
        self.output_dim = output_dim
        init = 'zero'  #XD
#        inner_init = 'uniform'  #XD
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.forget_bias_init = initializations.get(forget_bias_init)
#        self.activation = activations.get(activation)
        self.activation = activations.get('linear') #XD
#        self.inner_activation = activations.get(inner_activation)
        self.inner_activation = activations.get('sigmoid') #XD
        self.truncate_gradient = truncate_gradient
        self.return_sequences = return_sequences
        self.initial_weights = weights

        self.input_dim = input_dim
        self.input_length = input_length
        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim)
            
        #XD
        self.set_init_input()  
        assert forget_type in ['no', 'old', 'new']
        self.forget_type = forget_type
        
        super(ReducedLSTM3, self).__init__(**kwargs)

    def set_init_input(self):
        self.hidden0 = T.matrix()
        self.cell0 = T.matrix()
        
    def get_init_input(self):
        return [self.hidden0, self.cell0]

    def build(self):
        input_dim = self.input_shape[2]
        self.input = T.tensor3()

        self.W_f = self.init((input_dim, self.output_dim))
#        self.U_f = self.inner_init((self.output_dim, self.output_dim))
        self.U_f = sharedX(.0 * np.ones((self.output_dim, self.output_dim)))  #XD
#        self.b_f = self.forget_bias_init((self.output_dim))
        self.b_f = sharedX(1. * np.ones((self.output_dim,)))  #XD

        self.W_c = self.init((input_dim, self.output_dim))
#        self.U_c = self.inner_init((self.output_dim, self.output_dim))
        self.U_c = sharedX(-.5 * np.ones((self.output_dim, self.output_dim)))  #XD
        self.b_c = shared_zeros((self.output_dim))

        self.params = [
#            self.W_i, self.U_i, self.b_i,
            self.W_c, self.U_c, self.b_c,
            self.W_f, self.U_f, self.b_f,
#            self.W_o, self.U_o, self.b_o,
        ]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def _step(self,
              xf_t, xc_t, mask_tm1,
              h_tm1, c_tm1,  # f_tm1 and d_tm1 added by XD
              u_f, u_c):
#        h_mask_tm1 = mask_tm1 * h_tm1
#        c_mask_tm1 = mask_tm1 * c_tm1
        h_mask_tm1 = h_tm1  #XD
        c_mask_tm1 = c_tm1  #XD

        f_t = self.inner_activation(xf_t + 0. * T.dot(h_mask_tm1, u_f))
        b_t = self.activation(xc_t + T.dot(h_mask_tm1, u_c))
        a_t = f_t * 0. + 1.
        c_t = a_t * c_mask_tm1 + b_t
        h_t = self.activation(c_t)
        d_t = c_t - c_mask_tm1
        dh_t = T.dot(h_mask_tm1, u_c)
        dx_t = d_t - dh_t
        return h_t, c_t, a_t, b_t, d_t, dx_t, dh_t     # f_t and b_t added by XD

    def get_output(self, train=False):
        X = self.get_input(train)
        padded_mask = self.get_padded_shuffled_mask(train, X, pad=1)
        X = X.dimshuffle((1, 0, 2))

#        xi = T.dot(X, self.W_i) + self.b_i
        xf = T.dot(X, self.W_f) + self.b_f
        xc = T.dot(X, self.W_c) + self.b_c
#        xo = T.dot(X, self.W_o) + self.b_o

        [outputs, memories, forgets, increments, delta, delta_x, delta_h], updates = theano.scan(
            self._step,
            sequences=[xf, xc, padded_mask],
#            outputs_info=[
#                T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1),
#                T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1)
#            ],
            outputs_info=[
                self.hidden0,
                self.cell0,
                None,
                None,
                None,
                None,
                None
            ],
            non_sequences=[self.U_f, self.U_c],
            truncate_gradient=self.truncate_gradient)

        #XD
        if train == False:
            self.forgets = forgets.dimshuffle((1, 0, 2))
            self.increments = increments.dimshuffle((1, 0, 2))
            self.delta = delta.dimshuffle((1, 0, 2))
            self.delta_x = delta_x.dimshuffle((1, 0, 2))
            self.delta_h = delta_h.dimshuffle((1, 0, 2))

        if self.return_sequences:
            return outputs.dimshuffle((1, 0, 2))
        return outputs[-1]

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "output_dim": self.output_dim,
                  "init": self.init.__name__,
                  "inner_init": self.inner_init.__name__,
                  "forget_bias_init": self.forget_bias_init.__name__,
                  "activation": self.activation.__name__,
                  "inner_activation": self.inner_activation.__name__,
                  "truncate_gradient": self.truncate_gradient,
                  "return_sequences": self.return_sequences,
                  "input_dim": self.input_dim,
                  "input_length": self.input_length}
        base_config = super(ReducedLSTM3, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
        
class JZS1(Recurrent):
    '''
        Evolved recurrent neural network architectures from the evaluation of thousands
        of models, serving as alternatives to LSTMs and GRUs. See Jozefowicz et al. 2015.

        This corresponds to the `MUT1` architecture described in the paper.

        Takes inputs with shape:
        (nb_samples, max_sample_length (samples shorter than this are padded with zeros at the end), input_dim)

        and returns outputs with shape:
        if not return_sequences:
            (nb_samples, output_dim)
        if return_sequences:
            (nb_samples, max_sample_length, output_dim)

        References:
            An Empirical Exploration of Recurrent Network Architectures
                http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf
    '''
    def __init__(self, output_dim,
                 init='glorot_uniform', inner_init='orthogonal',
                 activation='tanh', inner_activation='sigmoid',
                 weights=None, truncate_gradient=-1, return_sequences=False,
                 input_dim=None, input_length=None, **kwargs):
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        self.truncate_gradient = truncate_gradient
        self.return_sequences = return_sequences
        self.initial_weights = weights

        self.input_dim = input_dim
        self.input_length = input_length
        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim)
        super(JZS1, self).__init__(**kwargs)

    def build(self):
        input_dim = self.input_shape[2]
        self.input = T.tensor3()

        self.W_z = self.init((input_dim, self.output_dim))
        self.b_z = shared_zeros((self.output_dim))

        self.W_r = self.init((input_dim, self.output_dim))
        self.U_r = self.inner_init((self.output_dim, self.output_dim))
        self.b_r = shared_zeros((self.output_dim))

        self.U_h = self.inner_init((self.output_dim, self.output_dim))
        self.b_h = shared_zeros((self.output_dim))

        # P_h used to project X onto different dimension, using sparse random projections
        if input_dim == self.output_dim:
            self.Pmat = theano.shared(np.identity(self.output_dim, dtype=theano.config.floatX), name=None)
        else:
            P = np.random.binomial(1, 0.5, size=(input_dim, self.output_dim)).astype(theano.config.floatX) * 2 - 1
            P = 1 / np.sqrt(input_dim) * P
            self.Pmat = theano.shared(P, name=None)

        self.params = [
            self.W_z, self.b_z,
            self.W_r, self.U_r, self.b_r,
            self.U_h, self.b_h,
            self.Pmat
        ]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def _step(self,
              xz_t, xr_t, xh_t, mask_tm1,
              h_tm1,
              u_r, u_h):
        h_mask_tm1 = mask_tm1 * h_tm1
        z = self.inner_activation(xz_t)
        r = self.inner_activation(xr_t + T.dot(h_mask_tm1, u_r))
        hh_t = self.activation(xh_t + T.dot(r * h_mask_tm1, u_h))
        h_t = hh_t * z + h_mask_tm1 * (1 - z)
        return h_t

    def get_output(self, train=False):
        X = self.get_input(train)
        padded_mask = self.get_padded_shuffled_mask(train, X, pad=1)
        X = X.dimshuffle((1, 0, 2))

        x_z = T.dot(X, self.W_z) + self.b_z
        x_r = T.dot(X, self.W_r) + self.b_r
        x_h = T.tanh(T.dot(X, self.Pmat)) + self.b_h
        outputs, updates = theano.scan(
            self._step,
            sequences=[x_z, x_r, x_h, padded_mask],
            outputs_info=T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1),
            non_sequences=[self.U_r, self.U_h],
            truncate_gradient=self.truncate_gradient)
        if self.return_sequences:
            return outputs.dimshuffle((1, 0, 2))
        return outputs[-1]

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "output_dim": self.output_dim,
                  "init": self.init.__name__,
                  "inner_init": self.inner_init.__name__,
                  "activation": self.activation.__name__,
                  "inner_activation": self.inner_activation.__name__,
                  "truncate_gradient": self.truncate_gradient,
                  "return_sequences": self.return_sequences,
                  "input_dim": self.input_dim,
                  "input_length": self.input_length}
        base_config = super(JZS1, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class JZS2(Recurrent):
    '''
        Evolved recurrent neural network architectures from the evaluation of thousands
        of models, serving as alternatives to LSTMs and GRUs. See Jozefowicz et al. 2015.

        This corresponds to the `MUT2` architecture described in the paper.

        Takes inputs with shape:
        (nb_samples, max_sample_length (samples shorter than this are padded with zeros at the end), input_dim)

        and returns outputs with shape:
        if not return_sequences:
            (nb_samples, output_dim)
        if return_sequences:
            (nb_samples, max_sample_length, output_dim)

        References:
            An Empirical Exploration of Recurrent Network Architectures
                http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf
    '''
    def __init__(self, output_dim,
                 init='glorot_uniform', inner_init='orthogonal',
                 activation='tanh', inner_activation='sigmoid',
                 weights=None, truncate_gradient=-1, return_sequences=False,
                 input_dim=None, input_length=None, **kwargs):
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        self.truncate_gradient = truncate_gradient
        self.return_sequences = return_sequences
        self.initial_weights = weights

        self.input_dim = input_dim
        self.input_length = input_length
        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim)
        super(JZS2, self).__init__(**kwargs)

    def build(self):
        input_dim = self.input_shape[2]
        self.input = T.tensor3()

        self.W_z = self.init((input_dim, self.output_dim))
        self.U_z = self.inner_init((self.output_dim, self.output_dim))
        self.b_z = shared_zeros((self.output_dim))

        self.U_r = self.inner_init((self.output_dim, self.output_dim))
        self.b_r = shared_zeros((self.output_dim))

        self.W_h = self.init((input_dim, self.output_dim))
        self.U_h = self.inner_init((self.output_dim, self.output_dim))
        self.b_h = shared_zeros((self.output_dim))

        # P_h used to project X onto different dimension, using sparse random projections
        if input_dim == self.output_dim:
            self.Pmat = theano.shared(np.identity(self.output_dim, dtype=theano.config.floatX), name=None)
        else:
            P = np.random.binomial(1, 0.5, size=(input_dim, self.output_dim)).astype(theano.config.floatX) * 2 - 1
            P = 1 / np.sqrt(input_dim) * P
            self.Pmat = theano.shared(P, name=None)

        self.params = [
            self.W_z, self.U_z, self.b_z,
            self.U_r, self.b_r,
            self.W_h, self.U_h, self.b_h,
            self.Pmat
        ]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def _step(self,
              xz_t, xr_t, xh_t, mask_tm1,
              h_tm1,
              u_z, u_r, u_h):
        h_mask_tm1 = mask_tm1 * h_tm1
        z = self.inner_activation(xz_t + T.dot(h_mask_tm1, u_z))
        r = self.inner_activation(xr_t + T.dot(h_mask_tm1, u_r))
        hh_t = self.activation(xh_t + T.dot(r * h_mask_tm1, u_h))
        h_t = hh_t * z + h_mask_tm1 * (1 - z)
        return h_t

    def get_output(self, train=False):
        X = self.get_input(train)
        padded_mask = self.get_padded_shuffled_mask(train, X, pad=1)
        X = X.dimshuffle((1, 0, 2))

        x_z = T.dot(X, self.W_z) + self.b_z
        x_r = T.dot(X, self.Pmat) + self.b_r
        x_h = T.dot(X, self.W_h) + self.b_h
        outputs, updates = theano.scan(
            self._step,
            sequences=[x_z, x_r, x_h, padded_mask],
            outputs_info=T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1),
            non_sequences=[self.U_z, self.U_r, self.U_h],
            truncate_gradient=self.truncate_gradient)
        if self.return_sequences:
            return outputs.dimshuffle((1, 0, 2))
        return outputs[-1]

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "output_dim": self.output_dim,
                  "init": self.init.__name__,
                  "inner_init": self.inner_init.__name__,
                  "activation": self.activation.__name__,
                  "inner_activation": self.inner_activation.__name__,
                  "truncate_gradient": self.truncate_gradient,
                  "return_sequences": self.return_sequences,
                  "input_dim": self.input_dim,
                  "input_length": self.input_length}
        base_config = super(JZS2, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class JZS3(Recurrent):
    '''
        Evolved recurrent neural network architectures from the evaluation of thousands
        of models, serving as alternatives to LSTMs and GRUs. See Jozefowicz et al. 2015.

        This corresponds to the `MUT3` architecture described in the paper.

        Takes inputs with shape:
        (nb_samples, max_sample_length (samples shorter than this are padded with zeros at the end), input_dim)

        and returns outputs with shape:
        if not return_sequences:
            (nb_samples, output_dim)
        if return_sequences:
            (nb_samples, max_sample_length, output_dim)

        References:
            An Empirical Exploration of Recurrent Network Architectures
                http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf
    '''
    def __init__(self, output_dim,
                 init='glorot_uniform', inner_init='orthogonal',
                 activation='tanh', inner_activation='sigmoid',
                 weights=None, truncate_gradient=-1, return_sequences=False,
                 input_dim=None, input_length=None, **kwargs):
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        self.truncate_gradient = truncate_gradient
        self.return_sequences = return_sequences
        self.initial_weights = weights

        self.input_dim = input_dim
        self.input_length = input_length
        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim)
        super(JZS3, self).__init__(**kwargs)

    def build(self):
        input_dim = self.input_shape[2]
        self.input = T.tensor3()

        self.W_z = self.init((input_dim, self.output_dim))
        self.U_z = self.inner_init((self.output_dim, self.output_dim))
        self.b_z = shared_zeros((self.output_dim))

        self.W_r = self.init((input_dim, self.output_dim))
        self.U_r = self.inner_init((self.output_dim, self.output_dim))
        self.b_r = shared_zeros((self.output_dim))

        self.W_h = self.init((input_dim, self.output_dim))
        self.U_h = self.inner_init((self.output_dim, self.output_dim))
        self.b_h = shared_zeros((self.output_dim))

        self.params = [
            self.W_z, self.U_z, self.b_z,
            self.W_r, self.U_r, self.b_r,
            self.W_h, self.U_h, self.b_h,
        ]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def _step(self,
              xz_t, xr_t, xh_t, mask_tm1,
              h_tm1,
              u_z, u_r, u_h):
        h_mask_tm1 = mask_tm1 * h_tm1
        z = self.inner_activation(xz_t + T.dot(T.tanh(h_mask_tm1), u_z))
        r = self.inner_activation(xr_t + T.dot(h_mask_tm1, u_r))
        hh_t = self.activation(xh_t + T.dot(r * h_mask_tm1, u_h))
        h_t = hh_t * z + h_mask_tm1 * (1 - z)
        return h_t

    def get_output(self, train=False):
        X = self.get_input(train)
        padded_mask = self.get_padded_shuffled_mask(train, X, pad=1)
        X = X.dimshuffle((1, 0, 2))

        x_z = T.dot(X, self.W_z) + self.b_z
        x_r = T.dot(X, self.W_r) + self.b_r
        x_h = T.dot(X, self.W_h) + self.b_h
        outputs, updates = theano.scan(
            self._step,
            sequences=[x_z, x_r, x_h, padded_mask],
            outputs_info=T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1),
            non_sequences=[self.U_z, self.U_r, self.U_h],
            truncate_gradient=self.truncate_gradient
        )
        if self.return_sequences:
            return outputs.dimshuffle((1, 0, 2))
        return outputs[-1]

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "output_dim": self.output_dim,
                  "init": self.init.__name__,
                  "inner_init": self.inner_init.__name__,
                  "activation": self.activation.__name__,
                  "inner_activation": self.inner_activation.__name__,
                  "truncate_gradient": self.truncate_gradient,
                  "return_sequences": self.return_sequences,
                  "input_dim": self.input_dim,
                  "input_length": self.input_length}
        base_config = super(JZS3, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
