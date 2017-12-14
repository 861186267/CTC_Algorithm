#! python 
# @Time    : 17-12-5
# @Author  : kay
# @File    : rnn_cell_impl.py
# @E-mail  : 861186267@qq.com
# @Function:

import collections
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base as base_layer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.util import nest

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"


def _concat(prefix, suffix, static=False):
    p = tensor_shape.as_shape(prefix)
    p_static = p.as_list() if p.ndims is not None else None
    p = (constant_op.constant(p.as_list(), dtype=dtypes.int32)
         if p.is_fully_defined() else None)

    s = tensor_shape.as_shape(suffix)
    s_static = s.as_list() if s.ndims is not None else None
    s = (constant_op.constant(s.as_list(), dtype=dtypes.int32)
         if s.is_fully_defined() else None)

    if static:
        shape = tensor_shape.as_shape(p_static).concatenate(s_static)
        shape = shape.as_list() if shape.ndims is not None else None
    else:
        shape = array_ops.concat((p, s), 0)

    return shape


def _zero_state_tensors(state_size, batch_size, dtype):
    """Create tensors of zeros based on state_size, batch_size, and dtype."""

    def get_state_shape(s):
        """Combine s with batch_size to get a proper tensor shape."""
        c = _concat(batch_size, s)
        size = array_ops.zeros(c, dtype=dtype)
        if context.in_graph_mode():
            c_static = _concat(batch_size, s, static=True)
            size.set_shape(c_static)
        return size

    return nest.map_structure(get_state_shape, state_size)


LSTMStateTuple = collections.namedtuple("LSTMStateTuple", ("c", "h"))


class _Linear(object):
    # forward 1th
    # (1, 40) (1, 128) --> (1, 168)
    # (1, 168) * (168, 512) + 512 --> (1, 512)
    # backward 1th
    # (1, 40) (1, 128) --> (1, 168)
    # (1, 168) * (168, 512) + 512 --> (1, 512)

    # forward 2th
    # (1, 128) (1, 128) --> (1, 256)
    # (1, 256) * (256, 512) + 512 --> (1, 512)
    # backward 2th
    # (1, 128) (1, 128) --> (1, 256)
    # (1, 256) * (256, 512) + 512 --> (1, 512)
    def __init__(self,
                 args,
                 output_size,
                 build_bias,
                 bias_initializer=None,
                 kernel_initializer=None):
        self._build_bias = build_bias

        self._is_sequence = True

        # Calculate the total size of arguments on dimension 1.
        total_arg_size = 0
        shapes = [a.get_shape() for a in args]
        for shape in shapes:
            total_arg_size += shape[1].value

        dtype = [a.dtype for a in args][0]

        scope = vs.get_variable_scope()
        with vs.variable_scope(scope) as outer_scope:
            self._weights = vs.get_variable(
                _WEIGHTS_VARIABLE_NAME, [total_arg_size, output_size],
                dtype=dtype,
                initializer=kernel_initializer)
            if build_bias:
                with vs.variable_scope(outer_scope) as inner_scope:
                    inner_scope.set_partitioner(None)
                    if bias_initializer is None:
                        bias_initializer = init_ops.constant_initializer(0.0, dtype=dtype)
                    self._biases = vs.get_variable(
                        _BIAS_VARIABLE_NAME, [output_size],
                        dtype=dtype,
                        initializer=bias_initializer)

    def __call__(self, args):
        self.args = args
        res = math_ops.matmul(array_ops.concat(args, 1), self._weights)
        if self._build_bias:
            res = nn_ops.bias_add(res, self._biases)
        return res


class BasicLSTMCell(base_layer.Layer):
    def __init__(self, num_units, forget_bias=1.0, activation=None, reuse=None):
        super(BasicLSTMCell, self).__init__(_reuse=reuse)

        self._num_units = num_units
        self._forget_bias = forget_bias
        self._activation = activation or math_ops.tanh
        self._linear = None
        self.inputs = None

    @property
    def state_size(self):
        return LSTMStateTuple(self._num_units, self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def zero_state(self, batch_size, dtype):
        with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
            state_size = self.state_size
            return _zero_state_tensors(state_size, batch_size, dtype)

    def call(self, inputs, state):
        """Long short-term memory cell (LSTM).
        Args:
          inputs: `2-D` tensor with shape `[batch_size x input_size]`.
          state: An `LSTMStateTuple` of state tensors, each shaped
            `[batch_size x self.state_size]`
            `[batch_size x 2 * self.state_size]`.
        Returns:
          A pair containing the new hidden state, and the new state (`LSTMStateTuple`).
        """
        sigmoid = math_ops.sigmoid
        # Parameters of gates are concatenated into one multiply for efficiency.
        self.inputs = inputs
        self.c, self.h = state
        # inputs, h : (1, 40) (1, 128) forward
        # inputs, h : (1, 40) (1, 128) backward
        # inputs, h : (1, 128) (1, 128) forward
        # inputs, h : (1, 128) (1, 128) backward
        if self._linear is None:
            self._linear = _Linear([inputs, self.h], 4 * self._num_units, True)
        self.i, self.j, self.f, self.o = array_ops.split(
            value=self._linear([inputs, self.h]), num_or_size_splits=4, axis=1)

        new_c = (
            self.c * sigmoid(self.f + self._forget_bias) + sigmoid(self.i) * self._activation(self.j))
        new_h = self._activation(new_c) * sigmoid(self.o)

        # new_c, new_h : (1, 128) (1, 128) forward
        # new_c, new_h : (1, 128) (1, 128) backward
        # new_c, new_h : (1, 128) (1, 128) forward
        # new_c, new_h : (1, 128) (1, 128) backward
        new_state = LSTMStateTuple(new_c, new_h)

        return new_h, new_state
