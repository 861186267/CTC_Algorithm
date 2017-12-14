#! python 
# @Time    : 17-12-5
# @Author  : kay
# @File    : bidyrnn.py
# @E-mail  : 861186267@qq.com
# @Function:

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.util import nest

from model import rnn_cell_impl

# pylint: disable=protected-access
_concat = rnn_cell_impl._concat


# pylint: disable=unused-argument
def _rnn_step(
        time, sequence_length, zero_output, state, call_cell):
    flat_state = nest.flatten(state)
    flat_zero_output = nest.flatten(zero_output)

    def _copy_one_through(output, new_output):
        # If the state contains a scalar value we simply pass it through.
        copy_cond = (time >= sequence_length)
        with ops.colocate_with(new_output):
            return array_ops.where(copy_cond, output, new_output)

    def _copy_some_through(flat_new_output, flat_new_state):
        # Use broadcasting select to determine which values should get
        # the previous state & zero output, and which values should get
        # a calculated state & output.
        flat_new_output = [
            _copy_one_through(zero_output, new_output)
            for zero_output, new_output in zip(flat_zero_output, flat_new_output)]
        flat_new_state = [
            _copy_one_through(state, new_state)
            for state, new_state in zip(flat_state, flat_new_state)]
        return flat_new_output + flat_new_state

    # TODO(ebrevdo): skipping these conditionals may cause a slowdown,
    # but benefits from removing cond() and its gradient.  We should
    # profile with and without this switch here.

    new_output, new_state = call_cell()
    nest.assert_same_structure(state, new_state)
    new_state = nest.flatten(new_state)
    new_output = nest.flatten(new_output)
    final_output_and_state = _copy_some_through(new_output, new_state)

    final_output = final_output_and_state[:len(flat_zero_output)]
    final_state = final_output_and_state[len(flat_zero_output):]

    for output, flat_output in zip(final_output, flat_zero_output):
        output.set_shape(flat_output.get_shape())
    for substate, flat_substate in zip(final_state, flat_state):
        substate.set_shape(flat_substate.get_shape())

    final_output = nest.pack_sequence_as(
        structure=zero_output, flat_sequence=final_output)
    final_state = nest.pack_sequence_as(
        structure=state, flat_sequence=final_state)

    return final_output, final_state


def _dynamic_rnn_loop(cell,
                      inputs,
                      initial_state,
                      parallel_iterations,
                      swap_memory,
                      sequence_length=None,
                      dtype=None):
    state = initial_state

    # [(len, 1, 40)]
    # [(len, 1, 128)]
    flat_input = nest.flatten(inputs)
    flat_output_size = nest.flatten(cell.output_size)

    # flat_input[0]: (len, 1, 40)
    # flat_input[0]: (len, 1, 128)
    # Construct an initial output
    input_shape = array_ops.shape(flat_input[0])
    time_steps = input_shape[0]
    batch_size = flat_input[0].shape[1].value

    inputs_got_shape = tuple(input_.get_shape() for input_ in flat_input)

    # len, 1
    const_time_steps, const_batch_size = inputs_got_shape[0].as_list()[:2]

    # Prepare dynamic conditional copying of state & output
    def _create_zero_arrays(size):
        size = _concat(batch_size, size)
        return array_ops.zeros(
            array_ops.stack(size), dtype)

    # (1, 128)
    flat_zero_output = tuple(_create_zero_arrays(output)
                             for output in flat_output_size)
    zero_output = nest.pack_sequence_as(structure=cell.output_size,
                                        flat_sequence=flat_zero_output)
    # (0)
    time = array_ops.constant(0, dtype=dtypes.int32, name="time")

    def _create_ta(name, dtype):
        return tensor_array_ops.TensorArray(dtype=dtype,
                                            size=time_steps,
                                            tensor_array_name="dynamic_rnn" + name)

    output_ta = tuple(_create_ta("output_%d" % i, dtype)
                      for i in range(len(flat_output_size)))
    input_ta = tuple(_create_ta("input_%d" % i, flat_input[i].dtype)
                     for i in range(len(flat_input)))

    # (len, 1, 40)
    input_ta = tuple(ta.unstack(input_)
                     for ta, input_ in zip(input_ta, flat_input))

    def _time_step(time, output_ta_t, state):
        """Take a time step of the dynamic RNN.
    
        Args:
          time: int32 scalar Tensor.
          output_ta_t: List of `TensorArray`s that represent the output.
          state: nested tuple of vector tensors that represent the state.
    
        Returns:
          The tuple (time + 1, output_ta_t with updated flow, new_state).
        """

        # input_ta (len, 1, 40)
        # input_t (1, 40)
        input_t = tuple(ta.read(time) for ta in input_ta)
        # Restore some shape information
        for input_, shape in zip(input_t, inputs_got_shape):
            input_.set_shape(shape[1:])

        input_t = nest.pack_sequence_as(structure=inputs, flat_sequence=input_t)
        call_cell = lambda: cell(input_t, state)

        (output, new_state) = _rnn_step(
            time=time,
            sequence_length=sequence_length,
            zero_output=zero_output,
            state=state,
            call_cell=call_cell, )

        # Pack state if using state tuples
        output = nest.flatten(output)

        output_ta_t = tuple(
            ta.write(time, out) for ta, out in zip(output_ta_t, output))

        return (time + 1, output_ta_t, new_state)

    _, output_final_ta, final_state = control_flow_ops.while_loop(
        cond=lambda time, *_: time < time_steps,
        body=_time_step,
        loop_vars=(time, output_ta, state),
        parallel_iterations=parallel_iterations,
        swap_memory=swap_memory)

    # Unpack final output if not using output tuples.
    final_outputs = tuple(ta.stack() for ta in output_final_ta)

    # Restore some shape information
    for output, output_size in zip(final_outputs, flat_output_size):
        shape = _concat(
            [const_time_steps, const_batch_size], output_size, static=True)
        output.set_shape(shape)

    final_outputs = nest.pack_sequence_as(
        structure=cell.output_size, flat_sequence=final_outputs)

    return (final_outputs, final_state)


def dynamic_rnn(cell, inputs, sequence_length=None,
                dtype=None, swap_memory=False, scope=None):
    # By default, time_major==False and inputs are batch-major: shaped
    #   [batch, time, depth]
    # For internal calculations, we transpose to [time, batch, depth]
    flat_input = nest.flatten(inputs)

    parallel_iterations = 32

    with vs.variable_scope(scope) as varscope:
        if varscope.caching_device is None:
            varscope.set_caching_device(lambda op: op.device)

        batch_size = flat_input[0].shape[1].value

        # LSTMStateTuple(0 c:(1, 128) h:(1, 128))
        state = cell.zero_state(batch_size, dtype)

        # (len, 1, 40) (len, 1, 128)
        inputs = nest.pack_sequence_as(structure=inputs, flat_sequence=flat_input)

        # (len, 1, 128) (len, 1, 128)
        (outputs, final_state) = _dynamic_rnn_loop(
            cell,
            inputs,
            state,
            parallel_iterations=parallel_iterations,
            swap_memory=swap_memory,
            sequence_length=sequence_length,
            dtype=dtype)

        return (outputs, final_state)


def bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs, sequence_length=None,
                              dtype=None, swap_memory=False, scope=None):
    with vs.variable_scope(scope or "bidirectional_rnn"):
        # Forward direction
        with vs.variable_scope("fw") as fw_scope:
            output_fw, output_state_fw = dynamic_rnn(
                cell=cell_fw, inputs=inputs, sequence_length=sequence_length, dtype=dtype, swap_memory=swap_memory,
                scope=fw_scope)

        time_dim = 0
        batch_dim = 1

        def _reverse(input_, seq_lengths, seq_dim, batch_dim):
            return array_ops.reverse_sequence(
                input=input_, seq_lengths=seq_lengths,
                seq_dim=seq_dim, batch_dim=batch_dim)

        with vs.variable_scope("bw") as bw_scope:
            inputs_reverse = _reverse(
                inputs, seq_lengths=sequence_length,
                seq_dim=time_dim, batch_dim=batch_dim)
            tmp, output_state_bw = dynamic_rnn(
                cell=cell_bw, inputs=inputs_reverse, sequence_length=sequence_length, dtype=dtype,
                swap_memory=swap_memory, scope=bw_scope)

    output_bw = _reverse(
        tmp, seq_lengths=sequence_length,
        seq_dim=time_dim, batch_dim=batch_dim)

    outputs = (output_fw, output_bw)
    output_states = (output_state_fw, output_state_bw)

    return (outputs, output_states)
