import tensorflow as tf
import tensorflow.contrib.rnn as rnn
from dnc import dnc
from dnc.access import AccessState
from dnc.dnc import DNCState
from tensorflow.python.util import nest
import sonnet as snt

# from tensorflow.contrib.rnn.python.ops.core_rnn_cel_impl import _linear
# from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import _checked_scope
batch_flatten = snt.BatchFlatten()

def choose_dnc_state(cond, s1, s2, structure):
    s1_flat = nest.flatten(s1)
    s2_flat = nest.flatten(s2)
    output_flat = [tf.where(cond, a, b) for a, b in zip(s1_flat, s2_flat)]

    return nest.pack_sequence_as(structure=structure,
                                 flat_sequence=output_flat)

def dnc_read(inputs, aux_output, cell, previous_state):
    """
    main_dnc: read

    Args:
        inputs: [x, prev_read, aux_output or 0s]
        cell: main_dnc cell
        previous_state: previous main_dnc state

    Returns:
        read words: read vectors from memory
        state
    """
    scope_name = 'main_dnc/memory_access'
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
        # TODO remember to use new controller state at next time step
        # previous states
        prev_access_output = previous_state.access_output
        prev_access_state = previous_state.access_state
        prev_controller_state = previous_state.controller_state

        # append prev_access_output to inputs, and build controller
        batch_flatten = snt.BatchFlatten()
        inputs = tf.concat(
                [batch_flatten(inputs), batch_flatten(aux_output), batch_flatten(prev_access_output)], 1)
        controller_output, controller_state = cell._controller(
                batch_flatten(inputs), prev_controller_state)

        # clip controller output
        controller_output = cell._clip_if_enabled(controller_output)
        controller_state = snt.nest.map(cell._clip_if_enabled, controller_state)

        # update usage without updating `linkage`
        access_inputs = cell._access._read_inputs(controller_output)
        usage = cell._access._freeness(
                write_weights=prev_access_state.write_weights,
                free_gate=access_inputs['free_gate'],
                read_weights=prev_access_state.read_weights,
                prev_usage=prev_access_state.usage)

        # erase memory
        write_weights = cell._access._write_weights(access_inputs, prev_access_state.memory,
                usage)
        expand_address = tf.expand_dims(write_weights, 3)
        reset_weights = tf.expand_dims(access_inputs['erase_vectors'], 2)
        weighted_resets = expand_address * reset_weights
        reset_gate = tf.reduce_prod(1 - weighted_resets, [1])
        memory = prev_access_state.memory * reset_gate

        # read from memory
        read_weights = cell._access._read_weights(
                access_inputs,
                memory=memory, # prev_access_state.memory,
                prev_read_weights=prev_access_state.read_weights,
                link=prev_access_state.linkage.link)
        read_words = tf.matmul(read_weights, memory)

        # dnc & access state after read
        access_state = AccessState(
                memory=memory, # prev_access_state.memory,
                read_weights=read_weights,
                write_weights=write_weights,
                linkage=prev_access_state.linkage,
                usage=usage)

    return read_words, DNCState(
            access_output=read_words,
            access_state=access_state,
            controller_state=controller_state)


class ACTWrapper(rnn.RNNCell):
    """Adaptive Computation Time wrapper (based on https://arxiv.org/abs/1603.08983)"""

    def __init__(self, main_dnc, aux_dnc, ponder_limit=100, epsilon=0.01, init_halting_bias=1.0, reuse=None, divergence_type=None):
        self._main_dnc = main_dnc
        self._aux_dnc = aux_dnc
        self._ponder_limit = ponder_limit
        self._epsilon = epsilon
        self._init_halting_bias = init_halting_bias
        self._reuse = reuse

        self._ponder_steps_op = None
        self._ponder_cost_op = None

        self._ponder_steps = []
        self._remainders = []
        self._memory_divergences = []

        self._main_dnc_not_created = True
        self._divergence_type = divergence_type

    @property
    def state_size(self):
        return (self._main_dnc.state_size,
                self._aux_dnc.state_size)

    @property
    def output_size(self):
        return self._main_dnc.output_size

    def get_ponder_steps(self, sequence_length=None):
        if len(self._ponder_steps) == 0:
            raise RuntimeError("ponder_steps should be invoked after all call()'s")
        if self._ponder_steps_op is None:
            stacked_steps = tf.stack(self._ponder_steps)
            if sequence_length is not None:
                mask = tf.sequence_mask(sequence_length, len(self._remainders))
                mask = tf.expand_dims(mask, axis=0) # to correct inconsistency between 2 act model
                stacked_steps *= tf.transpose(tf.cast(mask, stacked_steps.dtype))
            self._ponder_steps_op = stacked_steps
        return self._ponder_steps_op

    def get_ponder_cost(self, sequence_length=None):
        if len(self._remainders) == 0:
            raise RuntimeError("ponder_cost should be invoked after all call()'s")
        if self._ponder_cost_op is None:
            stacked_remainders = tf.stack(self._remainders)
            if sequence_length is not None:
                mask = tf.sequence_mask(sequence_length, len(self._remainders))
                mask = tf.expand_dims(mask, axis=0) # to correct inconsistency between 2 act model
                stacked_remainders *= tf.transpose(tf.cast(mask, stacked_remainders.dtype))
            batch_size = tf.cast(tf.shape(self._remainders[0])[0], stacked_remainders.dtype)
            self._ponder_cost_op = tf.reduce_sum(stacked_remainders) / batch_size
        return self._ponder_cost_op

    def get_memory_divergence_loss(self):
        if len(self._memory_divergences) == 0:
            raise RuntimeError("memory divergence loss should be invoked after all call()'s")
        return tf.reduce_mean(tf.stack(self._memory_divergences))

    def __call__(self, inputs, state, scope=None):
        # with _checked_scope(self, scope or "act_wrapper", reuse=self._reuse):
        with tf.variable_scope(scope or "act_wrapper", reuse=self._reuse):
            # batch_size = tf.shape(inputs)[0]
            batch_size = inputs.get_shape()[0]
            if isinstance(state, tuple):
                state_is_tuple = True
                state_tuple_type = type(state)
            else:
                state_is_tuple = False
                state_tuple_type = None

            inputs_and_zero = tf.concat([inputs, tf.fill([batch_size, 1], 0.0)], 1)
            inputs_and_one = tf.concat([inputs, tf.fill([batch_size, 1], 1.0)], 1)
            # zero_state = tf.convert_to_tensor(self._cell.zero_state(batch_size, state[0].dtype))
            # if isinstance(self._cell, dnc.DNC):
            #     zero_state = self._cell.initial_state(batch_size)
            # else:
            #     zero_state = self._cell.zero_state(batch_size, state[0].dtype)
            main_zero_state = self._main_dnc.initial_state(batch_size)
            aux_zero_state = self._aux_dnc.initial_state(batch_size)
            main_zero_output = tf.fill([batch_size, self._main_dnc.output_size[0]], tf.constant(0.0, tf.float32))
            aux_zero_output = tf.fill([batch_size, self._aux_dnc.output_size[0]], tf.constant(0.0, tf.float32))

            # pre-create variable if has not created
            if self._main_dnc_not_created:
                self._main_dnc_not_created = False
                main_inputs = tf.concat([inputs_and_one, batch_flatten(aux_zero_output)], 1)
                self._main_dnc(main_inputs, main_zero_state)

            def cond(finished, *_):
                return tf.reduce_any(tf.logical_not(finished))

            def body(previous_finished, time_step,
                     previous_state, running_output, running_state,
                     ponder_steps, remainders, running_p_sum, prev_aux_output, prev_aux_state):

                current_inputs = tf.where(tf.equal(time_step, 1), inputs_and_one, inputs_and_zero)
                # current_output, current_state = self._cell(current_inputs, previous_state)
                read_words, main_current_state = dnc_read(current_inputs, aux_zero_output, self._main_dnc, previous_state)
                aux_inputs = tf.concat(
                        [batch_flatten(current_inputs), batch_flatten(read_words)], 1)
                aux_current_output, aux_current_state = self._aux_dnc(aux_inputs, prev_aux_state)

                # TODO
                joint_current_state = tf.concat(aux_current_state.controller_state, 1)

                # current_h = tf.nn.sigmoid(tf.squeeze(
                #     _linear([joint_current_state], 1, True, self._init_halting_bias), 1
                # ))
                try:
                    current_h = tf.squeeze(tf.layers.dense(joint_current_state,
                                                           units=1,
                                                           activation=tf.sigmoid),
                                           axis=1)
                except ValueError:
                    current_h = tf.squeeze(tf.layers.dense(joint_current_state,
                                                           units=1,
                                                           activation=tf.sigmoid,
                                                           reuse=True),
                                           axis=1)

                current_h_sum = running_p_sum + current_h

                limit_condition = time_step >= self._ponder_limit
                halting_condition = current_h_sum >= 1.0 - self._epsilon
                current_finished = tf.logical_or(halting_condition, limit_condition)
                just_finished = tf.logical_xor(current_finished, previous_finished)

                current_p = tf.where(current_finished, 1.0 - running_p_sum, current_h)
                expanded_current_p = tf.expand_dims(current_p, 1)

                running_output += expanded_current_p # * read_words # of no use

                running_state = choose_dnc_state(previous_finished, running_state, main_current_state, self._main_dnc._state_size)

                ponder_steps = tf.where(just_finished, tf.fill([batch_size], time_step), ponder_steps)
                remainders = tf.where(just_finished, current_p, remainders)
                running_p_sum += current_p

                return (current_finished, time_step + 1,
                        main_current_state, running_output, running_state,
                        ponder_steps, remainders, running_p_sum,
                        aux_current_output, aux_current_state)
            _, _, _, final_output, final_state, all_ponder_steps, all_remainders, _, aux_output, aux_state = \
                tf.while_loop(cond, body, [
                    tf.fill([batch_size], False), tf.constant(1), state, main_zero_output, main_zero_state,
                    tf.fill([batch_size], 0), tf.fill([batch_size], 0.0), tf.fill([batch_size], 0.0),
                    aux_zero_output, aux_zero_state])
            # all_ponder_steps = tf.Print(
            #         all_ponder_steps,
            #         data=[all_ponder_steps],
            #         message="ponder_steps: ",
            #         summarize=256)
            inputs_and_one = tf.concat([inputs, tf.fill([batch_size, 1], 1.0)], 1)
            main_inputs = tf.concat([inputs_and_one, batch_flatten(aux_output)], 1)
            final_output, final_state = self._main_dnc(main_inputs, final_state)

            self._ponder_steps.append(all_ponder_steps)
            self._remainders.append(all_remainders)
            self._memory_divergences.append(self._main_dnc.get_memory_kl_divergence(self._divergence_type))

            return final_output, final_state
