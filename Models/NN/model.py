import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.python.ops import variable_scope
from tensorflow.python.framework import dtypes
import numpy as np
import copy

def seq2seq(feed_previous=False, input_dim=1, output_dim=1, input_length=120,
    output_length=48, hidden_dim=64, stacked_layers=2, GRADIENT_CLIPPING=2.5):

    tf.reset_default_graph()
    global_step = tf.Variable(initial_value=0, name="global_step", trainable=False,
                  collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])   

    weights = {
        'out': tf.get_variable('Weights_out', shape = [hidden_dim, output_dim], \
                               dtype = tf.float32, \
                               initializer = tf.truncated_normal_initializer()),
    }
    biases = {
        'out': tf.get_variable('Biases_out', shape = [output_dim], \
                               dtype = tf.float32, \
                               initializer = tf.constant_initializer(0.)),
    }
                                          
    with tf.variable_scope('Seq2seq'):
        encoder_input = [
            tf.placeholder(tf.float32, shape=(None, input_dim), name="input_{}".format(t))
               for t in range(input_length)
        ]

        target_sequence = [
            tf.placeholder(tf.float32, shape=(None, output_dim), name="y".format(t))
              for t in range(output_length)
        ]

        decoder_input = [tf.zeros_like(target_sequence[0], dtype=tf.float32, name="GO")] + target_sequence[:-1]

        with tf.variable_scope('LSTMCell'): 
            cells = []
            for i in range(stacked_layers):
                with tf.variable_scope('RNN_{}'.format(i)):
                    cells.append(tf.contrib.rnn.LSTMCell(hidden_dim))
            cell = tf.contrib.rnn.MultiRNNCell(cells)

        with variable_scope.variable_scope('basic_rnn_seq2seq'):
            encoder_cell = copy.deepcopy(cell)
            _, encoder_state = rnn.static_rnn(encoder_cell, encoder_input, dtype=dtypes.float32)

            with variable_scope.variable_scope('rnn_decoder'):
                state = encoder_state
                outputs = []
                for i, input_ in enumerate(decoder_input):
                    if i > 0:
                        variable_scope.get_variable_scope().reuse_variables()
                    output, state = cell(input_, state)
                    outputs.append(output)

            reshaped = [tf.matmul(i, weights['out']) + biases['out'] for i in outputs]
            return encoder_input, target_sequence, reshaped ,global_step

if __name__ == '__main__':
    X, y, predict, global_step = seq2seq()
    with tf.variable_scope('loss'):
        output_loss = 0
        for _y, _Y in zip(predict, y):
            output_loss += tf.reduce_mean(tf.pow(_y - _Y, 2))

        l2loss = 0
        for v in tf.trainable_variables():
            if 'Biases_' in v.name or 'Weights_' in v.name:
                l2loss += tf.reduce_mean(tf.nn.l2_loss(v))

        loss = output_loss + 0.003 * l2loss
        # parameter lambda l2 regulaization

    with tf.variable_scope('optimizer'):
        optimizer = tf.contrib.layers.optimize_loss(loss=loss, learning_rate=0.01,
                global_step=global_step, optimizer='Adam', clip_gradients=2.5)
        # parameter learning_rate clip_gradients

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        feed = {X[t]:np.random.rand(10, 1) for t in range(120)}
        feed.update({y[t]: np.random.rand(10, 1) for t in range(48)})
        session.run(optimizer, feed_dict=feed)
        print('finish')

