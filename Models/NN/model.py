import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.python.ops import variable_scope
from tensorflow.python.framework import dtypes
import numpy as np
import copy
import dataset

bj = ['qianmen_aq', 'miyun_aq', 'yongledian_aq', 'dongsihuan_aq', 
    'fengtaihuayuan_aq', 'dingling_aq', 'daxing_aq', 'huairou_aq', 
    'zhiwuyuan_aq', 'mentougou_aq', 'tongzhou_aq', 'yufa_aq', 
    'guanyuan_aq', 'nansanhuan_aq', 'gucheng_aq', 'shunyi_aq', 
    'wanliu_aq', 'aotizhongxin_aq', 'dongsi_aq', 'liulihe_aq', 
    'pingchang_aq', 'badaling_aq', 'xizhimenbei_aq', 'yongdingmennei_aq', 
    'miyunshuiku_aq', 'tiantan_aq', 'yanqin_aq', 'yungang_aq', 
    'beibuxinqu_aq', 'nongzhanguan_aq', 'yizhuang_aq', 
    'donggaocun_aq', 'fangshan_aq', 'wanshouxigong_aq', 'pinggu_aq',
]

ld = ['CD1', 'BL0', 'GR4', 'MY7', 'HV1', 'GN3', 'GR9', 'LW2', 'GN0', 'KF1', 'CD9', 'ST5', 'TH4']

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
    bj_air = ['PM2.5', 'PM10', 'O3']
    ld_air = ['PM2.5', 'PM10']
    bj.extend(ld)
    for where in bj:
        if where in ld:
            air = ld_air
        else:
            air = bj_air
        for which in air:
            print('-------------------------------------------')
            print(where, which)
            with tf.Session() as session:
                session.run(tf.global_variables_initializer())
                d = dataset.generate(where, which)
                n = 0
                while True:
                    try:
                        n += 1
                        x_, y_ = d.next()
                        feed = {X[t]:x_.reshape((-1, 120))[:,t].reshape((-1, 1)) for t in range(120)}
                        feed.update({y[t]: y_.reshape((-1, 48))[:,t].reshape((-1, 1)) for t in range(48)})
                        _, l = session.run([optimizer, loss], feed_dict=feed)
                        if n % 50 == 0:
                            print("loss after %d iteractions : %.3f" %(n ,l))
                            saver = tf.train.Saver()
                            save_path = saver.save(session, './save/iteraction_%s_%s_%d' % (where[:-3], which, n))
                            print("Checkpoint saved at: ", save_path)
                    except:
                        print("loss after %d iteractions : %.3f" %(n ,l))
                        saver = tf.train.Saver()
                        save_path = saver.save(session, 'iteraction%d' % (n))
                        print("Checkpoint saved at: ", save_path)
                        break