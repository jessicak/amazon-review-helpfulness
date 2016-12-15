from __future__ import division, print_function, absolute_import

import numpy as np
import tensorflow as tf

class ModelConfig(object):
    lstm_size = 33
    num_lstm_layers = 2
    lstm_keep_prob = 0.70
    learning_rate = 0.001
    fc_layer_sizes = [50, 40, 30]
    n_classes = 2
    n_cols = 32

class AmazonModel(object):

    def __init__(self, config):
        weights, biases = [], []
        prev_size = config.n_cols + config.lstm_size
        for h_size in config.fc_layer_sizes + [config.n_classes]:
            # Xavier initialization
            c = tf.constant(np.sqrt(2.0 / (prev_size * h_size)), dtype=tf.float32)
            w = tf.Variable(tf.scalar_mul(c, tf.random_normal([prev_size, h_size])), 
                name="fc_weights")
            b = tf.Variable(tf.fill([h_size], 0.001), name="fc_biases")
            weights.append(w)
            biases.append(b)
            prev_size = h_size

        self.weights = weights
        self.biases = biases
        self.config = config
        self.n_fc_layers = len(weights)

        # Leaky ReLU slope
        self.alpha = tf.constant(0.01, dtype=tf.float32)

    def regularization_penalty(self):
        penalty = tf.constant(0.0, dtype=tf.float32)
        reg_weight = tf.constant(self.config.reg_weight, dtype=tf.float32)
        for tensor in tf.trainable_variables():
            if str(tensor.name).lower().find('bias') < 0: # Don't regularize bias vectors
                penalty = tf.add(penalty, 
                    tf.scalar_mul(reg_weight, tf.nn.l2_loss(tensor)))
        return penalty

    def build_graph(self):
        weights = self.weights
        biases = self.biases
        config = self.config
        n_fc_layers = self.n_fc_layers
        
        def pred(x, x2, seqlen, lstm_keep_prob, fc_keep_prob):
            # LSTM Module
            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(config.lstm_size, 
                forget_bias=1.0, state_is_tuple=True)
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, 
                output_keep_prob=lstm_keep_prob)
            lstm_module = tf.nn.rnn_cell.MultiRNNCell(
                [lstm_cell] * config.num_lstm_layers, 
                state_is_tuple=True
            )
            outputs, states = tf.nn.dynamic_rnn(lstm_module, x, 
                dtype=tf.float32, sequence_length=seqlen)

            # Retrieve last relevant output
            batch_size = tf.shape(outputs)[0]
            max_length = tf.shape(outputs)[1]
            out_size = int(outputs.get_shape()[2])
            index = tf.range(0, batch_size) * max_length + (seqlen - 1)
            flat = tf.reshape(outputs, [-1, out_size])
            lstm_out = tf.gather(flat, index)
            
            # Feed LSTM output and review features into deep network
            fc_data = tf.concat(1, [lstm_out, x2])
            for i in xrange(n_fc_layers - 1):
                # # ReLU
                # fc_data = tf.nn.dropout(
                #     tf.nn.relu(tf.matmul(fc_data, weights[i]) + biases[i]), 
                #     fc_keep_prob
                # )

                # Leaky ReLU
                preactivation = tf.matmul(fc_data, weights[i]) + biases[i]
                fc_data = tf.nn.dropout(
                    tf.maximum(preactivation,tf.scalar_mul(self.alpha, preactivation)), 
                    fc_keep_prob
                )
            return tf.matmul(fc_data, weights[-1]) + biases[-1]
        return pred
