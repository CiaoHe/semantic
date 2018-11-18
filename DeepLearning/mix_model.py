import tensorflow as tf
import numpy as np

rnn_cells = {
    'GRU': tf.nn.rnn_cell.GRUCell,
    'LSTM': tf.nn.rnn_cell.BasicLSTMCell,
}

class MixModel():
    def __init__(self, max_sequence_len, lr, model_type, num_features):

        self.embedded_x1 = tf.placeholder(dtype=tf.float32, shape=[None, 300, max_sequence_len])
        self.embedded_x2 = tf.placeholder(dtype=tf.float32, shape=[None, 300, max_sequence_len])
        self.labels = tf.placeholder(dtype=tf.float32, shape=[None,])
        # self.ml_features = tf.placeholder(tf.float32,shape=[None,26])
        self.l2_reg = 1e-4

        # cnn 超参
        self.w = 4
        self.di = 50
        self.d0 = 300
        self.num_layers = 1
        self.features = tf.placeholder(tf.float32, shape=[None, 2])
        # lstm 超参
        self.embedding_size = 300
        self.hidden_size = 128
        self.cell_type='LSTM'

        #self.embedded_x1 = tf.placeholder(dtype=tf.float32, shape=[None, 300, max_sequence_len])

        def cos_sim(v1, v2):
            norm1 = tf.sqrt(tf.reduce_sum(tf.square(v1), axis=1))
            norm2 = tf.sqrt(tf.reduce_sum(tf.square(v2), axis=1))
            dot_products = tf.reduce_sum(v1 * v2, axis=1, name="cos_sim")

            return dot_products / (norm1 * norm2)

        def manhantan_diatance(v1,v2):
            return tf.reduce_sum(tf.abs(v1-v2), axis=1)

        def euclidean_score(v1, v2):
            euclidean = tf.sqrt(tf.reduce_sum(tf.square(v1 - v2), axis=1))
            return 1 / (1 + euclidean)

        def dropout(x, is_training, rate=0.2):
            return tf.layers.dropout(x, rate, training=tf.convert_to_tensor(is_training))

        with tf.variable_scope('cnn'):

            def pad_for_wide_conv(x):
                return tf.pad(x, np.array([[0, 0], [0, 0], [self.w - 1, self.w - 1], [0, 0]]), "CONSTANT", name="pad_wide_conv")
            def make_attention_mat(x1, x2):
                # x1, x2 = [batch, height, width, 1] = [batch, d, s, 1]
                # x2 => [batch, height, 1, width]
                # [batch, width, wdith] = [batch, s, s]
                euclidean = tf.sqrt(tf.reduce_sum(tf.square(x1 - tf.matrix_transpose(x2)), axis=1))
                return 1 / (1 + euclidean)

            def convolution(name_scope, x, d, reuse):
                with tf.name_scope(name_scope + "-conv"):
                    with tf.variable_scope("conv") as scope:
                        conv = tf.contrib.layers.conv2d(
                            inputs=x,
                            num_outputs=self.di,
                            kernel_size=(d, self.w),
                            stride=1,
                            padding="VALID",
                            activation_fn=tf.nn.tanh,
                            weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                            weights_regularizer=tf.contrib.layers.l2_regularizer(scale=self.l2_reg),
                            biases_initializer=tf.constant_initializer(1e-04),
                            reuse=reuse,
                            trainable=True,
                            scope=scope
                        )
                        # Weight: [filter_height, filter_width, in_channels, out_channels]
                        # output: [batch, 1, input_width+filter_Width-1, out_channels] == [batch, 1, s+w-1, di]

                        # [batch, di, s+w-1, 1]
                        conv_trans = tf.transpose(conv, [0, 3, 2, 1], name="conv_trans")
                        return conv_trans

            def w_pool(variable_scope, x, attention):
                # x: [batch, di, s+w-1, 1]
                # attention: [batch, s+w-1]
                with tf.variable_scope(variable_scope + "-w_pool"):
                    if model_type == "ABCNN2" or model_type == "ABCNN3":
                        pools = []
                        # [batch, s+w-1] => [batch, 1, s+w-1, 1]
                        attention = tf.transpose(tf.expand_dims(tf.expand_dims(attention, -1), -1), [0, 2, 1, 3])

                        for i in range(max_sequence_len):
                            # [batch, di, w, 1], [batch, 1, w, 1] => [batch, di, 1, 1]
                            pools.append(tf.reduce_sum(x[:, :, i:i + self.w, :] * attention[:, :, i:i + self.w, :],
                                                       axis=2,
                                                       keep_dims=True))

                        # [batch, di, s, 1]
                        w_ap = tf.concat(pools, axis=2, name="w_ap")
                    else:
                        w_ap = tf.layers.average_pooling2d(
                            inputs=x,
                            # (pool_height, pool_width)
                            pool_size=(1, self.w),
                            strides=1,
                            padding="VALID",
                            name="w_ap"
                        )
                        # [batch, di, s, 1]

                    return w_ap

            def all_pool(variable_scope, x):
                with tf.variable_scope(variable_scope + "-all_pool"):
                    if variable_scope.startswith("input"):
                        pool_width = max_sequence_len
                        d = self.d0
                    else:
                        pool_width = max_sequence_len + self.w - 1
                        d = self.di

                    all_ap = tf.layers.average_pooling2d(
                        inputs=x,
                        # (pool_height, pool_width)
                        pool_size=(1, pool_width),
                        strides=1,
                        padding="VALID",
                        name="all_ap"
                    )
                    # [batch, di, 1, 1]

                    # [batch, di]
                    all_ap_reshaped = tf.reshape(all_ap, [-1, d])
                    #all_ap_reshaped = tf.squeeze(all_ap, [2, 3])

                    return all_ap_reshaped

            def CNN_layer(variable_scope, x1, x2, d):
                # x1, x2 = [batch, d, s, 1]
                with tf.variable_scope(variable_scope):
                    if model_type == "ABCNN1" or model_type == "ABCNN3":
                        with tf.name_scope("att_mat"):
                            aW = tf.get_variable(name="aW",
                                                 shape=(max_sequence_len, d),
                                                 initializer=tf.contrib.layers.xavier_initializer(),
                                                 regularizer=tf.contrib.layers.l2_regularizer(scale=self.l2_reg))

                            # [batch, s, s]
                            att_mat = make_attention_mat(x1, x2)

                            # [batch, s, s] * [s,d] => [batch, s, d]
                            # matrix transpose => [batch, d, s]
                            # expand dims => [batch, d, s, 1]
                            x1_a = tf.expand_dims(tf.matrix_transpose(tf.einsum("ijk,kl->ijl", att_mat, aW)), -1)
                            x2_a = tf.expand_dims(tf.matrix_transpose(
                                tf.einsum("ijk,kl->ijl", tf.matrix_transpose(att_mat), aW)), -1)

                            # [batch, d, s, 2]
                            x1 = tf.concat([x1, x1_a], axis=3)
                            x2 = tf.concat([x2, x2_a], axis=3)

                    left_conv = convolution(name_scope="left", x=pad_for_wide_conv(x1), d=d, reuse=False)
                    right_conv = convolution(name_scope="right", x=pad_for_wide_conv(x2), d=d, reuse=True)

                    left_attention, right_attention = None, None

                    if model_type == "ABCNN2" or model_type == "ABCNN3":
                        # [batch, s+w-1, s+w-1]
                        att_mat = make_attention_mat(left_conv, right_conv)
                        # [batch, s+w-1], [batch, s+w-1]
                        left_attention, right_attention = tf.reduce_sum(att_mat, axis=2), tf.reduce_sum(att_mat, axis=1)

                    left_wp = w_pool(variable_scope="left", x=left_conv, attention=left_attention)
                    left_ap = all_pool(variable_scope="left", x=left_conv)
                    right_wp = w_pool(variable_scope="right", x=right_conv, attention=right_attention)
                    right_ap = all_pool(variable_scope="right", x=right_conv)

                    return left_wp, left_ap, right_wp, right_ap

        with tf.variable_scope('cnn_out'):
            x1_expanded = tf.expand_dims(self.embedded_x1, -1)
            x2_expanded = tf.expand_dims(self.embedded_x2, -1)

            LO_0 = all_pool(variable_scope="input-left", x=x1_expanded)
            RO_0 = all_pool(variable_scope="input-right", x=x2_expanded)

            LI_1, LO_1, RI_1, RO_1 = CNN_layer(variable_scope="CNN-1", x1=x1_expanded, x2=x2_expanded, d=self.d0)
            sims =[] 
            sims.append(cos_sim(LO_0, RO_0))
            sims.append(cos_sim(LO_1, RO_1))
            
            self.cnn_sen = LO_1 - RO_1
            # sims.append(manhantan_diatance(LO_0, RO_0))
            # sims.append(manhantan_diatance(LO_1, RO_1))

            if self.num_layers > 1:
                _, LO_2, _, RO_2 = CNN_layer(variable_scope="CNN-2", x1=LI_1, x2=RI_1, d=self.di)
                self.test = LO_2
                self.test2 = RO_2
                sims.append(cos_sim(LO_2, RO_2))
                # sims.append(manhantan_diatance(LO_2, RO_2))

            self.cnn_features = sims

        with tf.variable_scope('rnn'):

            def rnn_layer(embedded_x, bidirectional, reuse=False):
                with tf.variable_scope('recurrent', reuse=reuse):
                    cell = rnn_cells[self.cell_type]

                    fw_rnn_cell = cell(self.hidden_size)

                    if bidirectional:
                        bw_rnn_cell = cell(self.hidden_size)
                        rnn_outputs, _ = tf.nn.bidirectional_dynamic_rnn(fw_rnn_cell,
                                                                         bw_rnn_cell,
                                                                         embedded_x,
                                                                         dtype=tf.float32)
                        output = tf.concat([rnn_outputs[0], rnn_outputs[1]], axis=2)
                    else:
                        output, _ = tf.nn.dynamic_rnn(fw_rnn_cell,
                                                      embedded_x,
                                                      dtype=tf.float32)
                return output

        with tf.variable_scope('lstm_out'):
            outputs_sen1 = rnn_layer(tf.transpose(self.embedded_x1, [0, 2, 1]), bidirectional=True)
            outputs_sen2 = rnn_layer(tf.transpose(self.embedded_x2, [0, 2, 1]), bidirectional=True, reuse=True)
            #这里output应该是[None,time_step,hidden_size*2]

            out1 = tf.reduce_mean(outputs_sen1, axis=1)
            out2 = tf.reduce_mean(outputs_sen2, axis=1)
            sim1 = cos_sim(out1, out2)
            self.lstm_sen = out1- out2
            # sim2 = manhantan_diatance(out1, out2)
            self.lstm_features = tf.stack([sim1],axis=1)

        with tf.variable_scope('out'):
            # print(self.cnn_features.shape)
            self.output_features = tf.nn.tanh(tf.concat([tf.stack(self.cnn_features, axis=1), self.lstm_features], axis=1))

        with tf.variable_scope('regression'):

            self.estimation = tf.contrib.layers.fully_connected(
            inputs=self.output_features,
            num_outputs=1,
            activation_fn=None,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            weights_regularizer=tf.contrib.layers.l2_regularizer(scale=self.l2_reg),
            biases_initializer=tf.constant_initializer(1e-04),
            scope="FC"
            )
        with tf.variable_scope('loss'):
            print('================================')
            print(self.estimation)
            self.predictions = self.estimation
            # print(self.predictions)

            # self.y_reshape = tf.reshape(self.labels,[-1,1])
            # print(self.labels.dtype)
            # print(self.predictions.shape
            #self.mse_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.predictions))
            self.reshape_labels = tf.reshape(self.labels,[-1,1])
            self.mse_loss = tf.losses.mean_squared_error(self.reshape_labels,self.predictions)

        with tf.variable_scope('optimizer'):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.mse_loss)
