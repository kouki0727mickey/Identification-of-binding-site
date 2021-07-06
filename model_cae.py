import tensorflow as tf
import numpy as np
import plotly.offline as offline
import plotly.graph_objs as go
import os


class Model():
    def __init__(self, learning_rate, input_nodes, penalty, Lambda):
        self.lr          = learning_rate
        self.input_nodes = input_nodes
        self.penalty     = penalty
        self.Lambda      = Lambda
        self.num_lay     = len(self.input_nodes)

        self.encoder_weight_list, self.encoder_bias_list, self.decoder_weight_list, self.decoder_bias_list = self._define_variables()

        #self.filter_summary_encode, self.filter_summary_decode = self._define_summary()
        self.x_train_in_list, self.x_test_in_list = self.define_inputs()
        self.encoder_out_train_ops, self.encoder_out_test_ops = self._define_model_encoder()
        self.decoder_out_train_ops, self.decoder_out_test_ops = self._define_model_decoder()

        self.full_encode_op = self._define_full_encode()

        self.l2_jacobean = self.L2_Jacobean()
        self.train_loss_ops, self.test_loss_ops, self.l2_j, self.cost = self._define_loss()
        self.optimizer_ops = self._define_optimizer()
        self.optimize_ops = self.optimize()

        self.sess = tf.Session()
        self.init_op = tf.global_variables_initializer()
        self.merged = tf.summary.merge_all()

        self.sess.run(self.init_op)


    def _define_full_encode(self):
        full_encode = self.x_test_in_list[0]
        for layer in range(self.num_lay-1):
            full_encode = tf.nn.sigmoid(tf.add(tf.matmul(full_encode, self.encoder_weight_list[layer]), self.encoder_bias_list[layer]))

        return full_encode


    def L2_Jacobean(self):
        F_J = []
        for layer in range(self.num_lay - 1):
            grads = tf.gradients(self.encoder_out_train_ops[layer], self.x_train_in_list[layer])
            sqrt_grads = tf.square(grads)
            F_J.append(tf.reduce_sum(sqrt_grads))
        return F_J


    def _define_variables(self):
        list_encoder_weight, list_encoder_bias, list_decoder_weight, list_decoder_bias = [],[],[],[]
        for i in range(self.num_lay-1):
            list_encoder_weight.append(tf.Variable(tf.truncated_normal([self.input_nodes[i], self.input_nodes[i + 1]], stddev=0.1, dtype=tf.float32)))
            list_encoder_bias.append(tf.Variable(tf.truncated_normal([self.input_nodes[i + 1]], stddev=0.1, dtype=tf.float32)))
            #shared weights
            list_decoder_weight.append(tf.transpose(list_encoder_weight[i]))
            #list_decoder_weight.append(tf.Variable(tf.truncated_normal([self.input_nodes[i + 1], self.input_nodes[i]], stddev=0.1, dtype=tf.float32)))
            list_decoder_bias.append(tf.Variable(tf.truncated_normal([self.input_nodes[i]], stddev=0.1, dtype=tf.float32)))

        return list_encoder_weight, list_encoder_bias, list_decoder_weight, list_decoder_bias


    def _define_model_encoder(self):
        encoder_train_ops, encoder_test_ops= [],[]
        for layer in range(self.num_lay-1):
            encoder_output = tf.nn.sigmoid(tf.add(tf.matmul(self.x_train_in_list[layer], self.encoder_weight_list[layer]), self.encoder_bias_list[layer]))
            encoder_train_ops.append(encoder_output)
            encoder_output = tf.nn.sigmoid(tf.add(tf.matmul(self.x_test_in_list[layer], self.encoder_weight_list[layer]), self.encoder_bias_list[layer]))
            encoder_test_ops.append(encoder_output)


        return encoder_train_ops, encoder_test_ops


    def _define_model_decoder(self):
        decoder_train_ops, decoder_test_ops = [],[]
        for layer in range(self.num_lay-1):
            decoder_output = tf.add(tf.matmul(self.encoder_out_train_ops[layer], self.decoder_weight_list[layer]), self.decoder_bias_list[layer])
            decoder_train_ops.append(decoder_output)
            decoder_output = tf.add(tf.matmul(self.encoder_out_test_ops[layer], self.decoder_weight_list[layer]), self.decoder_bias_list[layer])
            decoder_test_ops.append(decoder_output)

        return decoder_train_ops, decoder_test_ops


    def define_inputs(self):
        x_train_list, x_test_list = [],[]
        for layer in range(self.num_lay-1):
            x_train_input_shape = (None, self.input_nodes[layer])
            x_train_list.append(tf.placeholder(tf.float32, shape=(x_train_input_shape)))
            x_test_input_shape = (None, self.input_nodes[layer])
            x_test_list.append(tf.placeholder(tf.float32, shape=(x_test_input_shape)))

        return x_train_list, x_test_list


    def _define_loss(self):
        loss_type = tf.losses.mean_squared_error
        train_loss_list, test_loss_list, fb, cost = [], [], [], []
        for layer in range(self.num_lay-1):
            loss = loss_type(labels=self.x_train_in_list[layer], predictions=self.decoder_out_train_ops[layer])
            cost.append(loss)
            if self.penalty:
                fb.append(self.Lambda*self.l2_jacobean[layer])
                loss += self.Lambda*self.l2_jacobean[layer]
            train_loss_list.append(loss)
            loss = loss_type(labels=self.x_test_in_list[layer], predictions=self.decoder_out_test_ops[layer])
            test_loss_list.append(loss)

        return train_loss_list, test_loss_list, fb, cost


    def _define_optimizer(self):
        optimizer_list = []
        for layer in range(self.num_lay-1):
            optimizer_list.append(tf.train.AdamOptimizer(learning_rate=self.lr[layer]))

        return optimizer_list


    def optimize(self):
        optimize_list = []
        for layer in range(self.num_lay-1):
            opt = self.optimizer_ops[layer].minimize(self.train_loss_ops[layer])
            optimize_list.append(opt)

        return optimize_list


class Plot():
    def __init__(self,data,category,name, file_source_id):
        self.data = data
        self.category = category
        self.name = name
        self.file_source_id = file_source_id

    # matplotlib, seaborn, e.t.c.
    def etc(self):
        return 0

    def plotly(self):
        trace_list=[]
        if np.shape(self.data)[1] == 3:
            for i in range(len(self.category) - 1):
                trace_list.append(go.Scatter3d(
                    x=self.data[self.category[i]:self.category[i+1], 0],  # それぞれの次元をx, y, zにセットするだけです．
                    y=self.data[self.category[i]:self.category[i+1], 1],
                    z=self.data[self.category[i]:self.category[i+1], 2],
                    mode='markers',
                    marker=dict(
                        sizemode='diameter',
                        opacity=0.9,
                        size=3  # ごちゃごちゃしないように小さめに設定するのがオススメです．
                    )))

        else:
            for i in range(len(self.category) - 1):
                trace.append(go.Scatter(
                    x=self.data[self.category[i]:self.category[i+1], 0],  # それぞれの次元をx, y, zにセットするだけです．
                    y=self.data[self.category[i]:self.category[i+1], 1],
                    mode='markers',
                    marker=dict(
                        sizemode='diameter',
                        opacity=0.9,
                        size=6  # ごちゃごちゃしないように小さめに設定するのがオススメです．
                    )))

        data = trace_list
        layout = dict(hovermode='closest', height=700, width=600, title='plot')
        fig = dict(data=data, layout=layout)

        name=self.file_source_id+'/'+self.name
        file_name=name+'_1.html'
        index=2
        while os.path.exists(file_name) == 1:
            file_name = name+'_'+str(index)+'.html'
            index += 1

        offline.plot(fig, filename=file_name)
