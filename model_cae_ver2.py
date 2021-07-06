
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() #disable the vihavioe of tensorflow version 2
import numpy as np
from tqdm import tqdm
import plotly.offline as offline
from chart_studio.grid_objs import Grid, Column
import plotly.graph_objs as go
import chart_studio.plotly as py
import matplotlib
import os
matplotlib.rcParams['animation.embed_limit'] = 2**128


class Model():
    def __init__(self, learning_rate, input_nodes, restorer=False):
        print("Model Initializing ...", end="")
        self.lr = learning_rate
        self.input_nodes = input_nodes
        self.num_lay = len(self.input_nodes)
        self.encoder_weight_list, self.encoder_bias_list, self.decoder_weight_list, self.decoder_bias_list = self._define_variables()

        self.x_train_in_list = self.define_inputs()
        self.latent = self.define_latent()

        self.encode_op = self.encode()
        self.decode_op = self.decode()
        self.generate_op = self.generate()

        self.loss_ops = self._define_loss()
        self.optimizer_ops = self._define_optimizer()
        self.optimize_ops = self.optimize()

        self.sess = tf.Session()
        if not restorer:
            self.init_op = tf.global_variables_initializer()
            self.sess.run(self.init_op)
        self.saver = tf.train.Saver()
        print(" Completed!\n")



    def linear(self, input, weight, bias):
        return tf.add(tf.matmul(input, weight), bias)


    def encode(self):
        activation = tf.nn.tanh
        en = self.x_train_in_list[0]
        for layer in range(self.num_lay-1):
            en = activation(self.linear(en, self.encoder_weight_list[layer],self.encoder_bias_list[layer]))
        return en


    # decoder(rebuild)
    def decode(self):
        activation = tf.nn.tanh
        de = self.encode_op
        for layer in range(1, self.num_lay):
            #de = activation(self.linear(de, tf.transpose(self.encoder_weight_list[-layer]), self.decoder_bias_list[self.num_lay-layer-1]))
            de = activation(self.linear(de, self.decoder_weight_list[self.num_lay - layer - 1],
                                        self.decoder_bias_list[self.num_lay - layer - 1]))
        return de


    def generate(self):
        activation = tf.nn.tanh
        de = self.latent
        for layer in range(1, self.num_lay):
            #de = activation(self.linear(de, tf.transpose(self.encoder_weight_list[-layer]), self.decoder_bias_list[self.num_lay-layer-1]))
            de = activation(self.linear(de, self.decoder_weight_list[self.num_lay - layer - 1],
                                        self.decoder_bias_list[self.num_lay - layer - 1]))
        return de


    def _define_variables(self):
        list_encoder_weight, list_encoder_bias, list_decoder_weight, list_decoder_bias = [],[],[],[]
        for i in range(self.num_lay-1):
            list_encoder_weight.append(tf.Variable(tf.truncated_normal([self.input_nodes[i], self.input_nodes[i + 1]], stddev=0.1, dtype=tf.float32)))
            list_encoder_bias.append(tf.Variable(tf.truncated_normal([self.input_nodes[i + 1]], stddev=0.1, dtype=tf.float32)))
            list_decoder_weight.append(tf.Variable(tf.truncated_normal([self.input_nodes[i + 1], self.input_nodes[i]], stddev=0.1, dtype=tf.float32)))
            list_decoder_bias.append(tf.Variable(tf.truncated_normal([self.input_nodes[i]], stddev=0.1, dtype=tf.float32)))

        return list_encoder_weight, list_encoder_bias, list_decoder_weight, list_decoder_bias


    def define_latent(self):
        l = tf.placeholder(tf.float32, shape=(None, self.input_nodes[-1]))

        return l#, x_test_list


    def define_inputs(self):
        x_train_list, x_test_list = [],[]
        for layer in range(self.num_lay-1):
            x_train_input_shape = (None, self.input_nodes[layer])
            x_train_list.append(tf.placeholder(tf.float32, shape=(x_train_input_shape)))
            #x_test_input_shape = (None, self.input_nodes[layer])
            #x_test_list.append(tf.placeholder(tf.float32, shape=(x_test_input_shape)))

        return x_train_list#, x_test_list


    def _define_loss(self):
        loss = tf.losses.mean_squared_error(labels=self.x_train_in_list[0],predictions=self.decode_op)

        return loss


    def _define_optimizer(self):
        opt = tf.train.AdamOptimizer(learning_rate=self.lr)

        return opt


    def optimize(self):
        optimize = self.optimizer_ops.minimize(self.loss_ops)

        return optimize






class Grad():
    def __init__(self,data):
        a = 1


    def mk_featuremap(self, input, id, session):
        import cv2

        input = standardization(input)
        input = (input + abs(np.min(input)))
        max = np.max(input)
        input = input
        cv2.imwrite(id+"/featuremap"+str(session+1)+".png", input)
        #for item in input[0]:
        #    file_trend.write("%s,"%item)
        #file_trend.write("\n")
        length = len(input[0])
        file3 = open(id + '/spectrum_histogram'+str(session+1)+'.csv', 'w')
        writer_histo = csv.writer(file3, lineterminator='\n')
        for j in range(len(input)):
            list = [0 for i in range(256)]
            for i in range(length):
                list[int(input[j][i])] = list[int(input[j][i])] + 1
            writer_histo.writerow(list)
        #save the last sequence of histogram
        writer_trend.writerow(list)
        file3.close()

class Plot():

    def __init__(self,data,category,name,r_name,label,training_iter):
    #def __init__(self,data,category,name,r_name,training_iter):
        self.data = np.array(data, dtype=float)
        self.category = category
        self.name = name
        self.r_name = r_name
        self.label = label
        self.training_iter = training_iter

    def plotly(self, frame_num=-1, allplot=True):
        if frame_num == -1:
            f_num = self.data.shape[0]
        else:
            f_num = frame_num
        output=self.data[f_num - 1, :, :]

        trace_list = []
        if np.shape(output)[1] == 3:
            if allplot:
                trace_list.append(go.Scatter3d(
                    x=output[:, 0],
                    y=output[:, 1],
                    z=output[:, 2],
                    mode='markers',
                    marker=dict(
                        sizemode='diameter',
                        color = 'grey',
                        opacity=0.2,
                        size=2  # ごちゃごちゃしないように小さめに設定するのがオススメです．
                    ),
                    name='000_all'
                ))
            for i in range(len(self.category) - 1):
                trace_list.append(go.Scatter3d(
                    x=output[self.category[i]:self.category[i + 1], 0],
                    y=output[self.category[i]:self.category[i + 1], 1],
                    z=output[self.category[i]:self.category[i + 1], 2],
                    mode='markers',
                    marker=dict(
                        sizemode='diameter',
                        opacity=0.9,
                        size=2  # ごちゃごちゃしないように小さめに設定するのがオススメです．
                    )
                    #marker={"size": 3},
                    #name=self.label[i]
                ))
        data = trace_list
        layout = go.Layout(hovermode='closest', height=700, width=600, title=self.name)  # 'i='+str(f_num))
        fig = go.Figure(data=data, layout=layout)

        name = self.r_name + '/' + self.name
        file_name = name + '_1.html'
        index = 2
        while os.path.exists(file_name) == 1:
            file_name = name + '_' + str(index) + '.html'
            index += 1
        offline.plot(fig, filename=file_name)

        """if frame_num == -1:
            offline.plot(fig, filename=self.r_name + '.html')
        else:
            offline.plot(fig, filename=self.r_name + '(' + str(f_num) + ').html')"""


    def mk_scaled_data(self):
        def COG(df):
            m = []
            n = []

            for i in range(np.shape(df)[1]):
                #meanからmedianに変更 by baba
                m.append(np.max([df[j][i] for j in range(len(df))]))
                n.append(np.min([df[j][i] for j in range(len(df))]))
                lim_nums = [(x + y)/2 for (x, y) in zip(m, n)]
            return lim_nums
        def subtract(df):
            m = COG(df)
            for i in range(np.shape(df)[1]):
                for j in range(len(df)):
                    df[j][i] -= m[i]
            return df

        centered = []
        for df in self.data:
            centered.append(subtract(df))
        scaled_data = centered[:]
        return scaled_data

    def mk_lims(self):
        last_output = self.data[self.data.shape[0]-1,:,:]
        lims=(np.max(last_output,axis=0)-np.min(last_output,axis=0))/2
        return lims

    #3d plot with label indicating num
    def matplot_anime_3d_with_label(self,rotation=(True,4)):
        out = self.data
        from matplotlib import pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.animation as animation
        color = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w','violet','goldenrod','lightgreen','black','grey']
        print(np.shape(out))
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        bar_template = "\r[{0}] {1}/{2} {3}"
        def animate(i):
            j = int((i+1)*30/np.shape(out)[0])
            bar = "#" * j + " " * (30-j)
            print(bar_template.format(bar, i+1, len(out), '読み込み'), end="")
            d = out[i]
            ax.cla()
            ax.view_init(25,90)
            # current_iter * nb_circulate * skipped_iter * 360° / total_iteration
            if rotation[0]:
                ax.view_init(25,i*rotation[1]*360/101)#*50*360/self.training_iter)
            #ax.view_init(40, 10)
            for j in range(len(self.category)-1):
                #x, y, z = [], [], []
                x = [d[k,0] for k in range(self.category[j],self.category[j+1])]
                y = [d[k,1] for k in range(self.category[j],self.category[j+1])]
                z = [d[k,2] for k in range(self.category[j],self.category[j+1])]
                ax.scatter(x, y, z, s=5, alpha=1.,c=color[j],label=self.label[j])
            ax.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0, fontsize=5)

            ax.set_title(self.name+' i=' + str(i))
        ani = animation.FuncAnimation(fig, animate, frames=len(out), interval=100)
        s = ani.to_jshtml()
        anime =  '_anime(r)' if rotation[0] else '_anime'
        with open(self.r_name+anime+'.html', 'w') as f:
            f.write(s)
        print(bar_template.format('#'*30, len(out), len(out), '出力完了'))
        #plt.show()
        #ani.save("output.gif")#, writer="imagemagick")]

    #3d plot with label indicating num(SameScale ver.)
    def matplot_anime_3d_with_label_samescale(self,out,lim_nums,rotation=(True,4)):
        from matplotlib import pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.animation as animation
        color = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w','violet','goldenrod','lightgreen','black','grey']
        print(np.shape(out))
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        bar_template = "\r[{0}] {1}/{2} {3}"
        def animate(i):
            j = int((i+1)*30/np.shape(out)[0])
            bar = "#" * j + " " * (30-j)
            print(bar_template.format(bar, i+1, len(out), '読み込み'), end="")
            d = out[i]
            ax.cla()
            ax.view_init(25,90)
            # current_iter * nb_circulate * skipped_iter * 360° / total_iteration
            if rotation[0]:
                ax.view_init(25,i*rotation[1]*50*360/self.training_iter)
            if 1:#not autoscale:#→⦿↑
                """
                lim_num =[0.07,0.1,0.03]
                a = 0
                b = 1
                c = 2
                ax.set_xlim(-lim_num[a],lim_num[a])
                ax.set_ylim(-lim_num[b],lim_num[b])
                ax.set_zlim(-lim_num[c],lim_num[c])
                """
                ax.set_xlim(-lim_nums[0],lim_nums[0])
                ax.set_ylim(-lim_nums[1],lim_nums[1])
                ax.set_zlim(-lim_nums[2],lim_nums[2])
            #ax.view_init(40, 10)
            for j in range(len(self.category)-1):
                #x, y, z = [], [], []
                x = [d[k,0] for k in range(self.category[j],self.category[j+1])]
                y = [d[k,1] for k in range(self.category[j],self.category[j+1])]
                z = [d[k,2] for k in range(self.category[j],self.category[j+1])]
                ax.scatter(x, y, z, s=5, alpha=1.,c=color[j],label=self.label[j])
            ax.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0, fontsize=5)
            ax.set_title(self.name+' i=' + str(i))
        ani = animation.FuncAnimation(fig, animate, frames=len(out), interval=100)
        s = ani.to_jshtml()
        anime =  '_anime(s,r)' if rotation[0] else '_anime(s)'
        with open(self.r_name+anime+'.html', 'w') as f:
            f.write(s)
        print(bar_template.format('#'*30, len(out), len(out), '出力完了'))
        #plt.show()
        #ani.save("output.gif")#, writer="imagemagick")


    def anime(self,out):

        column = [Column(out[int(i/3)][:,i%3],str(i)) for i in range(int(len(out)*3))]
        grid = Grid(column)
        py.grid_ops.upload(grid, 'grid' + str(time.time()), auto_open=False)

        figure = {
            'data': [
                {
                    'xsrc': grid.get_column_reference(str(0)),
                    'ysrc': grid.get_column_reference(str(1)),
                    'zsrc': grid.get_column_reference(str(2)),
                    #'x': out[0][:, 0],
                    #'y': out[0][:, 1],
                    #'z': out[0][:, 2],
                    'mode': 'markers',
                }
            ],
            'layout': {'title': 'Ping Pong Animation',
                       'xaxis': {'range': [-1, 1], 'autorange': False},
                       'yaxis': {'range': [-1, 1], 'autorange': False},
                       'zaxis': {'range': [-1, 1], 'autorange': False},
                       'updatemenus': [{
                           'buttons': [
                               {'args': [None],
                                'label': 'Time Lapse',
                                'method': 'animate'}
                           ],
                           'pad': {'r': 10, 't': 87},
                           'showactive': False,
                           'type': 'buttons'
                       }]},
            'frames': [
                {
                    'data': [
                        {
                            'xsrc': grid.get_column_reference(str(i)),
                            'ysrc': grid.get_column_reference(str(i+1)),
                            'zsrc': grid.get_column_reference(str(i+2)),
                            #'x': d[:, 0],
                            #'y': d[:, 1],
                            #'z': d[:, 2],
                            'mode': 'markers',
                        }
                    ]
                } for i in range(int((len(out)-1)*3))
            ]
        }

        py.create_animations(figure, 'anime' + str(time.time()))


    def anime3d(self,out):
        print(len(out))
        column = [Column(out[int(i/2)][:,i%2],str(i)) for i in range(int(len(out)*2))]
        grid = Grid(column)
        py.grid_ops.upload(grid, 'grid' + str(time.time()), auto_open=False)

        figure = {
            'data': [
                {
                    'xsrc': grid.get_column_reference(str(0)),
                    'ysrc': grid.get_column_reference(str(1)),
                    #'x': out[0][:, 0],
                    #'y': out[0][:, 1],
                    #'z': out[0][:, 2],
                    'mode': 'markers',
                }
            ],
            'layout': {'title': 'Ping Pong Animation',
                       'xaxis': {'range': [-1, 1], 'autorange': False},
                       'yaxis': {'range': [-1, 1], 'autorange': False},
                       'updatemenus': [{
                           'buttons': [
                               {'args': [None],
                                'label': 'Time Lapse',
                                'method': 'animate'}
                           ],
                           'pad': {'r': 10, 't': 87},
                           'showactive': False,
                           'type': 'buttons'
                       }]},
            'frames': [
                {
                    'data': [
                        {
                            'xsrc': grid.get_column_reference(str(i)),
                            'ysrc': grid.get_column_reference(str(i+1)),
                            #'x': d[:, 0],
                            #'y': d[:, 1],
                            #'z': d[:, 2],
                            'mode': 'markers',
                        }
                    ]
                } for i in range(0,int(len(out))*2,2)
            ]
        }

        py.create_animations(figure, 'anime' + str(time.time()))

    #plot without label
    def matplot_anime(self,out,name):
        from matplotlib import pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.animation as animation

        print(np.shape(out))
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        def animate(i):
            ax.cla()
            #ax.set_xlim(-1, 1)
            #ax.set_ylim(-1, 1)
            #ax.set_zlim(-1, 1)
            x, y, z = [], [], []
            x = out[i][:, 0]
            y = out[i][:, 1]
            z = out[i][:, 2]
            ax.view_init(40, 10)
            ax.scatter(x, y, z, s=5, alpha=1.,c="blue")
            """
            if not i==0:
                ax.scatter(out[i-1][:, 0], out[i-1][:, 1], out[i-1][:, 2], s=5, alpha=0.2,c="blue")
            if i>1:
                ax.scatter(out[i - 2][:, 0], out[i - 2][:, 1], out[i - 2][:, 2], s=5, alpha=0.1, c="blue")
            """
            ax.set_title("Mapping " + 'i=' + str(i))

        ani = animation.FuncAnimation(fig, animate, frames=len(out), interval=100)

        s = ani.to_jshtml()
        with open( 'animation_'+name+'.html', 'w') as f:
            f.write(s)

        #ani.save("output.gif", writer="pillow")

    """
    #3d with label indicating class
    def matplot_anime_mnist(self,out,label):
        from matplotlib import pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.animation as animation
        color = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w','violet','goldenrod','lightgreen','black']
        print(np.shape(out))
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        def animate(i):
            d = out[i]
            ax.cla()
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_zlim(-1, 1)
            ax.view_init(40, 10)
            for j in set(label):
                #x, y, z = [], [], []
                x = [d[k, 0] for k in range(len(d)) if label[k]==j]
                y = [d[k, 1] for k in range(len(d)) if label[k]==j]
                z = [d[k, 2] for k in range(len(d)) if label[k]==j]
                ax.scatter(x, y, z, s=5, alpha=1.,c=color[j])

            if not i==0:
                ax.scatter(out[i-1][:, 0], out[i-1][:, 1], out[i-1][:, 2], s=5, alpha=0.2,c="blue")
            if i>1:
                ax.scatter(out[i - 2][:, 0], out[i - 2][:, 1], out[i - 2][:, 2], s=5, alpha=0.1, c="blue")

            ax.set_title("Mapping " + 'i=' + str(i))

        ani = animation.FuncAnimation(fig, animate, frames=len(out), interval=100)

        ani.save("output.gif", writer="imagemagick")
        plt.show()
        """

    #3d with label indicating class
    def matplot_anime_mnist(self,out,label):
        from matplotlib import pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.animation as animation
        color = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w','violet','goldenrod','lightgreen','black']
        print(np.shape(out))
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        def animate(i):
            d = out[i]
            ax.cla()
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_zlim(-1, 1)
            ax.view_init(40, 10)
            for j in set(label):
                #x, y, z = [], [], []
                x = [d[k, 0] for k in range(len(d)) if label[k]==j]
                y = [d[k, 1] for k in range(len(d)) if label[k]==j]
                z = [d[k, 2] for k in range(len(d)) if label[k]==j]
                ax.scatter(x, y, z, s=5, alpha=1.,c=color[j])
            """
            if not i==0:
                ax.scatter(out[i-1][:, 0], out[i-1][:, 1], out[i-1][:, 2], s=5, alpha=0.2,c="blue")
            if i>1:
                ax.scatter(out[i - 2][:, 0], out[i - 2][:, 1], out[i - 2][:, 2], s=5, alpha=0.1, c="blue")
            """
            ax.set_title("Mapping " + 'i=' + str(i))

        ani = animation.FuncAnimation(fig, animate, frames=len(out), interval=100)
        s = ani.to_jshtml()
        with open( 'anim.html', 'w') as f:
            f.write(s)
        #ani.save("output.gif", writer="pillow")
        #plt.show()


    #with label indicating num
    def matplot_anime_2d_with_label(self,out):
        from matplotlib import pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.animation as animation
        color = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w','violet','goldenrod','lightgreen','black']
        print(np.shape(out))
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        def animate(i):
            d = out[i]
            ax.cla()
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            #ax.view_init(40, 10)
            for j in range(len(self.category)-1):
                #x, y, z = [], [], []
                x = d[self.category[j]:self.category[j+1],0]
                y = d[self.category[j]:self.category[j+1],1]
                ax.scatter(x, y, s=5, alpha=1.,c=color[j])
            """
            if not i==0:
                ax.scatter(out[i-1][:, 0], out[i-1][:, 1], out[i-1][:, 2], s=5, alpha=0.2,c="blue")
            if i>1:
                ax.scatter(out[i - 2][:, 0], out[i - 2][:, 1], out[i - 2][:, 2], s=5, alpha=0.1, c="blue")
            """
            ax.set_title("Mapping " + 'i=' + str(i))

        ani = animation.FuncAnimation(fig, animate, frames=len(out), interval=100)

        ani.save("output.gif", writer="imagemagick")
        plt.show()


    def simpleanime(self,out):
        figure_or_data = {'data': [{'x': out[0][:,0], 'y': out[0][:,1], 'z': out[0][:,2], 'type': 'scatter3d'}],
                          'layout': {'scene': {'xaxis': {'range': [-1, 1], 'autorange': False},
                                     'yaxis': {'range': [-1, 1], 'autorange': False},
                                     'zaxis': {'range': [-1, 1], 'autorange': False}},
                                     'updatemenus': [{'type': 'buttons',
                                                      'buttons': [{'label': 'Play',
                                                                   'method': 'animate',
                                                                   'args': [None]}]}]},
                          'frames': [{'data': [{'x': out[i][:,0], 'y': out[i][:,1], 'z': out[i][:,2],'type': 'scatter3d'} for i in range(len(out))]}]}

        offline.plot(figure_or_data, filename='anime.html')

    """
    def plotly(self):
        trace_list=[]
        if np.shape(self.data)[1] == 3:
            for i in range(len(self.category) - 1):
                trace_list.append(go.Scatter3d(
                    x=self.data[self.category[i]:self.category[i+1], 0],
                    y=self.data[self.category[i]:self.category[i+1], 1],
                    z=self.data[self.category[i]:self.category[i+1], 2],
                    mode='markers',
                    marker={"size":3},
                    name=self.label[i]

                ))

        else:
            for i in range(len(self.category) - 1):
                trace_list.append(go.Scatter(
                    x=self.data[self.category[i]:self.category[i+1], 0],
                    y=self.data[self.category[i]:self.category[i+1], 1],
                    mode='markers',
                    marker={"sizemode":'diameter',"opacity":0.9,"size":6}
                    ))
    """