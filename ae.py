import csv
import time
import random
import datetime
import copy
import sys
import os
import os.path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # モデル作成時のコメントを省略

#import tensorflow as tf
import numpy as np
from tqdm import tqdm
from glob import glob

#from model import Model
#from plot_2 import Plots
from model_cae_ver2 import Model
from model_cae_ver2 import Plot

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() #disable the vihavioe of tensorflow version 2

if __name__ == "__main__":

    source_folder = 'hist'
    #batch_size = [96]
    bs = 96   #batch_size
    #input_nodes_array = [ [16384, 4096, 1024, 256, 64, 3] ]
    input_nodes_ = [16384, 9261, 4096, 1024, 256, 64, 3]
    #print(input_nodes_array)
    #input('check')
    #training_iter = 1000  # 0#150#00 #学習回数
    coef = 0.0003



    # 0:手動 1:自動(ファイルごと), 2:自動(リスト指定)
    auto_label = 0
    manual_category_num = [0,17,156,406,7050,10385,12608,15318,21550,25349,26819,27877,29211,30691,32633,32644,]
    sort_column_num = [4]  # 2の時、上のファイルの何列目でソートするか





    # 出力の選択(0:なし, 1:出力する)
    first_epoch_result = 0  # 1回目の学習後の出力
    last_epoch_result = 1  # 最後学習後の出力
    autoscale_anime = 0  # 自動スケール
    autoscale_anime_rotation = 0  # 自動スケール回転あり
    samescale_anime = 0  # 固定スケール
    samescale_anime_rotation = 0  # 固定スケール回転あり


############################################################################
############## Nitada 編集
    """test = int(input('repeat number:'))
    if test == 0:
        batch_size = [16]
        training_iter = 100
        repeat = 1
    else:
        repeat = test"""
    repeat = 1
    test = 1

    print("学習回数を設定します(入力無しの場合は5000回)")
    training_iter = input('学習回数 = ')
    if (training_iter == ""):
        training_iter = 5000
    else:
        training_iter = int(training_iter)

    print(training_iter)

    print("ヒストグラム化した入力データのファイル名を入力してください。")
    print("Ex.)HLA.csvの場合 :  data_source = HLA)")
    data_source_id = input('data_source = ')
    #####################################################################

    file_id = data_source_id + '/ae_{0}'.format(training_iter)
    count = 0
    if os.path.isdir(data_source_id) == False:
        os.mkdir(data_source_id)

    while (1):
        if os.path.isdir(file_id) == False:
            os.mkdir(file_id)
            break
        else:
            count = count + 1
            file_id = data_source_id + '/ae_{0}_{1}'.format(training_iter, count)

    file_dataxyz = file_id + '/data_xyz_' + data_source_id + '.csv'  # 学習後のx,y,z座標を入れておくファイル

    name = data_source_id + '_' + str(training_iter)

    hist_data = data_source_id + '/' + data_source_id + ".csv"

    file_fasta_id = data_source_id + '/' + data_source_id + '_list.txt'

#############################################################################

    save = False
    i_num = int(training_iter / 100)


    #hist_data_list = glob(source_folder + '\\' + '*.csv')
    # list_file=folder_path+'/'+'gene_list.csv'
    # gene_name_list=np.loadtxt(list_file, delimiter=',', dtype = 'str')
    #print(hist_data_list)

#for hist_data in hist_data_list:
    # 入力データ(ヒストグラム,データ数,ファイル数）の読み込み
    def load_hist_num_label(see=False):
        i = 0
        # for hist_data in hist_data_list:
        if (1):
            # hist_data_name=hist_data.replace(source_folder+'\\', "").replace('.csv','')
            if i == 0:
                label[0] = "test"
                input_all = np.loadtxt(hist_data, delimiter=',', dtype='float')
                n = np.shape(input_all)[0]
                num.append(n)
                # if 1: print('n:' + str(n) +', ' + hist_data_name)
            else:
                label.append(hist_data_name)
                input_a = np.loadtxt(hist_data, delimiter=',', dtype='float')
                input_all = np.append(input_all, input_a, axis=0)
                n = np.shape(input_all)[0]
                num.append(n)
                # if 1: print('n:' + str(n) +', ' + hist_data_name)
            i += 1
        input_data = input_all / np.amax(input_all)
        if auto_label == 0:
            if manual_category_num[-1] == input_data.shape[0]:
                pass
            else:
                manual_category_num.append(input_data.shape[0])
        if see:
            print(num)
            print(label)
            print(input_data)
        return input_data


    # 学習係数の表示設定
    def e_notation(num=coef, cut=False, see=False):
        a = num
        e = 0
        while a < 1:
            a *= 10
            e += 1
        a = round(a, 5) if cut else a
        b = "{0:.5f}e-{1}".format(a, e)
        if see: print(b)
        return [b, a, e]


    # 結果名の設定(入力データ名＋学習係数＋学習回数＋バッチサイズ＋試行回数）
    def mk_name():
        if not os.path.exists('rslt_' + source_folder):
            os.mkdir('rslt_' + source_folder)
        index = 1
        name = '{0}_{1}_{2}_bs{3}_d{4}_s{5}'.format(data_name, e_notation()[0], training_iter, bs, input_nodes_[-1], int(index))
        while os.path.exists('rslt_' + source_folder + '/' + name) == 1:
            index += 1
            name = '{0}_{1}_{2}_bs{3}_d{4}_s{5}'.format(data_name, e_notation()[0], training_iter, bs, input_nodes_[-1], int(index))
        return name, 'rslt_' + source_folder + '/' + name


    def mk_test_name():
        if not os.path.exists('TEST_rslt_' + source_folder):
            os.mkdir('TEST_rslt_' + source_folder)
        index = 1
        name = 'TEST_{0}_{1}_{2}_bs{3}_d{4}_s{5}'.format(data_name, e_notation()[0], training_iter, bs, input_nodes_[-1], int(index))
        while os.path.exists('TEST_rslt_' + source_folder + '/' + name) == 1:
            index += 1
            name = 'TEST_{0}_{1}_{2}_bs{3}_d{4}_s{5}'.format(data_name, e_notation()[0], training_iter, bs, input_nodes_[-1], int(index))
        return name, 'TEST_rslt_' + source_folder + '/' + name


    # 入力データリスト作成
    """def mk_hist_list(see=False):  # 使用しない
        hist_data_list = glob(source_folder + '\\' + '*.csv')
        if os.path.exists(source_folder + '\\' + 'sort.csv') == 1:
            hist_data_list.remove(source_folder + '\\' + 'sort.csv')
        if see: print(hist_data_list)
        return hist_data_list"""


    # 学習パラメータの記録
    def write_param():
        """with open(r_name[1] + '/parameter.txt', 'w') as f:
            f.write('Learning Rate: %s\n' % str(coef))
            f.write('Number of Loop(Training): %s\n' % str(training_iter))
            f.write('Details of Layer:%s\n' % str(input_nodes))
            f.write('Batch Size:%s\n' % str(bs))"""
        with open(file_id + '/' + 'parameter_memo.txt', 'w') as f:
            f.write('name : %s\n' % str(data_source_id))
            f.write('Learning Rate: %s\n' % str(coef))
            f.write('Number of Loop(Training): %s\n' % str(training_iter))
            f.write('Details of Layer:%s\n' % str(input_nodes))
            f.write('batch size : %s\n' % str(bs))
            f.write('色分け 区切り : %s\n' % str(num))


    # 学習
    def auto_encoder(see=False):
        n = num[-1]
        loss_min = 1
        loss_val = -1
        loss_temp = "\rloss at {0:5d}th epoch = {1:10s}"  # {2}'
        for epoch in range(training_iter):  # ,desc="Loss={}".format(loss_val):
            r = [random.randint(0, n - 1) for i in range(bs)]
            train = []
            for cnt in range(bs):
                train.append(input_data[np.array(r[cnt]), :].tolist())
            train = np.array(train).astype(np.float32)
            train_dict = {mdl.x_train_in_list[0]: train}
            __, loss_val = mdl.sess.run([mdl.optimize_ops, mdl.loss_ops], feed_dict=train_dict)

            if (epoch + 1) % i_num == 0 or epoch == 0:
                dict_test = {mdl.x_train_in_list[0]: input_data.astype(np.float32)}
                output = mdl.sess.run(mdl.encode_op, feed_dict=dict_test)
                t_output.append(output)
                # print('Loss at {0}th epoch = {1}'.format(epoch, loss_val))
            if (epoch + 1) % 50 == 0 or epoch == 0:
                if loss_val < loss_min:
                    loss_min = loss_val
                    print(loss_temp.format(epoch + 1, e_notation(num=loss_val, cut=True)[0]))
                else:
                    print(loss_temp.format(epoch + 1, e_notation(num=loss_val, cut=True)[0]), end="")
        dict_test = {mdl.x_train_in_list[0]: input_data.astype(np.float32)}
        loss_test, output = mdl.sess.run([mdl.loss_ops, mdl.encode_op], feed_dict=dict_test)
        print("\nTest Loss: " + str(loss_test))
        if save:
            save_path = mdl.saver.save(mdl.sess, r_name[1] + "/model.ckpt")
            print("Model saved in path: %s" % save_path)
            np.save(r_name[1] + "/transition.npy", t_output)
        if see: print(t_output)


    # 結果の保存
    def last_output_save():
        """
        file=open(name+'/'+name+'.csv','w')
        writer=csv.writer(file,lineterminator='\n')
        writer.writerows(output)
        file.close()
        """
        last_output = np.array(t_output, dtype=float)[np.array(t_output, dtype=float).shape[0] - 1, :, :]
        #fileA = open(r_name[1] + '/' + r_name[0] + '.csv', 'w')
        #fastaファイルからaccession numberの情報を抜き出す。
        file_fasta_id = data_source_id + '/' + data_source_id + '_list.txt'
        print("fasta loading")
        file_fasta = open(file_fasta_id, 'r')
        #string = file_fasta.readline()  # 一つ目のみスキップ
        string = file_fasta.readline()
        accession = []
        while string:
            accession.append(string.rstrip('\n'))
            string = file_fasta.readline()
            # string = file_fasta.readline()
        file_fasta.close()

        #accession, x, y, z, 距離 の情報をcsvファイルとして出力
        fileA = open(file_dataxyz, 'w')
        writerA = csv.writer(fileA, lineterminator='\n')
        data_output = []
        data_output.append(["accession", "XX", "YY", "ZZ", "RR"])
        for i in range(0, len(last_output)):
            x = last_output[i][0]
            y = last_output[i][1]
            z = last_output[i][2]
            kyori = np.sqrt(x * x + y * y + z * z)
            data_output.append([accession[i], str(x), str(y), str(z), str(kyori)])
        writerA.writerows(data_output)
        fileA.close()
        """fileB = open(r_name[1] + '.csv', 'w')
        writerB = csv.writer(fileB, lineterminator='\n')
        writerB.writerows(last_output)
        fileB.close()"""
        return last_output


    num = [0,]
    label = [""]
    data_name = hist_data.replace('.csv', '').replace(source_folder + '\\', '')
    if auto_label == 2:
        sort_file = (hist_data[:-8] + 'label.csv').replace('hist', 'label').replace('_initial_18nt', '').replace('_last_18nt', '').replace('_middle_24nt', '').replace('_ORF3_NS5', '').replace('_S',
                                                                                                       '').replace(
            '_whole', '')
        if os.path.exists(sort_file):
            sort = np.loadtxt(sort_file, delimiter=',', dtype='str')
            sort = np.reshape(sort[1:, :], (sort.shape[0] - 1, sort.shape[1]))
        else:
            print(sort_file + "ファイルが存在しません。")
            exit(1)
            #continue
    # sort_file=folder_path+'/'+source_folder+'/sort.csv'    #色分け用のラベル情報ファイル

    print("Loading ............. [" + data_name + "]")
    input_data = load_hist_num_label()
    if auto_label == 2:
        if input_data.shape[0] != sort.shape[0]:
            print(input_data.shape)
            print(sort.shape)
            print('データ数がラベルリストと一致しません。')
            exit(1)
            #continue

#for input_nodes_ in input_nodes_array:
    #hist_data_list = mk_hist_list()  # see=True)

    if test==0:
        #input_nodes = input_nodes_#[input_data.shape[1],input_nodes_[-1]]
        input_nodes = [input_data.shape[1],input_nodes_[-1]]
        print(input_nodes)
    else:
        input_nodes = [n for n in input_nodes_ if int(n) < input_data.shape[1] + 1]
    print("Data Ready\n")

    for r in range(repeat):
    #for bs in batch_size:

        start_time = time.time()
        d = datetime.datetime.today()
        print("%s\n" % d)

        # enote=e_notation()#cut=True,see=True)
        if test==0:
            r_name = mk_test_name()
        else:
            r_name = mk_name()

        print(r_name[0] + ' start!\n')

        # print("Model Initializing")
        mdl = Model(coef, input_nodes)
        t_output = []
        # print("Initialization Completed!\n\n")
        auto_encoder()
        last_output = last_output_save()
        write_param()

        r_name = [r_name[0].replace("_s", "_f_s"), r_name[1].replace("_s", "_f_s")]
        #print(r_name)

        havent_sorted = [True]
        #print(havent_sorted[0])


        def use_sortfile(data, sort, s_num):
            #sort_column_num = [n + 2 for n in sort_column_num]
            s_num = s_num + input_nodes[-1] -1
            print(s_num)
            r_name[0] = r_name[0].replace("_f", "_l" + str(s_num - input_nodes[-1]+1))
            r_name[1] = r_name[1].replace("_f", "_l" + str(s_num - input_nodes[-1]+1))
            print(r_name[0])
            print(r_name[1])
            data_sort = np.append(data, sort, axis=1)
            sorted_data = data_sort[data_sort[:, s_num].argsort()]
            # X = X[ X[:,n].argsort() ]  #配列Xをn行でソートする

            if havent_sorted[0]:
                print("sort_num = " + str(s_num))
                label.clear()
                num.clear()
                label.append(sorted_data[0, s_num])
                num.append(0)
                j = 0
                for i in range(len(data)):
                    # print(str(i)+':'+data2[i,sort_column_num])
                    if (label[j] == sorted_data[i, s_num]):
                        pass
                    else:
                        label.append(sorted_data[i, s_num])
                        num.append(i)
                        j += 1
                num.append(i + 1)
                print(label)
                print(num)
                havent_sorted[0] = False

            return sorted_data[:, 0:3]


        def mk_plot(r_output):
            r_output1 = np.array(r_output, dtype=float)
            print(r_output1.shape)
            #p = Plot(np.array(r_output1, dtype=float), num, r_name[0], r_name[1], label, training_iter)  # r_name+"_"+str(s_num)
            p = Plot(r_output1, num, name, file_id, label, training_iter)
            if last_epoch_result: p.plotly()
            if first_epoch_result: p.plotly(frame_num=1)
            if autoscale_anime: p.matplot_anime_3d_with_label(rotation=(False, 4))
            if autoscale_anime_rotation: p.matplot_anime_3d_with_label(rotation=(True, 4))

            scaled_data = p.mk_scaled_data()
            lims = p.mk_lims()
            if samescale_anime: p.matplot_anime_3d_with_label_samescale(scaled_data, lims, rotation=(False, 4))
            if samescale_anime_rotation: p.matplot_anime_3d_with_label_samescale(scaled_data, lims,
                                                                                 rotation=(True, 4))


        if auto_label == 0:
            num = manual_category_num
            label = manual_category_num
            mk_plot(t_output)
        if auto_label == 1:
            mk_plot(t_output)
        if auto_label == 2:
            for s_num in sort_column_num:
                t_output2 = []
                t_output2.append(use_sortfile(t_output[-1], sort, s_num))
                mk_plot(t_output2)

        end_time = time.time()
        print('end-start time: ' + str(end_time - start_time))

    del input_data
    #####################################################################
