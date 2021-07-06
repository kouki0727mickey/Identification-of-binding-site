import numpy as np
import csv
import sys, time
import collections
import plotly.offline as offline
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from mpl_toolkits.mplot3d import Axes3D

def clean_fasta(r_name, w_name):
    """
    FASTA形式ファイルからDNAシークエンスを抽出する関数
    引数 
        r_name: FASTA形式のファイル
        w_name: 出力ファイルの名前

    """
    r_file = open(r_name, 'r')
    w_file = open(w_name, 'w')
    flg = True
    cnt = 0
    while 1:
        string = r_file.readline()
        if not string:
            break

        if(string[0]=='>'):
            if flg:
                flg = False
            else:
                w_file.write('\n')
        elif(string[0] =='\n'):
            pass
        else:
            w_file.write(string.rstrip('\n'))
        cnt += 1
        if cnt%500000 == 0:
            print(str(cnt) + '完了')
    r_file.close()
    w_file.close()


def meta_fasta(r_name, w_name):
    """
    FASTA形式ファイルからDNAシークエンスのメタデータを抽出する関数
    引数 
        r_name: FASTA形式のファイル
        w_name: 出力ファイルの名前
    
    """
    r_file = open(r_name, 'r')
    w_file = open(w_name, 'w')
    cnt = 1
    while 1:
        string = r_file.readline()
        if not string:
            break
        if(string[0]=='>'):
            #w_file.write(string.rstrip('\n'))
            w_file.write(string)
            cnt += 1
        #elif(string[0] =='\n'):
           # w_file.write('\n')
        else:
            pass
    r_file.close()
    w_file.close()


def seq2hist(r_name, w_name, normalize=True, window_size=5, base=dict({'A':0, 'T':1, 'C':2, 'G':3, 'N':0, 'R':0, 'M':0, 'Y':1, 'W':0, 'K':1, 'S':1,'B':1, 'D':0,'H':0,'V':1, 'L':0, 'I':0, 'F':0, 'Q':0, 'P':0, 'E':0, 'X':0, 'J':0, 'X':0, 'Z':0})):
    """
    シーケンスをヒストグラムに変換する関数
    引数
        r_name: 読み込むファイル名(clean_fastaしたもの)
        w_name: 出力ファイルの名前(.csv)
        normalize: シーケンスごとの正規化をするならばTrue, しないならFalse
        window_size: 一度に変換する塩基の個数
        base: 塩基とその対応する番号の辞書4

    """
    w_file = open(w_name, 'w')
    writer = csv.writer(w_file, lineterminator='\n')
    keys = [k for k, _ in base.items()]
    val = [v for _, v in base.items()]
    type = len(set(val))
    exception = ['\n']
    ncol  = np.power(type,window_size)
    with open(r_name, 'r') as r_file:
        cnt=0
        while 1:
            line = r_file.readline()
            if not line:
                break
            hist = [0]*ncol
            for i in range(len(exception)):
                line = line.replace(exception[i],"")
            """if np.array([line[i] not in keys for i in range(len(line))]).any():
                print(keys)
                print(line[i])
                print(line[i] not in keys)
                print("Error: An Exception '{0}' at {1}th Element".format(line[i],i))
                sys.exit()"""
            #print(line)
            numeric_seq = [base[l] for l in line]
            #print(line)
            for i in range(len(line)-window_size+1):
                at = sum([np.power(type,window_size-1-j)*int(numeric_seq[i+j]) for j in range(window_size)])
                hist[at] += 1/(len(line)-window_size+1) if normalize else 1
            writer.writerow(hist)

            cnt += 1
            if cnt%500 == 0:
                print(str(cnt) + '完了')
    w_file.close()

def pseq2hist(r_name, w_name, normalize=True, window_size=5, base=dict({'A':0, 'G':1, 'M':2, 'S':3, 'C':4, 'H':5, 'N':6, 'T':7, 'D':8, 'I':9, 'P':10,'V':11, 'E':12,'K':13,'Q':14,'W':15,'F':16,'L':17,'R':18,'Y':19})):
    """
    シーケンスをヒストグラムに変換する関数
    引数
        r_name: 読み込むファイル名(clean_fastaしたもの)
        w_name: 出力ファイルの名前(.csv)
        normalize: シーケンスごとの正規化をするならばTrue, しないならFalse
        window_size: 一度に変換する塩基の個数
        base: 塩基とその対応する番号の辞書
        baseに登録されていないAmino AcidはUnknownとして計算. したがって, ヒストグラムは21*21*21(=9261)次元.
    """
    unknown = 20
    w_file = open(w_name, 'w')
    writer = csv.writer(w_file, lineterminator='\n')
    num_seq = sum([1 for _ in open(r_name)])
    keys = [k for k, _ in base.items()]
    val = [v for _, v in base.items()]
    type = len(set(val))+1
    exception = ['\n']
    notinkeys = []
    ncol  = np.power(type,window_size)
    with open(r_name, 'r') as r_file:
        cnt = 1
        while 1:
            sys.stdout.write("\rProcessing %dth sequence" % cnt)
            sys.stdout.flush()
            line = r_file.readline()
            if not line:
                break
            hist = [0]*ncol
            for i in range(len(exception)):
                line = line.replace(exception[i],"")
            exception_index = [i for i in range(len(line)) if line[i] not in keys]
            if len(exception_index):
                notinkeys += [line[i] for i in exception_index]
            numeric_seq = [base[l] if l not in notinkeys else unknown for l in line]
            for i in range(len(line)-window_size+1):
                at = sum([np.power(type,window_size-1-j)*int(numeric_seq[i+j]) for j in range(window_size)])
                hist[at] += 1/(len(line)-window_size+1) if normalize else 1
            writer.writerow(hist)
            cnt += 1
    w_file.close()
    if len(numeric_seq):
        print("\nUnknown amino acids were detected during the process:\n{Unknown amino acids, #}\n")
        print(collections.Counter(notinkeys))

def plot_plotly(data,category,outname):
    """
    プロット関数(plotly)
    引数
        data: プロットするデータ(2 or 3次元)
        category: 種別インデックス番号のリスト（色指定用）
        outname: 出力ファイルの名前

    """
    trace_list=[]
    if np.shape(data)[1] == 3:
        for i in range(len(category) - 1):
            trace_list.append(go.Scatter3d(
                x=data[category[i]:category[i+1], 0],  # それぞれの次元をx, y, zにセットするだけです．
                y=data[category[i]:category[i+1], 1],
                z=data[category[i]:category[i+1], 2],
                mode='markers',
                marker=dict(
                    sizemode='diameter',
                    opacity=0.9,
                    size=3  # ごちゃごちゃしないように小さめに設定するのがオススメです．
                )))

    else:
        for i in range(len(category) - 1):
            trace_list.append(go.Scatter(
                x=data[category[i]:category[i+1], 0],  # それぞれの次元をx, y, zにセットするだけです．
                y=data[category[i]:category[i+1], 1],
                mode='markers',
                marker=dict(
                    sizemode='diameter',
                    opacity=0.9,
                    size=6  # ごちゃごちゃしないように小さめに設定するのがオススメです．
                )))

    data = trace_list
    layout = dict(hovermode='closest', height=700, width=600, title='plot')
    fig = dict(data=data, layout=layout)
    offline.plot(fig, filename=outname+'.html')

