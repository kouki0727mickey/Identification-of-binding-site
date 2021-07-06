import numpy as np
import csv
import sys
import plotly.offline as offline
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
import dna_toolkit_ver3
from dna_toolkit_ver3 import clean_fasta
from dna_toolkit_ver3 import meta_fasta
from dna_toolkit_ver3 import pseq2hist
from dna_toolkit_ver3 import plot_plotly
import sys,os,os.path

print('データ名を入力してください')
print('Ex.)HLA.fastaの場合 : name = HLA')
print('name = ')
name =input()
print('mer = ')
mer = int(input())


if os.path.isdir(name)==False:
    os.mkdir(name)

if os.path.exists(name+'/'+name+'.csv'):
    print('error : 同じ名前のcsvファイルを作成しようとしています。')
    print('このまま上書きでよろしければ　1　を入力してください。')
    nyuryoku = input()
    if int(nyuryoku) == 1:
        pass
    else:
        print('error')
        sys.exit()

clean_fasta('fasta/'+name+'.fasta.txt',name+'/'+name+'_dna.txt')
meta_fasta('fasta/'+name+'.fasta.txt',name+'/'+name+'_list.txt')
print('ヒストグラムを作成中。。。')
pseq2hist(name+'/'+name+'_dna.txt',name+'/'+name+'.csv', window_size = mer)


print("ヒストグラムファイルが作成されました。Enter keyを押してください。")
input()


