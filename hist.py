import numpy as np
import csv
import sys
import plotly.offline as offline
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
import dna_toolkit
from dna_toolkit import clean_fasta
from dna_toolkit import meta_fasta
from dna_toolkit import seq2hist
from dna_toolkit import plot_plotly
import sys,os,os.path

print('データ名を入力してください')
print('Ex.)HLA.fastaの場合 : name = HLA')
print('name = ')
name =input()



if os.path.isdir(name)==False:
    os.mkdir(name)
clean_fasta('fasta/'+name+'.fasta.txt',name+'/'+name+'_dna.txt')
meta_fasta('fasta/'+name+'.fasta.txt',name+'/'+name+'_list.txt')
print('ヒストグラムを作成中。。。')
seq2hist(name+'/'+name+'_dna.txt',name+'/'+name+'.csv')


print("ヒストグラムファイルが作成されました。Enter keyを押してください。")
input()


