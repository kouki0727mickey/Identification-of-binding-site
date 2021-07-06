from buttonSource import button
import random

#全体をランダムに指定した数変異を入れて増幅させるプログラム

print("変異を挿入するアミノ酸配列(fastaファイル)を選択してください")
nameSource = button()

name = nameSource.split('/')[-1]
name = name.split('.')[0]

print("遺伝子変異を起こす箇所を指定してください")
first = int(input("first:"))
last = int(input("last:"))

f = open(nameSource,"r",encoding = "shift-jis")

#遺伝子名を取り除く
words = ""
title = ""
sequence = ""
for line in f.readlines():
    if line[0] == ">":
        line = line.replace("\n", "")
        title = line

        continue

    #"if line[0] == "":
       # continue


#塩基配列のみの文字列を作成する。

    line = line.replace("\n", "")
    sequence += line
    if line == "":
        continue

words = words + title + "_0\n" + sequence + "\n\n"


for i in range(first-1,last):
    change = sequence
    change = change[:i] + "R" + change[i+1:]

    words = words + title + "_" + str(i+1) + "\n" + change + "\n\n"

f.close()



    #for i in range(len(line)//3):
     #   sequence += dna_dic[line[3*i:3*i+3]]

    #next=line[-(len(line)%3):-1]



f = open("./fasta/" + name + "ArgScan{}-{}.fasta.txt".format(first,last), "w")
f.write(words)

f.close()











