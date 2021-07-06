from buttonSource import button
import random

#全体をランダムに指定した数変異を入れて増幅させるプログラム

print("変異を挿入するアミノ酸配列(fastaファイル)を選択してください")
nameSource = button()

name = nameSource.split('/')[-1]
name = name.split('.')[0]

print("何か所に変異を挿入しますか")
N = int(input("変異数:"))

print("何本の遺伝子を作成しますか?")
number=int(input("本:"))

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


for i in range(number):
    print("1")
    A = []
    k = N
    change = sequence
    while k > 0:
        n = random.randint(first-1,last-1)
        if n in A:
            continue

        m = change[n]

        while m == change[n]:
            m = random.choice("ARNDCQEGHILKMFPSTWYV")
        print(m, change[n],n)
        change = change[:n] + m + change[n+1:]
        A.append(n)
        k -= 1

    words = words + title + "_" + str(i+1) + "\n" + change + "\n\n"

f.close()



    #for i in range(len(line)//3):
     #   sequence += dna_dic[line[3*i:3*i+3]]

    #next=line[-(len(line)%3):-1]



f = open("./fasta/" + name + "change{}_{}-{}_{}.fasta.txt".format(N,first,last,number), "w")
f.write(words)

f.close()











