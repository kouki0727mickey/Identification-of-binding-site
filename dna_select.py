import pandas as pd
import csv
from buttonSource import button


nameSource = button()

df = pd.read_csv(nameSource,encoding = "shift-jis",header=0)

name = nameSource.split('/')[-1]
name = name.split('.')[0]

con = "y"

while con == "y":
    print("XX,YY,ZZ,RR、どの値でシーケンスを抽出しますか？")
    select = input()

    print("以上ですか？以下ですか？(1 or 2)")
    UD = input()

    print("値はいくつですか？")
    number = float(input())

    if UD == "1":
        df = df.where(df[select] >= number).dropna(how="any")

    if UD == "2":
        df = df.where(df[select] <= number).dropna(how="any")

    print("範囲の選択を続けますか？")
    con = input("続ける=y, 終了=Enter")


"""if select == "XX":
        if UD == "1":
                df = df.where(df["XX"] >= number).dropna(how="any")

        if UD == "2":
                df = df.where(df["XX"] <= number).dropna(how="any")

if select == "YY":
        if UD == "1":
                df = df.where(df["YY"] >= number).dropna(how="any")

        if UD == "2":
                df = df.where(df["YY"] <= number).dropna(how="any")

if select == "ZZ":
        if UD == "1":
                df = df.where(df["ZZ"] >= number).dropna(how="any")

        if UD == "2":
                df = df.where(df["ZZ"] <= number).dropna(how="any")"""

accession = list(df["accession"])
XX=df["XX"]
YY=df["YY"]
ZZ=df["ZZ"]
RR=df["RR"]

data = zip(accession, XX, YY, ZZ, RR)

file = open(name + "_select.csv","w")
w = csv.writer(file)
w.writerows(data)

print("シーケンス情報のファイルを選択してください。")

nameSource = button()

f = open(nameSource,"r",encoding = "shift-jis")

#遺伝子名を取り除く
words = ""
sequence=""
title = ""
for line in f.readlines():

    if line[0] == ">":
        line = line.replace("\n", "")
        title = line
        continue

    line = line.replace("\n", "")

    if line == "":
        if title in accession:
            words = words + title +"\n"+ sequence + "\n\n"

        else:
                sequence = ""


    sequence += line


f = open("./fasta/" + name + "select.fasta.txt", "w")
f.write(words)

f.close()


