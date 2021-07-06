from buttonSource import button
import random

print("アミノ酸に変換するファイルを選択してください")
nameSource = button()

name = nameSource.split('/')[-1]
name = name.split('.')[0]


print("遺伝子変異を起こす箇所を指定してください")
first = int(input("first:"))
last = int(input("last:"))

print("どのくらいの確率で変異を起こしますか？")
change = float(input("%:"))


words = ""
dna_dic = {"AAA":"K","AGA":"R","CAA":"Q","CGA":"R","GAA":"E","GGA":"G","TAA":"","TGA":"",
"AAC":"N","AGC":"S","CAC":"H","CGC":"R","GAC":"D","GGC":"G","TAC":"Y","TGC":"C",
"AAG":"K","AGG":"R","CAG":"Q","CGG":"R","GAG":"E","GGG":"G","TAG":"","TGG":	"W",
"AAT":"N","AGT":"S","CAT":"H","CGT":"R","GAT":"D","GGT":"G","TAT":"Y","TGT":"C",
"ACA":"T","ATA":"I","CCA":"P","CTA":"L","GCA":"A","GTA":"V","TCA":"S","TTA":"L",
"ACC":"T","ATC":"I","CCC":"P","CTC":"L","GCC":"A","GTC":"V","TCC":"S","TTC":"F",
"ACG":"T","ATG":"M","CCG":"P","CTG":"L","GCG":"A","GTG":"V","TCG":"S","TTG":"L",
"ACT":"T","ATT":"I","CCT":"P","CTT":"L","GCT":"A","GTT":"V","TCT":"S","TTT":"F"}

pro_change_dic = {"S":"A","I":"V","V":"I","T":"S","R":"K","L":"I","G":"S","F":"T","K":"R","A":"S"}



f = open(nameSource,"r",encoding = "shift-jis")

#遺伝子名を取り除く
next=""
sequence=""

for line in f.readlines():

    if line[0] == ">":

        words += line
        continue

    #"if line[0] == "":
       # continue


#塩基配列のみの文字列を作成する。
    line.replace("\n","")
    sequence += line

    if line[0] == "\n":
        print("1")


        for i in range(first-1, last):

            #if sequence[i] == "S" or "I" or "V" or "T" or "R"  or "L" or "G" or "F" or "A"
            if random.uniform(0,100) <=  change:
                n = sequence[i]
                while n == sequence[i]:
                    n = random.choice("ATGC")

                sequence = sequence[:i] + n + sequence[i+1:]

        words = words + sequence +"\n\n"




    #for i in range(len(line)//3):
     #   sequence += dna_dic[line[3*i:3*i+3]]

    #next=line[-(len(line)%3):-1]


f.close()


f = open("./fasta/" + name + "changedna.fasta.txt", "w")
f.write(words)

f.close()






