from buttonSource import button

print("アミノ酸に変換するファイルを選択してください")
nameSource = button()

name = nameSource.split('/')[-1]
name = name.split('.')[0]

words = ""
dna_dic = {"AAA":"K","AGA":"R","CAA":"Q","CGA":"R","GAA":"E","GGA":"G","TAA":"","TGA":"",
"AAC":"N","AGC":"S","CAC":"H","CGC":"R","GAC":"D","GGC":"G","TAC":"Y","TGC":"C",
"AAG":"K","AGG":"R","CAG":"Q","CGG":"R","GAG":"E","GGG":"G","TAG":"","TGG":	"W",
"AAT":"N","AGT":"S","CAT":"H","CGT":"R","GAT":"D","GGT":"G","TAT":"Y","TGT":"C",
"ACA":"T","ATA":"I","CCA":"P","CTA":"L","GCA":"A","GTA":"V","TCA":"S","TTA":"L",
"ACC":"T","ATC":"I","CCC":"P","CTC":"L","GCC":"A","GTC":"V","TCC":"S","TTC":"F",
"ACG":"T","ATG":"M","CCG":"P","CTG":"L","GCG":"A","GTG":"V","TCG":"S","TTG":"L",
"ACT":"T","ATT":"I","CCT":"P","CTT":"L","GCT":"A","GTT":"V","TCT":"S","TTT":"F"}


f = open(nameSource,"r",encoding = "shift-jis")

#遺伝子名を取り除く
next=""
for line in f.readlines():
    if line[0] == ">":
        words += line
        continue

    #"if line[0] == "":
       # continue


#三つずつlineから塩基を読み取ってタンパク質に変換する
    line.replace("\n","")
    line = next + line


    for i in range(len(line)//3):
        words += dna_dic[line[3*i:3*i+3]]

    next=line[-(len(line)%3):-1]

    if len(f.readlines())%3 == 0:
        words += "\n"

f.close()


f = open("./fasta/" + name + "_pro.fasta.txt", "w")
f.write(words)

f.close()






