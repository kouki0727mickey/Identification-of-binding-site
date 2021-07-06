from buttonSource import button

print("塩基に変換するファイルを選択してください")
nameSource = button()

name = nameSource.split('/')[-1]
name = name.split('.')[0]

words = ""
protein_dic = {"\n":"\n","H" : "CAC","X":"TAA", "F" : "TTC", "L":"CTG", "I": "ATC", "M":"ATG", "V" : "GTG", "S": "AGC", "P": "CCC", "T":"ACC","A": "GCC","Y":"TAC", "Q":"CAG","N":"AAC","K":"AAG","D":"GAC","E":"GAG","C":"TGC","W":"TGG","R":"CGG","G":"GGC"}



f = open(nameSource,"r",encoding = "shift-jis")

print(f)

for line in f.readlines():
    if line[0] == ">":
        words += line
        continue

    if line[0] == "":
        continue

    print(words)


    for i in line:
        words += protein_dic[i]

f.close()


f = open("./fasta/" + name + "_ATGC.fasta.txt", "w")
f.write(words)

f.close()






