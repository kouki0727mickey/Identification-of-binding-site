from buttonSource import button

print("塩基に変換するファイルを選択してください")
nameSource = button()

name = nameSource.split('/')[-1]
name = name.split('.')[0]

words = ""
protein_dic = {"\n":"\n","H" : "CAG","X":"ATC", "F" : "TTT", "L":"TTA", "I": "ATT", "M":"ATG", "V" : "GTT", "S": "TCT", "P": "CCT", "T":"ACT","A": "GCT","Y":"GAT", "Q":"GAA","N":"AAT","K":"AAA","D":"GAT","E":"GAA","C":"TGT","W":"TGA","R":"CGT","G":"GGT"}



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
f.write(words)f.close()






