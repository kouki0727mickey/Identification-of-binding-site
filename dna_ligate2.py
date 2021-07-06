from buttonSource import button

print("結合させるファイル1を選択してください。")
nameSource = button()
f1 = open(nameSource,"r",encoding = "shift-jis")


words1 = ""
words2 = ""
for line in f1.readlines():
    if line[0] == ">":
        continue

    line = line.replace("\n", "")
    words1 += line

f1.close()

print("ファイル2を選択してください。")

nameSource = button()
name = nameSource.split('/')[-1]
name = name.split('.')[0]
with open(nameSource,"r",encoding = "shift-jis") as f2, open("./fasta/" + name + "_ligate.fasta.txt", "w") as fw:
    i = 0
    for line in f2.readlines():
        if i == 0:
            fw.write(line)
            i += 1

        elif line[0] == ">":
            fw.write(words1 + words2 +"\n" + line)
            words2 = ""
    
        else:
            line = line.replace("\n","")
            words2 += line

    f2.close()
    fw.close()