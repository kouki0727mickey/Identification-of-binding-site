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

words = ""

print("ファイル2を選択してください。")


nameSource = button()
name = nameSource.split('/')[-1]
name = name.split('.')[0]
f2 = open(nameSource,"r",encoding = "shift-jis")


for line in f2.readlines():
    if line[0] == ">":
        words += line
        continue

    #"if line[0] == "":
       # continue


    if line[0] == "\n":
        words = words + words2 + words1 +"\n\n"
        words2 = ""
        continue

#三つずつlineから塩基を読み取ってタンパク質に変換する

    line = line.replace("\n", "")
    words2 += line




f2.close()


f = open("./fasta/" + name + "_ligate.fasta.txt", "w")
f.write(words)

f.close()






