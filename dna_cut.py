from buttonSource import button

print("切り出したい遺伝子を選択してください")
nameSource = button()

name = nameSource.split('/')[-1]
name = name.split('.')[0]

words = ""

f = open(nameSource,"r",encoding = "shift-jis")

print("何個の塩基を何番目からを切り出しますか")

N = int(input("塩基数:"))
number = int(input("何番目:"))

#遺伝子名を取り除く
sequence = ""
for line in f.readlines():
    if line[0] == ">":
        words += line
        continue

    #"if line[0] == "":
       # continue


    if line[0] == "\n":
        if number > 0:
            words = words + sequence[number-1:number + N-1] + "\n\n"

        if number == -1:
            words = words + sequence[-N:] + "\n\n"

        if number < -1:
            words = words + sequence[number-N:number] + "\n\n"

        sequence = ""
        continue

#三つずつlineから塩基を読み取ってタンパク質に変換する

    line = line.replace("\n", "")
    sequence += line




f.close()


f = open("./fasta/" + name + "_cut{}-{}.fasta.txt".format(number,N), "w")
f.write(words)

f.close()






