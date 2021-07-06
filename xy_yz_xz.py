import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as mpl
from buttonSource import button

while (1):
    print("xy,yz,xzグラフ化するcsvファイルを選択してください")
    nameSource = button()
    name = nameSource.split("/")[-1]
    name = nameSource.rstrip(name)

    df = pd.read_csv( nameSource, encoding = "shift-jis",header=0)
    print(df)

    X=np.array(df["XX"])
    Y=np.array(df["YY"])
    Z=np.array(df["ZZ"])

    x0=np.mean(X)
    y0=np.mean(Y)
    z0=np.mean(Z)

    print("移動したい点を入力してください。")
    print("重心はx=",x0," y0=",y0," z0=",z0,"です。")

    print("(入力しない場合は重心点になります。)")

    x = input("x座標 =")
    if x == "":
        x1 = x0

    else:
        x1 = float(x)

    y = input("y座標 =")
    if y == "":
        y1 = y0

    else:
        y1 = float(y)

    z = input("z座標 =")
    if z == "":
        z1 = z0

    else:
        z1 = float(z)

    print("")


    X1 = X - x1
    Y1 = Y - y1
    Z1 = Z - z1
    R1 = np.sqrt(np.square(X1) + np.square(Y1) + np.square(Z1))

    fig = mpl.figure()

    #add_subplot()でグラフを描画する領域を追加する．引数は行，列，場所
    ax1 = fig.add_subplot(2, 3, 1)
    ax2 = fig.add_subplot(2, 3, 2)
    ax3 = fig.add_subplot(2, 3, 3)
    ax4 = fig.add_subplot(2, 3, 4)
    ax5 = fig.add_subplot(2, 3, 5)
    ax6 = fig.add_subplot(2, 3, 6)


    c1,c2,c3 = "blue","green","red"     # 各プロットの色  # 各ラベル

    ax1.scatter(X1, Y1, color=c1, s=1)
    ax2.scatter(Y1, Z1, color=c2, s=1)
    ax3.scatter(X1, Z1, color=c3, s=1)
    ax4.scatter(R1, X1, color=c1, s=1)
    ax5.scatter(R1, Y1, color=c2, s=1)
    ax6.scatter(R1, Z1, color=c3, s=1)

    ax1.set_xlabel("X")
    ax2.set_xlabel("Y")
    ax3.set_xlabel("X")
    ax4.set_xlabel("R")
    ax5.set_xlabel("R")
    ax6.set_xlabel("R")

    ax1.set_ylabel("Y")
    ax2.set_ylabel("Z")
    ax3.set_ylabel("Z")
    ax4.set_ylabel("X")
    ax5.set_ylabel("Y")
    ax6.set_ylabel("Z")



    ax1.set_title('XY')
    ax2.set_title('YZ')
    ax3.set_title('XZ')
    ax4.set_title('RX')
    ax5.set_title("RY")
    ax6.set_title("RZ")

    fig.tight_layout()
    fig.savefig("{}".format(name) + "2Dglaph.jpg")
    #レイアウトの設定
    mpl.show()


