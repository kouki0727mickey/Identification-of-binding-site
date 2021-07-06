import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt
from buttonSource import button
from sklearn.ensemble import IsolationForest
from mpl_toolkits.mplot3d import Axes3D
#CVSファイルを読み込む
print("外れ値を見つけるcvsファイルを選択してください。")
nameSource = button()
name = nameSource.split("/")[-1]
name = nameSource.rstrip(name)

#dfに変換
df = pd.read_csv(nameSource, encoding="shift-jis", header=0)
print(df)

#グラフを作成
X = np.array(df["XX"])
Y = np.array(df["YY"])
Z = np.array(df["ZZ"])
print(X.shape)

XYZ = np.stack([X, Y, Z], 1)

#dataを学習
clf = IsolationForest()
clf.fit(XYZ)
pre = clf.predict(XYZ)

#autoで外れ値を設定
fig = plt.figure()
ax = Axes3D(fig)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

print(pre == -1,XYZ[pre == -1],XYZ[pre == -1,0])
ax.plot(XYZ[pre == -1, 0], XYZ[pre == -1, 1], XYZ[pre == -1,2], marker="o", linestyle="None")

plt.show()

# 外れ値スコアを算出する
outlier_score = clf.decision_function(XYZ)
# 外れ値スコアの閾値を設定する
while (1):
    print("外れ値スコアを設定してください")
    THRETHOLD = float(input())
    # 外れ値スコア以下のインデックスを取得する
    predicted_outlier_index = np.where(outlier_score < THRETHOLD)

    # 外れ値と判定したデータを緑色でプロットする
    predicted_outlier = XYZ[predicted_outlier_index]
    fig = plt.figure()
    ax = Axes3D(fig)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    print(predicted_outlier)

    ax.plot(
        predicted_outlier[:, 0],
        predicted_outlier[:, 1],
        predicted_outlier[:, 2],
        marker="o", linestyle = "None")

    plt.show()

