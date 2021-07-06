#0行目にaccession numberが入った二つのcsvファイルを統合するプログラム。
#0行目にaccession numberが入っていることが条件
import numpy as np
import pandas as pd
import csv
import os
import sys
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox

def button():

    def button1_clicked():
        fTyp = [("","*")]
        iDir = os.path.abspath(os.path.dirname(__file__))
        filepath = filedialog.askopenfilename(filetypes = fTyp,initialdir = iDir)
        file1.set(filepath)

    # button2クリック時の処理
    def button2_clicked():
        messagebox.showinfo('FileReference Tool', u'参照ファイルは↓↓\n' + file1.get())
        #button4 = ttk.Button(frame2, text='決定', command=closeWindow)
        #button4.pack(side=LEFT)
        #global file_name
        #file_name = file1.get()
        #print(file1.get())
        #print(file_name)
        #print("!!!!!!!")
        #return r_file1


    def closeWindow():
        global file_name
        file_name = file1.get()
        #print(file1.get())
        #print(file_name)
        #print("!!!!!!!")
        root.destroy()

    #if __name__ == '__main__':
    # rootの作成
    root = Tk()
    root.title('FileReference Tool')
    root.resizable(False, False)

    # Frame1の作成
    frame1 = ttk.Frame(root, padding=10)
    frame1.grid()

    # 参照ボタンの作成
    button1 = ttk.Button(root, text=u'参照', command=button1_clicked)
    button1.grid(row=0, column=3)

    # ラベルの作成
    # 「ファイル」ラベルの作成
    s = StringVar()
    s.set('ファイル>>')
    label1 = ttk.Label(frame1, textvariable=s)
    label1.grid(row=0, column=0)

    # 参照ファイルパス表示ラベルの作成
    file1 = StringVar()
    file1_entry = ttk.Entry(frame1, textvariable=file1, width=50)
    file1_entry.grid(row=0, column=2)

    # Frame2の作成
    frame2 = ttk.Frame(root, padding=(0,5))
    frame2.grid(row=1)

    # Startボタンの作成
    button2 = ttk.Button(frame2, text='確認', command=button2_clicked)
    button2.pack(side=LEFT)

    # Cancelボタンの作成
    button3 = ttk.Button(frame2, text='決定', command=closeWindow)
    button3.pack(side=LEFT)

    #print(file1.get())

    root.mainloop()

    return file_name

    #print("hogehoge")
    #print(r_file1)
    #r_file1 = file_name
    #print(r_file1)
    #test = input()