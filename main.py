#!bin/python3
import requests
import PIL

from tkinter import *
from tkinter.ttk import *

#!pip install sklearn

####functions######
def getData():
    response=requests.get(url="")

    data=response.json()
   


#####ende functions######


#####TK-UI_setUp######

window=Tk()
window.title("ML-Py")

canvas1=Canvas(window,width=200,height=100)
canvas1.pack()

canvas2=Canvas(window,width=200,height=100)
canvas2.pack()

canvas3=Canvas(window,width=200,height=100)
canvas3.pack()
entry1=Entry(window)
entry2X=Entry(window)
entry2y=Entry(window)
canvas1.create_window(200,100,window=entry1)
canvas2.create_window(200,100,window=entry2X)
canvas3.create_window(200,100,window=entry2y)


 #####set UP#######




def pwd():
    global dir
    dir=entry1.get()
    
    
    global df
    df=pd.read_csv(dir)
    label1=Label(text=dir)

    label1.pack()
   
    print(f"'{dir}'")
    print(df.head())
    def printHead():
        df.head()
        label2=Label(text=df)
        label2.pack()
       
    
    printhead=Button(text="print head",command=printHead)
    printhead.pack()
    





#########################

button=Button(window,text="los",command=pwd)

#####TK-UI_setUp end#######



#####UI-style#####
button.pack(side="top")
window.geometry("400x400")

#####UI styles ende######



###ML Model#####
####Import######
import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.metrics import f1_score,accuracy_score,jaccard_score,log_loss
from sklearn.neighbors import KNeighborsRegressor,KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
#####ende import########

###Data Preprocessing####

# functions for preprocessing data

def DataVal():
        ###hier ist das Problem
        global X
        X=pd.DataFrame(df,columns=[f"'{entry2X.get()}'"])
        global y
        y=pd.DataFrame(df,columns=[f"'{entry2y.get()}'"])
      

        regr=LinearRegression()
        regr.fit(X,y)

        LR=Label(text=regr.predict(X))
        LR.pack()
        plt.scatter(X,y,alpha=0.4,c="r",linewidths=5)

        print(regr.predict(X))


def isnanVal():
    print(df.isna().sum())
    print(df.describe(include="all"))
    
    print(X)
    


# scaler=MinMaxScaler()

####Button preprocessing######
buttonval=Button(text="preporcess",command=DataVal)
buttonval.pack()
buttonisnan=Button(text="isnan Sum",command=isnanVal)
buttonisnan.pack()
######ende BTN#############
####ende data preprocessing######
#####ML Model Ende######



window.mainloop()