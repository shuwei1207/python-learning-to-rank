# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 14:01:57 2019

@author: USER
"""

import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import preprocessing

df = pd.read_excel('return.xlsx',index_col= "股票代號") #讀擋
index = df.index
total = df.iloc[:,1:]
col = [total.columns]

#Z-Score標準化
#建立StandardScaler物件
zscore = preprocessing.StandardScaler()
# 標準化處理
data1 = zscore.fit_transform(total)
data2 = zscore.fit_transform(total)

data1 = pd.DataFrame(data=data1,index= index)
data2 = pd.DataFrame(data=data2,index= index)
data1 = data1.sort_values(by=[1999], ascending=False)
data2 = data2.sort_values(by=[1998], ascending=False)

for i in range(len(data1)):
    if (0<=i<10):
        data1 = data1.replace( data1.iloc[i,-1] , 0)
    elif (10<=i<15):
        data1 = data1.replace( data1.iloc[i,-1] , 1)
    elif (15<=i<25):
        data1 = data1.replace( data1.iloc[i,-1] , 2)
    elif (25<=i<35):
        data1 = data1.replace( data1.iloc[i,-1] , 3)
    else:
        data1 = data1.replace( data1.iloc[i,-1] , 4)

for i in range(len(data2)):
    if (0<=i<10):
        data2 = data2.replace( data2.iloc[i,-2] , 0)
    elif (10<=i<15):
        data2 = data2.replace( data2.iloc[i,-2] , 1)
    elif (15<=i<25):
        data2 = data2.replace( data2.iloc[i,-2] , 2)
    elif (25<=i<35):
        data2 = data2.replace( data2.iloc[i,-2] , 3)
    else:
        data2 = data2.replace( data2.iloc[i,-2] , 4)
#X= data[:,0:-1]
#y = data[:,-1]

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
        
X_train = data2.iloc[:,0:-2]
y_train = data2.iloc[:,-2]
X_test = data1.iloc[:,1:-1]
y_test = data1.iloc[:,-1]
X_train = X_train.as_matrix()
X_test = X_test.as_matrix()

# use DMatrix for xgboost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# set xgboost params
param = {
    'max_depth': 5,  # the maximum depth of each tree
    'eta': 0.5,  # the training step for each iteration
    'silent': 1,  # logging mode - quiet
    'objective': 'multi:softprob',  # error evaluation for multiclass training
    'num_class': 5}  # the number of classes that exist in this datset
num_round = 100 # the number of training iterations

#------------- numpy array ------------------
# training and testing - numpy matrices
bst = xgb.train(param, dtrain, num_round)
preds = bst.predict(dtest)

# extracting most confident predictions
best_preds = np.asarray([np.argmax(line) for line in preds])
print ("Numpy array precision:", precision_score(y_test, best_preds, average='micro'))

def plot_confusion_matrix(confusion_matrix, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    fig, ax = plt.subplots(figsize=(20, 10))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()


cmdt = confusion_matrix(y_test, best_preds)
plot_confusion_matrix(cmdt,[0,1])
print(classification_report(y_test, best_preds))

#extract
df2 = pd.read_excel('return.xlsx') #原始檔案 才可抓profit

profit = []
pickrow = []

for i in range(len(best_preds)):
   if best_preds[i] ==0 : #test資料中 預測等於0
       #print(i)
       print('屬於0的個股:',df2.iloc[i,0])
       #print('機率:',preds[i,0])
       print('漲跌幅:',df2.iloc[i,-1])
       profit.append(df2.iloc[i,-1])
       pickrow.append(i)


print('平均漲跌幅:',sum(profit)/len(profit))