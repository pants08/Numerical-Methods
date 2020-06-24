import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import csv
 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
 
#读取数据集并用‘;’分割数据
datatemp = []
with open('.\\student-mat.csv', 'r') as f:
    reader = csv.reader(f)
    result = list(reader)
    for i in range(1,396):
        str1 = str(result[i])
        datatemp.append(str1.split(';'))
#将数据集中的字符型转换为整型
data = np.array(datatemp)
labelencoder = LabelEncoder()
for i in range(30):
    data[:, i] = labelencoder.fit_transform(data[:, i])
#处理成绩
for i in range(np.shape(data)[0]):
    for j in range(30,32):
        if((data[i][j][1] == '1' or data[i][j][1] == '2') and data[i][j][2] >= '0' and data[i][j][2] <= '9'):
            data[i][j] = data[i][j][1] + data[i][j][2]
        else:
            data[i][j] = data[i][j][1]
    if((data[i][32][0] == '1' or data[i][32][0] == '2') and data[i][32][1] >= '0' and data[i][32][1] <= '9'):
        data[i][32] = data[i][32][0] + data[i][32][1]
    else:
        data[i][32] = data[i][32][0]
data = np.array(data,dtype=int)

#将数据分为训练集和数据集
# random_state 随机数种子  保证多次执行的时候 切分完的数据一致
x_train,x_test,y_train,y_test = train_test_split(data[:,0:32],data[:,32],train_size=0.7,test_size=0.3,random_state=50)
 
#模型构建
# 1.使用numpy的API将DataFrame转换为矩阵形式   将x_train  y_train 由DataFrame转为矩阵 然后进行矩阵计算
x = np.mat(x_train)
y = np.mat(y_train).reshape(-1,1)
 
# 2 求解析式
theta = (x.T * x).I * x.T * y
 
# 3.使用模型对数据做预测
y_predict = np.mat(x_test)*theta

#可视化
t = np.arange(len(x_test))
plt.figure(facecolor='w')
plt.plot(t,y_test,'r-',linewidth=1,label = 'true')
plt.plot(t,y_predict,'g-',linewidth=1,label = 'predict')
plt.legend(loc = 'lower right')  #设置label
plt.title("linear regression")
plt.show()

TP,FP,FN,TN = 0,0,0,0
for i in range(0,len(x_test)):
    if(y_test[i] < 10 and y_predict[i] < 10):
        TN += 1
    elif(y_test[i] < 10 and y_predict[i] >= 10):
        FP += 1
    elif(y_test[i] >= 10 and y_predict[i] < 10):
        FN += 1
    else:
        TP += 1

P = TP / (TP + FP)
R = TP / (TP + FN)
print("P=%.12E\nR=%.12E"%(P,R))
print("F1score=%.12E"%(2 * P * R / (P + R)))
