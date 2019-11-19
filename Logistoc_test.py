from Class_Logistic import Logistic
import numpy as np
##############################################
#参数初始化
##############################################
m=4                      #m=4
x=np.random.random((3,4))#x是3维向量,共有4个样本
y=np.array([[1,0,1,0]])  #y是1维数据,共有4个样本
w=np.zeros((3,1))        #w是3x1的数组
b=0                      #b是1维的数据
rate=0.01                #学习率
k=1000                   #训练次数

##############################################
#实例化类
##############################################
lg=Logistic(m,x,y,w,b,rate,k)

##############################################
#进行连续训练
##############################################
lg.con_training()