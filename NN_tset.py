##from Class_Layer import Layer
##import numpy as np
################################################
###参数初始化
################################################
##x=np.random.random((3,6))#x是3维向量,共有6个样本
##da_in=np.random.random((4,6))
##nodes_num=4
##print('x:','\r\n',x,'\r\n')
##print('da_in:','\r\n',da_in,'\r\n')
################################################
###实例化类
################################################
##ly=Layer(x.shape[0],nodes_num)
##print('ly.w:','\r\n',ly.w,'\r\n')
##print('ly.b:','\r\n',ly.b,'\r\n')
################################################
###进行单层一次正反向
################################################
##ly.my_training(x,da_in)

#alt+3是快捷注释，alt+4是快捷去注释

from Class_NN import NN
import numpy as np
##############################################
#参数初始化
##############################################
x=np.random.random((3,6))#x是3维向量,共有6个样本
y=np.random.random((1,6))
print('x:','\r\n',x,'\r\n')
print('y:','\r\n',y,'\r\n')
rate=1
k=1000
Hidden_Layer_para=[4]
##############################################
#实例化类
##############################################
nn=NN(Hidden_Layer_para)
nn.input(x,y)
##nn.NN_paraUpdate(rate)
nn.NN_training(rate,k)
print(y)
print(nn.a)
print(nn.L)
#print(nn.H_L[3].Fun_Type)
##############################################
#进行单层一次正反向
##############################################

