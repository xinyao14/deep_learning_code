import numpy as np
class Logistic:
    m=4                      #m=4
    x=np.random.random((3,4))#x是3维向量,共有4个样本
    y=np.array([[1,0,1,0]])  #y是1维数据,共有4个样本
    w=np.zeros((3,1))        #w是3x1的数组
    b=0                      #b是1维的数据
    rate=0.01                #学习率
    k=0                      #训练次数

    ##############################
    #参数初始化
    ##############################
    def __init__(self,m,x,y,w,b,rate,k):
        self.m=m
        self.x=x
        self.y=y
        self.w=w
        self.b=b
        self.rate=rate
        self.k=k
        
    ##############################
    #z=w^T x + b
    ##############################        
    def Z(self):
        B=np.hstack([self.b,self.b,self.b,self.b])
        ans=np.dot(self.w.T,self.x)+B  #.dot代表矩阵内积,即矩阵乘法,*代表对应元素相乘
        dz_dwt=self.x
        dz_db=1
        return ans,dz_dwt,dz_db
    
    ##############################
    #sigmoid函数σ(z)=1/(1+e^(-z))
    ##############################
    def my_sigmoid(self,z): #sigmoid函数
        ans=1/(1+np.exp(-z))
        da_dz=ans*(1-ans)
        return ans,da_dz
    
    ##############################
    #Lost Function损失函数，单个样本的error
    ##############################
    #L(y^,y)=-[ylog(y^)+(1-y)log(1-y^)]
    ##############################
    def my_lost(self,a):
        ans=-(self.y*np.log(a)+(1-self.y)*np.log(1-a))
        dL_da=-(self.y/a)+(1-self.y)/(1-a)
        return ans,dL_da
    
    ##############################
    #Cost Function成本函数，全体样本的error
    ##############################
    #J(w,b)=(1/m)ΣL(y^[i],y[i])
    ##############################
    def my_cost(self,L):
        return (1/self.m)*np.sum(L)
    
    ##############################
    #反向传播
    ##############################
    #就是根据导数的传递法则求dJ(w,b)/dw和dJ(w,b)/db
    ##############################
    #损失函数:
    #dL/dw=[dL/dσ]*[dσ/dz]*[dz/dw]
    #dL/dw=[dL/dσ]*[dσ/dz]*[dz/db]
    ##############################
    #成本函数:
    #dJ(w,b)/dw=(1/m)Σ(dL/dw)
    #dJ(w,b)/dw=(1/m)Σ(dL/db)
    ##############################
    def my_dJ(self,dL_dz,dz_dwt):
        dJ_dw=(1/self.m)*np.dot(dz_dwt,dL_dz.T)
        dJ_db=(1/self.m)*np.sum(dL_dz)
        return dJ_dw,dJ_db
    
    ##############################
    #正向传播
    ##############################
    def my_forward(self):
        [z,dz_dwt,dz_db]=self.Z()
        [a,da_dz]=self.my_sigmoid(z)
        [L,dL_da]=self.my_lost(a)
        J=self.my_cost(L)
        print(J,'\r\n')
        return J,L,dL_da,da_dz,dz_dwt,dz_db
    
    ##############################
    #反向传播
    ##############################
    def my_backward(self,dL_da,da_dz,dz_dwt,dz_db):
        dL_dz=dL_da*da_dz
        [dJ_dw,dJ_db]=self.my_dJ(dL_dz,dz_dwt)
        #print(dJ_dw,'\r\n',dJ_db,'\r\n')
        return dJ_dw,dJ_db
    
    ##############################
    #参数更新
    ##############################
    ##############################
    #梯度下降法
    ##############################
    #α为学习率
    #w(new)=w-α[dJ(w,b)/dw]
    #b(new)=b-α[dJ(w,b)/db]
    ##############################
    def my_update(self,dJ_dw,dJ_db):
        #print(self.w,'\r\n',self.b,'\r\n')
        self.w-=self.rate*dJ_dw
        self.b-=self.rate*dJ_db

    ##############################
    #一次训练
    ##############################   
    def my_training(self):
        [J,L,dL_da,da_dz,dz_dwt,dz_db]=self.my_forward()
        [dJ_dw,dJ_db]=self.my_backward(dL_da,da_dz,dz_dwt,dz_db)
        self.my_update(dJ_dw,dJ_db)

    ##############################
    #连续训练
    ##############################
    def con_training(self):
        i=0
        for i in range(self.k):
            self.my_training()
##            print(self.b)








            
