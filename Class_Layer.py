import numpy as np
import math
class Layer:
    x=np.random.random((3,4))#x是3维向量,共有4个样本
    da_in=np.random.random((4,4))

    ##############################
    #参数初始化
    ##############################
    def __init__(self,x_scale,nodes_num,Fun_Type):
        self.x_scale=x_scale
        self.w=np.random.random((self.x_scale,nodes_num))*0.01
        self.b=np.zeros((nodes_num,1))
        self.Fun_Type=Fun_Type
        
    ##############################
    #z=w^T x + b
    ##############################        
    def __Z(self):
        i=0
        B=np.copy(self.b)
        for i in range(self.m-1):
            B=np.hstack([B,self.b])
        ans=np.dot(self.w.T,self.x)+B  #.dot代表矩阵内积,即矩阵乘法,*代表对应元素相乘
        dz_dwt=self.x                  #向量对矩阵求导
        dz_db=1                        #向量对向量求导
        dz_dx=self.w
        return ans,dz_dwt,dz_db,dz_dx
    
    ##############################
    #激活函数
    ##############################
    def __my_sigmoid(self,z):   #sigmoid函数,几乎只能用在二元分类的输出层
        ans=1.0/(1+np.exp(-z))
        da_dz=ans*(1-ans)
        return ans,da_dz
    def __my_tanh(self,z):      #tanh函数,几乎总比sigmoid更优秀
        e1=np.exp(z)
        e2=np.exp(-z)
        ans=(e1-e2)/(e1+e2)
        da_dz=1-ans*ans
        return ans,da_dz
    def __my_ReLU(self,z):      #ReLU函数，最常用，不知道用哪个时就用这个
        ans=np.maximum(0,z)
        da_dz=z.copy()
        for j in range(0,da_dz.shape[1]):
            for i in range(0,da_dz.shape[0]):
                if (da_dz[i,j]>=0):
                    da_dz[i,j]=1
                else:
                    da_dz[i,j]=0
        return ans,da_dz
    def __my_Leaky_ReLU(self,z):#Leaky_ReLU函数
        ans=np.maximum(0,0.01*z,z)
        da_dz=z.copy()
        for j in range(0,da_dz.shape[1]):
            for i in range(0,da_dz.shape[0]):
                if (da_dz[i,j]>=0):
                    da_dz[i,j]=1
                else:
                    da_dz[i,j]=0.01*z[i,j]
        return ans,da_dz
    def __my_Actfunction(self,z,Fun_Type):
        if (Fun_Type.lower()=='sigmoid'.lower()):
            [ans,da_dz]=self.__my_sigmoid(self.z)
        elif(Fun_Type.lower()=='tanh'.lower()):
            [ans,da_dz]=self.__my_tanh(self.z)
        elif(Fun_Type.lower()=='ReLU'.lower()):
            [ans,da_dz]=self.__my_ReLU(self.z)
        elif(Fun_Type.lower()=='Leaky ReLU'.lower()):
            [ans,da_dz]=self.__my_Leaky_ReLU(self.z)
        else:
            print('没有此种激活函数','\r\n')
            pass
        return ans,da_dz
      
    ##############################
    #根据导数的传递法则求da(x,w,b)/dw,da(x,w,b)/db和da(x,w,b)/dx
    ##############################
    def __my_da(self,dz,dz_dwt,dz_dx,da_in):
        da_dw=(1/self.m)*np.dot(dz_dwt,dz.T)
        da_db=(1/self.m)*np.sum(dz,axis=1,keepdims=1)
        da_dx=np.dot(dz_dx,dz)
        return da_dw,da_db,da_dx
    
    ##############################
    #正向传播
    ##############################
    def my_forward(self,x):
        self.x=x
        self.m=self.x.shape[1]
        [self.z,self.dz_dwt,self.dz_db,self.dz_dx]=self.__Z()
        [a,self.da_dz]=self.__my_Actfunction(self.z,self.Fun_Type)
##        print('z:','\r\n',self.z,'\r\n')
##        print('a:','\r\n',a,'\r\n')
##        print('da_dz:','\r\n',self.da_dz,'\r\n')
##        print('dz_dwt:','\r\n',self.dz_dwt,'\r\n')
##        print('dz_db:','\r\n',self.dz_db,'\r\n')
        return a
    
    ##############################
    #反向传播
    ##############################
    def my_backward(self,da_in):
        dz=da_in*self.da_dz
        [dw,db,da_out]=self.__my_da(dz,self.dz_dwt,self.dz_dx,da_in)
##        print('dw:','\r\n',dw,'\r\n')
##        print('db:','\r\n',db,'\r\n')
##        print('da_out:','\r\n',da_out,'\r\n')
        return dw,db,da_out
    
    ##############################
    #一次训练
    ##############################   
    def my_training(self,x,da_in):
        a=self.my_forward(x)
        [dw,db,da_out]=self.my_backward(da_in)
        print(db.shape)
        return a,dw,db,da_out











            
