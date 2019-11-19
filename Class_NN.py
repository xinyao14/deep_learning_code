import numpy as np
from Class_Layer import Layer

class NN():
    dw=[]
    db=[]
    dL_da=[]
    def __init__(self,Hidden_Layer_para):
        self.Hidden_Layer_para=Hidden_Layer_para
        self.H_L=self.__Layer_create(Hidden_Layer_para)
        #print(self.H_L)
        
    def __Layer_create(self,Hidden_Layer_para):
        H_L=[]        
        for i in range(len(Hidden_Layer_para)-1):
            H_L.append(Layer(Hidden_Layer_para[i],Hidden_Layer_para[i+1],'ReLU'))
        return H_L
       
    def input(self,x,y):
        self.x=x
        self.y=y
        Input_Layer=Layer(self.x.shape[0],self.Hidden_Layer_para[0],'ReLU')
        Output_Layer=Layer(self.Hidden_Layer_para[-1],self.y.shape[0],'sigmoid')
        self.H_L.insert(0,Input_Layer)
        self.H_L.append(Output_Layer)
##        for i in range(0,len(self.H_L)):
##            print(self.H_L[i].x_scale)
##        print(self.H_L[len(self.H_L)-1].b.shape[0])

    def __Lost_fun(self,a):
##        print(a)
##        print(np.log(a))
        ans=-(self.y*np.log(a)+(1-self.y)*np.log(1-a))
##        print(ans)
        dL_da=-(self.y/a)+(1-self.y)/(1-a)
        return ans,dL_da
        
    def __Cost_fun(self,L):
        return (np.sum(L))/(self.x.shape[1])

    def NN_forward(self):
        a=self.x
        for i in range(0,len(self.H_L)):
            a=self.H_L[i].my_forward(a)
        self.a=a
        #损失函数
        [self.L,self.dL_da]=self.__Lost_fun(a)
##        print('L:',self.L,'\r\n')
        #成本函数
        self.J=self.__Cost_fun(self.L)
        print('J:',self.J,'\r\n')

    def NN_backward(self):
        da=self.dL_da
        for i in range(len(self.H_L),0,-1):
            [dw,db,da]=self.H_L[i-1].my_backward(da)
            self.dw.insert(0,dw)
            self.db.insert(0,db)

    def NN_paraUpdate(self,rate):
        for i in range(0,len(self.H_L)):
            self.H_L[i].w-=rate*self.dw[i]
            self.H_L[i].b-=rate*self.db[i]
            
    def NN_training(self,rate,k):
        for i in range(k):
            self.NN_forward()
            self.NN_backward()
            self.NN_paraUpdate(rate)







        
