import numpy as np
class neuralnetwork:
    def __init__(self,Ninput,Nhidden,Noutput):
        self.Nin=Ninput
        self.Nhid=Nhidden
        self.Nout=Noutput
        self.InitializeWeight()
        
    def InitializeWeight(self):
        self.Whi=np.random.rand(self.Nhid,self.Nin)*0.1
        self.Woh=np.random.rand(self.Nout,self.Nhid)*0.1
        self.Bh=np.zeros((self.Nhid,1))
        self.Bo=np.zeros((self.Nout,1))
        self.Zhd=np.zeros((self.Nhid,1))
        self.Zod=np.zeros((self.Nout,1))
        self.Yhd=np.zeros((self.Nhid,1))
        self.mseLoging=np.array([])

    def sigmoid(self,valin):
        return (1/(1+np.exp(-(valin))))
    
    def derSigmoid(self,valin):
        return (valin*(1-valin))
    
    def Forward(self,inputData,bias=False):
            if bias==True:
                self.Zhd=np.dot(self.Whi, inputData)+self.Bh
                self.Yhd=self.sigmoid(self.Zhd)
                self.Zod=np.dot(self.Woh, self.Yhd)+self.Bo
                self.Yod=self.sigmoid(self.Zod)
            else:
                self.Zhd=np.dot(self.Whi, inputData)
                self.Yhd=self.sigmoid(self.Zhd)
                self.Zod=np.dot(self.Woh,self.Yhd)
                self.Yod=self.sigmoid(self.Zod)
            return self.Yod.T
    
    def Train(self,inDataSet,outDataSet,maxIt,lr,bias=False):
            for it in range(maxIt):
                Yn=self.Forward(inDataSet,bias)#proses forward neuron
                error=(outDataSet - Yn)#menghitung error
                mse=np.square(error).mean()#menghitung mean square error (MSE)
                self.mseLoging=np.append(self.mseLoging,mse)#logging mse
                deYo=error*self.derSigmoid(Yn)
                #Menghitung Turunan Bobot pada Hidden to Output
                dWoh=np.dot(self.Yhd, deYo)
                dBo=np.mean(deYo,axis=0,keepdims=True).T
                #Menghitung Turunan Bobot pada Input to Hidden
                dBh=(np.dot(deYo, self.Woh)*self.derSigmoid(self.Yhd).T)
                dWhi=np.dot(inDataSet, dBh)
                dBh=(np.mean(dBh,axis=0,keepdims=True)).T
                #Update Bobot
                self.Woh=self.Woh+lr*dWoh.T # .T adalah operator transpose
                self.Whi=self.Whi+lr*dWhi.T # .T adalah operator transpose
                #Update bias jika digunakan bias=True
                if bias==True:
                    self.Bo=self.Bo+lr*dBo
                    self.Bh=self.Bh+lr*dBh
                if it%10==0:
                    print("Iteration: {} MSE: {}".format(it,mse))
            return self.mseLoging, mse