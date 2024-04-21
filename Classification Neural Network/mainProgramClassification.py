import numpy as np
from neuralnetwork import neuralnetwork
import matplotlib.pyplot as pylot

inDataSet=np.genfromtxt("phair.csv",delimiter=";",skip_header=1,usecols=(0,1,2))
outDataSet=np.genfromtxt("phair.csv",delimiter=";",skip_header=1,usecols=(3,4))
inDatatest=np.genfromtxt("phairtes.csv",delimiter=";",skip_header=1,usecols=(0,1,2))

maxVal=255
minVal=0
maxph=14
minph=0

inDataSetNorm=(inDataSet-minVal)/(maxVal-minVal)
outDataSetNorm=(outDataSet-minph)/(maxph-minph)
inDatatestNorm=(inDatatest-minVal)/(maxVal-minVal)

nn=neuralnetwork(3,25,2)
mseLoging,mse=nn.Train(inDataSetNorm.T,outDataSetNorm,100 , 0.1, bias=True)

outtest=nn.Forward(inDatatestNorm.T, True)
outtestnorm=(outtest*(maxph-minph))+minph
outClassification=np.round(outtestnorm)
print(outClassification)
pylot.plot(mseLoging)
pylot.show()