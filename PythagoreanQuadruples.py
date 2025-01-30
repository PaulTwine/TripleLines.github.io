#%%
import numpy as np
#%%
class SigmaRotationMatrix():
    def __init__(self,inSigma: int):
        self.__intSigma = inSigma
    def FindSigmaMatrices(self):
        arrOut = self.FindPythagorasQuadruples(self.__intSigma)
        arrPos = np.where(np.round(np.matmul(arrOut,np.transpose(arrOut)),10)==0)
        lstMatrices = []
        for j in range(len(arrPos[0])):
            lstMatrix = []
            arrVector1 = arrOut[arrPos[0][j]]/self.__intSigma
            arrVector2 = arrOut[arrPos[1][j]]/self.__intSigma
            arrVector3 = np.cross(arrVector1,arrVector2)
            lstMatrix.append(arrVector1)
            lstMatrix.append(arrVector2)
            lstMatrix.append(arrVector3)
            arrMatrix = np.vstack(lstMatrix)
            if len(set(np.abs(np.unique(arrMatrix))).difference([0.0,1.0])) > 0:
                lstMatrices.append(arrMatrix)
        return lstMatrices
    def SetSigmaValue(self, inSigma):
        self.__intSigma = inSigma
    def GetSigmaValue(self):
        return self.__intSigma
    def FindPythagorasQuadruples(self,inInt):
        intTest = inInt**2
        i = 0 
        j = 0
        intSum = 0
        lstQuadruples = []
        while i <= inInt:
            j = i
            while i**2 +j**2 <= intTest:
                k =np.floor(np.sqrt(intTest-i**2-j**2))
                blnStop = False
                while not(blnStop):
                    intSum = i**2+j**2+k**2
                    if intSum == intTest:
                        arrValues = np.array([i,j,k,-i,-j,-k])
                        arrReturn = np.zeros([48,3])
                        for a in range(6):
                            arrReturn[8*a:8*(a+1),0] = arrValues[a]
                            lstPositions = list(range(6))
                            lstPositions.remove(a)
                            lstPositions.remove(np.mod(a+3,6))
                            arrFours = arrValues[lstPositions]
                            for b in range(8):
                                arrReturn[8*a+b,1] = arrFours[np.mod(b,4)]
                                arrTows = arrFours[[np.mod(b+1,4),np.mod(b+3,4)]]
                                arrReturn[8*a+b,2] = arrTows[np.mod(b+1,2)]
                        blnStop = True
                        lstQuadruples.append(np.unique(arrReturn,axis=0))
                    elif intSum < intTest:
                        k +=1
                    else:
                        blnStop =True
                intSum = 0
                j += 1
            i+=1
        return np.unique(np.vstack(lstQuadruples),axis=0)
#%%
objCSLMatrix = SigmaRotationMatrix(501)
lstMatrices = objCSLMatrix.FindSigmaMatrices()
print(lstMatrices[-200]*501)
#%%
#%%
arrSim = np.matmul(np.linalg.inv(lstMatrices[0]),np.array([lstMatrices[1:]]))
#%%
lstUniqueValues = list(map(lambda x: np.unique(np.abs(x)),lstMatrices))
print(lstUniqueValues)
# %%
