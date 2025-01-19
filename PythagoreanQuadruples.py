#%%
import numpy as np
import itertools as it
#%%
def FindPythagorasQuadruples(inInt):
    intTest = inInt**2
    i = 0 
    j = 0
    intSum = 0
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
                elif intSum < intTest:
                    k +=1
                else:
                    blnStop =True
            intSum = 0
            j += 1
        i+=1
    return np.unique(arrReturn,axis=0)
#%%
arrOut = FindPythagorasQuadruples(147)
#%%
arrPos = np.where(np.round(np.matmul(arrOut,np.transpose(arrOut)),10)==0)
#%%
arrPos
#%%
lstMatrices = []
for j in range(len(arrPos[0])):
    intSigma = 25
    lstMatrix = []
    arrVector1 = arrOut[arrPos[0][j]]/5
    arrVector2 = arrOut[arrPos[1][j]]/5
    arrVector3 = np.cross(arrVector1,arrVector2)
    lstMatrix.append(arrVector1)
    lstMatrix.append(arrVector2)
    lstMatrix.append(arrVector3)
    #print(np.vstack(lstMatrix))
    arrMatrix = np.vstack(lstMatrix)
    if np.any(np.abs(np.unique(arrMatrix)) != 1):
        lstMatrices.append(np.vstack(lstMatrix))

# %%
lstMatrices[0]
# %%
arrOut
# %%
arrOut
# %%
