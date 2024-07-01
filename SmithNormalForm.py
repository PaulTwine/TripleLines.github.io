#%%
import numpy as np
import GeometryFunctions as gf
#%%

class SmithNormalForm(object):
    def __init__(self,inMatrix: np.array):
        self.__OriginalMatrix = np.round(inMatrix)
        self.__intRows = np.shape(inMatrix)[0]
        self.__intColumns = np.shape(inMatrix)[1]
        self.__DiagonalMatrix = np.round(inMatrix)
        self.__Identity = np.identity(max([self.__intColumns,self.__intRows]))
        self.__LeftMatrix = np.copy(self.__Identity)
        self.__RightMatrix = np.copy(self.__Identity)
    def GetOriginalMatrix(self):
        return self.__OriginalMatrix
    def SwapColumns(self, inMatrix,i,j):
        outMatrix = np.copy(inMatrix)
        outMatrix[:,[i,j]] = inMatrix[:,[j,i]]
        return outMatrix
    #def SwapRows(self,i,j):
        self.__DiagonalMatrix[[i,j],:] = self.__DiagonalMatrix[[j,i],:] 
    def SubtractRowsBelow(self,i):
        for j in range(i+1,self.__intRows):
                if self.__DiagonalMatrix[j,i] !=0:
                    x = int(self.__DiagonalMatrix[i,i]/self.__DiagonalMatrix[j,i])
                    self.__DiagonalMatrix[j] = self.__DiagonalMatrix[j]-x*self.__DiagonalMatrix[i]
    def SubtractRows(self,intStep, intPivot):
        for j in range(intStep,self.__intRows):
                if j != intPivot:
                    x = int(self.__DiagonalMatrix[j,intPivot]/self.__DiagonalMatrix[intStep,intPivot])
                    self.__DiagonalMatrix[j] = self.__DiagonalMatrix[j]-x*self.__DiagonalMatrix[intPivot]
    def FindPivot(self, in1DArray: np.array):
        fltMin = np.min(abs(in1DArray[np.nonzero(in1DArray)]))
        return np.argwhere(abs(in1DArray) == fltMin)[0]
    def CheckIfZeroed(self, in1DArray):
        blnReturn = False
        intNumberOfZeros = len(in1DArray[in1DArray > 0]) 
        if intNumberOfZeros == 0 or intNumberOfZeros == 1:
            blnReturn = True
        return blnReturn
    def egcd(self,a, b):
        if a == 0:
            return (b, 0, 1)
        else:
            g, x, y = self.egcd(b % a, a)
            return (g, y - (b // a) * x, x)
    def SwapMatrix(self,i,j):
        arrMatrix = np.identity(max([self.__intRows,self.__intColumns]))
        arrMatrix[:,[i,j]] = arrMatrix[:,[j,i]]
        return arrMatrix
    def SwapColumns(self, i,j):
        arrSwap = self.SwapMatrix(i,j)
        self.__DiagonalMatrix = np.round(np.matmul(self.__DiagonalMatrix,arrSwap))
        self.__RightMatrix = np.round(np.matmul(self.__RightMatrix,arrSwap))
    def SwapRows(self, i,j):
        arrSwap = self.SwapMatrix(i,j)
        self.__DiagonalMatrix = np.round(np.matmul(arrSwap,self.__DiagonalMatrix))
        self.__LeftMatrix =np.round(np.matmul(arrSwap,self.__LeftMatrix))    
    def InvertRow(self,i):
        arrInvert = np.copy(self.__Identity)
        arrInvert[i,i] = -1
        self.__DiagonalMatrix = np.round(np.matmul(arrInvert,self.__DiagonalMatrix))
        self.__LeftMatrix = np.round(np.matmul(arrInvert,self.__LeftMatrix))
    def InvertColumn(self,i):
        arrInvert = np.copy(self.__Identity)
        arrInvert[i,i] = -1
        self.__DiagonalMatrix = np.round(np.matmul(self.__DiagonalMatrix,arrInvert))
        self.__RightMatrix = np.round(np.matmul(self.__RightMatrix,arrInvert))    
    def ReduceByFirstRow(self,intStep):
        arrOriginalRow = np.copy(self.__DiagonalMatrix[:,intStep])
        arrRow = np.zeros(len(arrOriginalRow))
        if arrOriginalRow[intStep] != 0:
            for i in range(len(arrOriginalRow)):
                if i ==intStep:
                    arrRow[i] = 1
                else:
                    arrRow[i] = -np.round(arrOriginalRow[i]/arrOriginalRow[intStep])
        arrReduce = np.copy(self.__Identity)
        arrReduce[:,intStep] = arrRow
        self.__DiagonalMatrix = np.round(np.matmul(arrReduce,self.__DiagonalMatrix))
        self.__LeftMatrix = np.round(np.matmul(arrReduce,self.__LeftMatrix))
    def ReduceByFirstCol(self,intStep):
        arrOriginalCol = np.copy(self.__DiagonalMatrix[intStep,:])
        arrCol = np.zeros(len(arrOriginalCol))
        if arrOriginalCol[intStep] != 0:
            for i in range(len(arrOriginalCol)):
                if i == intStep:
                    arrCol[i]= 1
                else:
                    arrCol[i] = -np.round(arrOriginalCol[i]/arrOriginalCol[intStep])
        arrReduce = np.copy(self.__Identity)
        arrReduce[intStep,:] = arrCol
        self.__DiagonalMatrix = np.round(np.matmul(self.__DiagonalMatrix,arrReduce))
        self.__RightMatrix = np.round(np.matmul(self.__RightMatrix,arrReduce))
    def CheckZeros(self, i):
        blnReturn = False
        if i+1 < self.__intRows:
            arrZeros = np.append(self.__DiagonalMatrix[i,i+1:],self.__DiagonalMatrix[i+1:,i],axis=0)
        if np.all(np.unique(arrZeros) == 0):
            blnReturn = True
        return blnReturn                     
    def FindSmithNormal(self,intMaxIter = 100):
        n = 0
        i = 0
        arrSwap = self.FindPivot(self.__DiagonalMatrix) ##initially place the smallest absolute value in top left
        self.SwapRows(arrSwap[0],0)
        self.SwapColumns(arrSwap[1],0)
        blnStop = False
        while n < intMaxIter and i < self.__intRows-1 and not(blnStop):
            self.ReduceByFirstRow(i)
            arrSwap = self.FindPivot(self.__DiagonalMatrix[i:,i:]) + i
            self.SwapRows(arrSwap[0],i)
            self.ReduceByFirstCol(i)
            arrSwap = self.FindPivot(self.__DiagonalMatrix[i:,i:]) + i
            self.SwapColumns(arrSwap[1],i)
            if self.IsDiagonal(): #Check whether diagonal form is achieved
                blnStop = True
            elif self.CheckZeros(i): # are the ith row and column all zero except at the diagonal  
                i = i+1 #increment to look at the next submatrix
            n +=1 
        arrDiagonal = np.copy(np.diag(self.__DiagonalMatrix))
        k = 0
        while k < len(arrDiagonal):
            if arrDiagonal[k] < 0:
                self.InvertRow(k)
                arrDiagonal[k] = -arrDiagonal[k]
            k += 1
        arrSort = np.argsort(arrDiagonal)
        i = 0
        while i < len(arrSort): #put the diagonal entries in ascending
            if i != arrSort[i]:
                self.SwapRows(i,arrSort[i])
                self.SwapColumns(i,arrSort[i])
                arrSort[[i,arrSort[i]]] = arrSort[[arrSort[i],i]]
            i +=1 
        return self.__DiagonalMatrix
    def GetDiagonalMatrix(self):
        return self.__DiagonalMatrix
    def GetLeftMatrix(self):
        return self.__LeftMatrix
    def GetRightMatrix(self):
        return self.__RightMatrix
    def IsDiagonal(self):
        blnReturn = False
        arrMatrix = np.copy(self.__DiagonalMatrix)
        np.fill_diagonal(arrMatrix,0)
        if np.all(np.unique(arrMatrix)==0):
            blnReturn = True
        return blnReturn
#%%
class GenericCSLandDSC(SmithNormalForm):
    def __init__(self, inTransition,inBasis):
        arrConjugate = np.matmul(np.linalg.inv(inBasis), np.matmul(inTransition,inBasis))
        blnInt = False
        n = 0
        while not(blnInt) and n <50000:
            n +=1
            arrTest = n*arrConjugate
            if np.all(np.round(arrTest,0) == np.around  (arrTest,10)):
                blnInt=True
        self.__RationalDenominator = n
        self.__IntegerTransition = np.round(n*arrConjugate)
        SmithNormalForm.__init__(self,n*arrConjugate)
        (self.__IntegerTransition)
        self.__ConjugateTransition= arrConjugate
        self.__Basis = inBasis
    def GetConjugateSigma(self):
        return self.__ConjugateSigma
    def GetCSLPrimtiveCell(self):
        if not(self.IsDiagonal()):
            self.FindSmithNormal()
        lstLeftFactors = []
        lstRightFactors = []
        for j in range(3):
            intDiagonal = int(self.GetDiagonalMatrix()[j,j])
            lstLeftFactors.append(intDiagonal/np.gcd(intDiagonal,int(self.__RationalDenominator)))
            lstRightFactors.append(self.__RationalDenominator/np.gcd(intDiagonal,int(self.__RationalDenominator)))
        self.__LeftScaling = np.diag(lstLeftFactors)
        self.__RightScaling = np.diag(lstRightFactors)
        self.__Sigma = np.prod(np.array(lstLeftFactors))
    def GetLeftScaling(self):
        return self.__LeftScaling
    def GetRightScaling(self):
        return self.__RightScaling
    def GetSigma(self):
        return self.__Sigma
    def GetLeftCoordinates(self):
        return np.linalg.inv(self.GetLeftMatrix())
    def GetRightCoordinates(self):
        return np.linalg.inv(self.GetRightMatrix())
        
#%%
arrTest =     np.array([[4,3,0],[-3,4,0],[0,0,5]])
print(np.linalg.inv(arrTest))
#print(np.linalg.det(arrTest))
objSmith = SmithNormalForm(arrTest)
#print(objSmith.FindPivot(np.array([0,1,1,-1,2,3])))
print(objSmith.FindSmithNormal())



# gcd(a,b), p, q such that p*a+q*b = gcd(a,b)    
print(np.matmul(np.linalg.inv(objSmith.GetLeftMatrix()),np.matmul(objSmith.GetDiagonalMatrix(),np.linalg.inv(objSmith.GetRightMatrix()))))

#%%
import GeneralLattice as gl
import LatticeDefinitions as ld
#%%
objSigma = gl.SigmaCell(np.array([1,1,1]),ld.FCCCell)
objSigma.MakeCSLCell(49)
# %%
print(objSigma.GetCSLPrimitiveVectors()*2)

# %%
arrMatrix = objSigma.GetOriginalBasis()
print(arrMatrix)
print(str(arrMatrix[0,0].as_integer_ratio()))
print(arrMatrix)
# %%
objSmith1 = SmithNormalForm(arrMatrix)
objSmith1.FindSmithNormal().astype('int')
np.linalg.inv(objSmith1.GetLeftMatrix())
# %%
arrCheck = np.matmul(np.linalg.inv(2*np.transpose(ld.FCCPrimitive)),arrMatrix,np.transpose(2*ld.FCCPrimitive))
print(arrCheck)
objSmith3 = SmithNormalForm(arrCheck)
objSmith3.FindSmithNormal()
#np.linalg.det(2*ld.FCCPrimitive)
# %%
objCon = GenericCSLandDSC(arrMatrix, 2*np.transpose(ld.FCCPrimitive))
objCon.FindSmithNormal()
objCon.GetCSLPrimtiveCell()
print(objCon.GetRightScaling(),objCon.GetSigma())
np.linalg.det(objCon.GetDiagonalMatrix()/49)
# %%
blnInt = False
n = 0
while not(blnInt) and n <50000:
    n +=1
    arrTest = n*arrCheck
    if np.all(np.round(arrTest,0) == np.around(arrTest,10)):
        blnInt=True
        print(n)

print(n)
# %%
print(arrTest)
# %%
