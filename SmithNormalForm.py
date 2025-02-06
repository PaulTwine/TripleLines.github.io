import numpy as np

class IntegerMatrix(object):
    def __init__(self,inMatrix: np.array):
        self.__OriginalMatrix = np.round(inMatrix)
        self.ResetMatrices()
        self.PackWithZeros()
    def ResetMatrices(self):
        self.__intRows = np.shape(self.__OriginalMatrix)[0]
        self.__intColumns = np.shape(self.__OriginalMatrix)[1]
        self.__TransformedMatrix = np.round(self.__OriginalMatrix)
        self.__MaxSize = np.max(np.shape(self.__OriginalMatrix))
        self.__Identity = np.identity(max([self.__intColumns,self.__intRows]))
        self.__LeftMatrix = np.copy(self.__Identity)
        self.__RightMatrix = np.copy(self.__Identity)
    def PackWithZeros(self):
        arrZeros = np.zeros([self.__MaxSize, self.__MaxSize])
        arrZeros[:self.__intRows,:self.__intColumns] = self.__TransformedMatrix
        self.__TransformedMatrix = arrZeros
    def GetOriginalMatrix(self):
        return self.__OriginalMatrix
    def GetTransformedMatrix(self):
        return self.__TransformedMatrix
    def FindCurrentPivot(self,i):
        arrCurrent = np.copy(self.__TransformedMatrix[i:,i:])
        fltMin = np.min(abs(arrCurrent[np.nonzero(arrCurrent)]))
        return np.argwhere(abs(arrCurrent) == fltMin)[0]+i
    def FindCurrentColumnPivot(self,i):
        arrCurrent = np.copy(self.__TransformedMatrix[i,i:])
        fltMin = np.min(abs(arrCurrent[np.nonzero(arrCurrent)]))
        return np.argwhere(abs(arrCurrent) == fltMin)[0]+i
    def FindCurrentRowPivot(self,i):
        arrCurrent = np.copy(self.__TransformedMatrix[i:,i])
        fltMin = np.min(abs(arrCurrent[np.nonzero(arrCurrent)]))
        return np.argwhere(abs(arrCurrent) == fltMin)[0]+i
    def FindPivot(self, in1DArray: np.array):
        fltMin = np.min(abs(in1DArray[np.nonzero(in1DArray)]))
        return np.argwhere(abs(in1DArray) == fltMin)[0]
    def SwapColumns(self, i,j):
        arrSwap = self.SwapMatrix(i,j)
        self.__TransformedMatrix = np.round(np.matmul(self.__TransformedMatrix,arrSwap))
        self.__RightMatrix = np.round(np.matmul(self.__RightMatrix,arrSwap))
    def SwapRows(self, i,j):
        arrSwap = self.SwapMatrix(i,j)
        self.__TransformedMatrix = np.round(np.matmul(arrSwap,self.__TransformedMatrix))
        self.__LeftMatrix =np.round(np.matmul(arrSwap,self.__LeftMatrix))    
    def InvertRow(self,i):
        arrInvert = np.copy(self.__Identity)
        arrInvert[i,i] = -1
        self.__TransformedMatrix = np.round(np.matmul(arrInvert,self.__TransformedMatrix))
        self.__LeftMatrix = np.round(np.matmul(arrInvert,self.__LeftMatrix))
    def InvertColumn(self,i):
        arrInvert = np.copy(self.__Identity)
        arrInvert[i,i] = -1
        self.__TransformedMatrix = np.round(np.matmul(self.__TransformedMatrix,arrInvert))
        self.__RightMatrix = np.round(np.matmul(self.__RightMatrix,arrInvert))    
    def ReduceByFirstRow(self,intStep):
        arrOriginalRow = np.copy(self.__TransformedMatrix[:,intStep])
        arrRow = np.zeros(len(arrOriginalRow))
        if arrOriginalRow[intStep] != 0:
            for i in range(len(arrOriginalRow)):
                if i ==intStep:
                    arrRow[i] = 1
                else:
                    arrRow[i] = -np.round(arrOriginalRow[i]/arrOriginalRow[intStep])
            arrReduce = np.copy(self.__Identity)
            arrReduce[:,intStep] = arrRow.astype('int')
            self.__TransformedMatrix = np.round(np.matmul(arrReduce,self.__TransformedMatrix)).astype('int')
            self.__LeftMatrix = np.round(np.matmul(arrReduce,self.__LeftMatrix)).astype('int')
    def ReduceByFirstCol(self,intStep):
        arrOriginalCol = np.copy(self.__TransformedMatrix[intStep,:])
        arrCol = np.zeros(len(arrOriginalCol))
        if arrOriginalCol[intStep] != 0:
            for i in range(len(arrOriginalCol)):
                if i == intStep:
                    arrCol[i]= 1
                else:
                    arrCol[i] = -np.round(arrOriginalCol[i]/arrOriginalCol[intStep])
            arrReduce = np.copy(self.__Identity)
            arrReduce[intStep,:] = arrCol.astype('int')
            self.__TransformedMatrix = np.round(np.matmul(self.__TransformedMatrix,arrReduce)).astype('int')
            self.__RightMatrix = np.round(np.matmul(self.__RightMatrix,arrReduce)).astype('int')
    def SwapMatrix(self,i,j):
        if i == j:
            arrMatrix = np.identity(self.__MaxSize).astype('int')
        else:
            arrMatrix = np.zeros([self.__MaxSize,self.__MaxSize]).astype('int')
            arrMatrix[i,j] = 1
            arrMatrix[j,i] = 1
            lstRange = list(range(self.__MaxSize))
            lstRange.remove(i)
            lstRange.remove(j)
            for n in lstRange:
                arrMatrix[n,n] =1
        return arrMatrix
    def GetLeftMatrix(self):
        return self.__LeftMatrix
    def GetRightMatrix(self):
        return self.__RightMatrix
    def IsDiagonal(self):
        blnReturn = False
        arrMatrix = np.copy(self.__TransformedMatrix)
        np.fill_diagonal(arrMatrix,0)
        if np.all(np.unique(arrMatrix)==0):
            blnReturn = True
        return blnReturn
    def GetNumberOfRows(self):
        return self.__intRows
    def GetNumberOfColumns(self):
        return self.__intColumns
    def CheckRowZerosToRight(self,i):
        blnReturn = False
        if i+1 < self.__intRows:
            arrZeros = self.__TransformedMatrix[i,i+1:]
            if np.all(np.unique(arrZeros) == 0):
                blnReturn = True
        return blnReturn
    def CheckColumnZerosBelow(self,i):
        blnReturn = False
        if i+1 < self.__intRows:
            arrZeros = self.__TransformedMatrix[i+1:,i]
        if np.all(np.unique(arrZeros) == 0):
            blnReturn = True
        return blnReturn
    def CheckZeros(self, i):
        blnReturn = False
        if i+1 < self.__intRows:
            arrZeros = np.append(self.__TransformedMatrix[i,i+1:],self.__TransformedMatrix[i+1:,i],axis=0)
        if np.all(np.unique(arrZeros) == 0):
            blnReturn = True
        return blnReturn

class SmithNormalForm(IntegerMatrix):
    def __init__(self,inMatrix: np.array):
        IntegerMatrix.__init__(self,inMatrix)
    def IsLowerTriangular(self):
        blnReturn = False
        arrLower = np.tril(self.GetTransformedMatrix())
        if np.all(arrLower == self.GetTransformedMatrix()):
            blnReturn = True
        return blnReturn
    def CheckIfZeroed(self, in1DArray):
        blnReturn = False
        intNumberOfZeros = len(in1DArray[in1DArray > 0]) 
        if intNumberOfZeros <= 0 or intNumberOfZeros == 1:
            blnReturn = True
        return blnReturn
    def egcd(self,a, b):
        if a == 0:
            return (b, 0, 1)
        else:
            g, x, y = self.egcd(b % a, a)
            return (g, y - (b // a) * x, x)
    def FindLowerTriangular(self,intMaxIter = 100):
        self.ResetMatrices()
        n = 0
        i = 0
        arrSwap = self.FindCurrentColumnPivot(0) ##initially place the column with least
        ##absolute value at the start
        intRows = self.GetNumberOfColumns()
      #  self.SwapRows(arrSwap[0],0)
        self.SwapColumns(arrSwap[0],0)
        blnStop = False
        while n < intMaxIter and i < intRows-1 and not(blnStop):
            #self.ReduceByFirstRow(i)
            #arrSwap = self.FindCurrentPivot(i)
            #self.SwapRows(arrSwap[0],i)
            self.ReduceByFirstCol(i)
            arrSwap = self.FindCurrentColumnPivot(i)
            self.SwapColumns(arrSwap[0],i)
            if self.IsLowerTriangular(): #Check whether diagonal form is achieved
                blnStop = True
            elif self.CheckRowZerosToRight(i): # are the ith row and column all zero except at the diagonal  
                i = i+1 #increment to look at the next submatrix
            n +=1 
        return self.GetTransformedMatrix()                              
    def FindSmithNormal(self,intMaxIter = 100):
        self.ResetMatrices()
        n = 0
        i = 0
        arrSwap = self.FindCurrentPivot(0) ##initially place the smallest absolute value in top left
        intRows = self.GetNumberOfRows()
        self.SwapRows(arrSwap[0],0)
        self.SwapColumns(arrSwap[1],0)
        blnStop = False
        while n < intMaxIter and i < intRows-1 and not(blnStop):
            self.ReduceByFirstRow(i)
            arrSwap = self.FindCurrentPivot(i)
            self.SwapRows(arrSwap[0],i)
            self.ReduceByFirstCol(i)
            arrSwap = self.FindCurrentPivot(i)
            self.SwapColumns(arrSwap[1],i)
            if self.IsDiagonal(): #Check whether diagonal form is achieved
                blnStop = True
            elif self.CheckZeros(i): # are the ith row and column all zero except at the diagonal  
                i = i+1 #increment to look at the next submatrix
            n +=1 
        arrDiagonal = np.copy(np.diag(self.GetTransformedMatrix()))
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
        return self.GetTransformedMatrix()
  

class GenericCSLandDSC(SmithNormalForm):
    def __init__(self, inTransition,inBasis):
        arrConjugate = np.matmul(np.linalg.inv(inBasis), np.matmul(inTransition,inBasis))
        blnInt = False
        n = 0
        while not(blnInt) and n <50000:
            n +=1
            arrTest = n*arrConjugate
            if np.all(np.around(arrTest,0) == np.around(arrTest,10)):
                blnInt=True
        self.__RationalDenominator = n
        self.__IntegerTransition = np.round(n*arrConjugate)
        SmithNormalForm.__init__(self,n*arrConjugate)
        (self.__IntegerTransition)
        self.__ConjugateTransition= arrConjugate
        self.__Basis = inBasis
    def GetConjugateTransitionMatrix(self):
        return self.__ConjugateTransition
    def GetCSLPrimtiveCell(self):
        if not(self.IsDiagonal()):
            self.FindSmithNormal()
        lstLeftFactors = []
        lstRightFactors = []
        for j in range(3):
            intDiagonal = int(self.GetTransformedMatrix()[j,j])
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

