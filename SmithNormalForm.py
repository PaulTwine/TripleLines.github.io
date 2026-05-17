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
        self.__Identity = np.round(np.identity(max([self.__intColumns,self.__intRows])))
        self.__RowOperations = np.round(np.copy(self.__Identity))
        self.__ColumnOperations = np.round(np.copy(self.__Identity))
    def PackWithZeros(self):
        arrZeros = np.zeros([self.__MaxSize, self.__MaxSize])
        arrZeros[:self.__intRows,:self.__intColumns] = np.round(self.__TransformedMatrix)
        self.__TransformedMatrix = np.round(arrZeros)
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
        self.__ColumnOperations = np.round(np.matmul(self.__ColumnOperations,arrSwap))
    def SwapRows(self, i,j):
        arrSwap = self.SwapMatrix(i,j)
        self.__TransformedMatrix = np.round(np.matmul(arrSwap,self.__TransformedMatrix))
        self.__RowOperations =np.round(np.matmul(arrSwap,self.__RowOperations))    
    def InvertRow(self,i):
        arrInvert = np.copy(self.__Identity)
        arrInvert[i,i] = -1
        self.__TransformedMatrix = np.round(np.matmul(arrInvert,self.__TransformedMatrix))
        self.__RowOperations = np.round(np.matmul(arrInvert,self.__RowOperations))
    def InvertColumn(self,i):
        arrInvert = np.copy(self.__Identity)
        arrInvert[i,i] = -1
        self.__TransformedMatrix = np.round(np.matmul(self.__TransformedMatrix,arrInvert))
        self.__ColumnOperations = np.round(np.matmul(self.__ColumnOperations,arrInvert))    
    def ReduceByFirstRow(self,intStep):
        arrOriginalRow = np.round(np.copy(self.__TransformedMatrix[:,intStep]))
        arrRow = np.zeros(len(arrOriginalRow))
        if np.abs(arrOriginalRow[intStep]) > 0:
            for i in range(len(arrOriginalRow)):
                if i ==intStep:
                    arrRow[i] = 1
                else:
                    arrRow[i] = -np.trunc(np.round(arrOriginalRow[i]/arrOriginalRow[intStep],1))
            arrReduce = np.copy(self.__Identity)
            arrReduce[:,intStep] = arrRow
            self.__TransformedMatrix = np.round(np.matmul(np.round(arrReduce),self.__TransformedMatrix))
            self.__RowOperations = np.round(np.matmul(np.round(arrReduce),self.__RowOperations))
    def ReduceByFirstCol(self,intStep):
        arrOriginalCol = np.round(np.copy(self.__TransformedMatrix[intStep,:]))
        arrCol = np.zeros(len(arrOriginalCol))
        if np.abs(arrOriginalCol[intStep]) > 0:
            for i in range(len(arrOriginalCol)):
                if i == intStep:
                    arrCol[i]= 1
                else:
                    arrCol[i] = -np.trunc(np.round(arrOriginalCol[i]/arrOriginalCol[intStep],1))
            arrReduce = np.copy(self.__Identity)
            arrReduce[intStep,:] = arrCol
            self.__TransformedMatrix = np.round(np.matmul(self.__TransformedMatrix,np.round(arrReduce)))
            self.__ColumnOperations = np.round(np.matmul(self.__ColumnOperations,np.round(arrReduce)))
    def SwapMatrix(self,i,j):
        arrMatrix = np.copy(self.__Identity)
        if i !=j:
            arrMatrix[i] = self.__Identity[j]
            arrMatrix[j] = self.__Identity[i]
        return arrMatrix
    def GetRowOperations(self):
        return self.__RowOperations
    def GetColumnOperations(self):
        return self.__ColumnOperations
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
    def SubtractColumn(self, col_alter: int, col_subtract: int, n: int):
        self.__TransformedMatrix[:, col_alter] = self.__TransformedMatrix[:, col_alter] - n*self.__TransformedMatrix[:, col_subtract]
    def ScaleColumn(self, col_scale: int, n: int):
        self.__TransformedMatrix[:, col_scale] = n*self.__TransformedMatrix[:, col_scale] 
    def GetColumn(self, n):
        return self.__TransformedMatrix[:,n]
    def get_projection(self, i: int, j: int):
        return np.dot(self.GetColumn(i), self.GetColumn(j))/np.dot(self.GetColumn(j),self.GetColumn(j))
    def get_gram_schmidt(self):
        for j in range(self.GetNumberOfColumns()):
            for k in range(j):
                self.SubtractColumn(j, k, np.round(self.get_projection(j,k)))
        return self.GetTransformedMatrix()
class HermiteNormalForm(IntegerMatrix):
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
    def FindHermiteNormalForm(self, intMaxIter=100):
        self.ResetMatrices()
        self.FindLowerTriangular(intMaxIter)
        for j in range(1,self.GetNumberOfColumns()):
            self.ReduceByFirstCol(j)
        return self.GetTransformedMatrix()
    def FindLLLForm(self, d=0.5):
        self.ReduceCoefficentMagnitude()
        self.get_gram_schmidt()
        i = 1
        bln_stop = False
        while i < self.GetNumberOfColumns() and not(bln_stop):
            j = i-1
            while j >= 0:
                self.SubtractColumn(i,j,np.round(self.get_projection(i,j)))
                j -= 1 
            if (d-self.get_projection(i,i-1)**2)*np.linalg.norm(self.GetColumn(i-1))**2 <= np.linalg.norm(self.GetColumn(i))**2:                 
                i +=1
            else:
                self.SwapColumns(i,i-1)
                i = np.max([1,i-1]) 
        return self.GetTransformedMatrix()
    
    def ReduceCoefficentMagnitude(self, intMaxIter=100):
        self.FindHermiteNormalForm(intMaxIter)        
        arr_shape = np.shape(self.GetTransformedMatrix())
        arr_max_rows = np.argmax(np.abs(self.GetTransformedMatrix()), axis=1)[::-1]
        for i in arr_max_rows:
            bln_stop = False
            n=0
            while not(bln_stop) and n < 100:
                int_max_col = np.where(abs(self.GetTransformedMatrix()[i]) == np.max(abs(self.GetTransformedMatrix()[i])))[0][0]
                arr_max_arg = np.unravel_index(np.argmax(np.abs(self.GetTransformedMatrix())),np.shape(self.GetTransformedMatrix()))
                flt_max_signed = self.GetTransformedMatrix()[arr_max_arg]
                int_min_col = np.min(np.where(abs(self.GetTransformedMatrix()[i]) == np.min(abs(self.GetTransformedMatrix()[i])))[0])
                flt_min_value = self.GetTransformedMatrix()[i, int_min_col]
                if np.round(flt_min_value) != 0:
                    self.SubtractColumn(int_max_col, int_min_col, np.sign(flt_max_signed/flt_min_value))
                flt_new_max = np.max(np.abs(self.GetTransformedMatrix()))
                if flt_new_max > abs(flt_max_signed):
                    self.SubtractColumn(int_max_col, int_min_col, np.sign(flt_max_signed/flt_min_value))
                    bln_stop = True
                n +=1
        return self.GetTransformedMatrix()



class SmithNormalForm(HermiteNormalForm):
    def __init__(self,inMatrix: np.array):
        HermiteNormalForm.__init__(self,inMatrix)                              
    def FindSmithNormal(self,intMaxIter = 100):
        self.ResetMatrices()
        n = 0
        i = 0
        intRows = self.GetNumberOfRows()
        blnStop = False
        while n < intMaxIter and i < intRows-1 and not(blnStop):
            self.ReduceByFirstRow(i)
            self.ReduceByFirstCol(i)
            arrSwap = self.FindCurrentPivot(i)
            self.SwapRows(arrSwap[0],i)
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
  
class GenericCSL(SmithNormalForm):
    def __init__(self, inTransformation,inBasis):
        arrConjugate = np.matmul(np.linalg.inv(inBasis), np.matmul(inTransformation,inBasis))
        blnInt = False
        n = 0
        while not(blnInt) and n <50000:
            n +=1
            arrTest = n*arrConjugate
            if np.all(np.isclose(np.round(arrTest,0), np.round(arrTest,10), rtol = 1e-5, atol=1e-10)):
            #if np.all(np.around(arrTest,0) == np.around(arrTest,10)):
                blnInt=True
        self.__RationalDenominator = n
        intMatrix = np.round(n*arrConjugate)
        self.__IntegerTransition = intMatrix
        SmithNormalForm.__init__(self,intMatrix)
        self.__ConjugateTransition= arrConjugate
        self.__Basis = np.round(inBasis)
        self.__Transformation = inTransformation
    def GetConjugateTransitionMatrix(self):
        return self.__ConjugateTransition
    def GetCSLPrimitiveCell(self):
        if not(self.IsDiagonal()):
            self.FindSmithNormal()
        lstCSLLeftFactors = []
        lstCSLRightFactors = []
        for j in range(3):
            intDiagonal = int(self.GetTransformedMatrix()[j,j])
            lstCSLLeftFactors.append(intDiagonal/np.gcd(intDiagonal,int(self.__RationalDenominator)))
            lstCSLRightFactors.append(self.__RationalDenominator/np.gcd(intDiagonal,int(self.__RationalDenominator)))
        self.__LeftScaling = np.diag(lstCSLLeftFactors)
        self.__RightScaling = np.diag(lstCSLRightFactors)
        self.__Sigma = np.prod(np.array(lstCSLLeftFactors))
        arr_return =  np.matmul(self.__Transformation,np.matmul(self.__Basis, np.matmul(self.GetColumnOperations(),self.GetRightScaling())))
        return arr_return
    def GetLeftScaling(self):
        return self.__LeftScaling
    def GetRightScaling(self):
        return self.__RightScaling
    def GetSigma(self):
        return self.__Sigma
    def GetLeftCoordinates(self):
        return np.round(np.linalg.inv(self.GetRowOperations()))
    def GetRightCoordinates(self):
        return np.round(self.GetColumnOperations())
    def GetBasis(self):
        return self.__Basis

class GenericCSLandDSC(GenericCSL):
    def __int__(self, inTransformation,inBasis):
        GenericCSL.__init__(self,inTransformation, inBasis)
    def GetDSCPrimitiveCell(self):
        if not(self.IsDiagonal()):
            self.FindSmithNormal()
        arr_return =np.matmul(self.GetBasis(),self.GetColumnOperations(), np.linalg.inv(self.GetLeftScaling()))
        return arr_return