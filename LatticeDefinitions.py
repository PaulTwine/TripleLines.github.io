import numpy as np
FCCCell = np.array([[0,0,0], [1,0,0],[0,1, 0],[1,1,0],[0.5,0.5,0], [0,0.5,0.5],[1,0.5, 0.5],[0.5,1,0.5],[0.5,0,0.5],[0,1,1], [1,0,1], [1,1,1],[0,0,1],[0.5,0.5,1]]) 
BCCCell = np.array([[0,0,0], [1,0,0],[0,1, 0],[1,1,0],[0.5,0.5,0.5],[0,1,1], [1,0,1], [1,1,1],[0,0,1]])
SCCCell = np.array([[0,0,0], [1,0,0],[0,1, 0],[1,1,0],[0,1,1], [1,0,1], [1,1,1],[0,0,1]])
HCPCell = np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0], [1/3,1/3,0.5],[0,0,1],[1,0,1],[0,1,1],[1,1,1]])   
HCPBasisVectors = np.array([[1,0,0],[1/2,np.sqrt(3)/2,0],[0,0,1]]) 
def GetCellNodes(strLatticeType):
    objdct = dict()
    objdct['1'] = FCCCell
    objdct['2'] = HCPCell
    objdct['3'] = BCCCell
    return objdct[strLatticeType]

FCCPrimitive = np.array([[1,0,1],[1,1,0],[0,1,1]])/2
BCCPrimitive = np.array([[1,0,0],[0,1,0],[0.5,0.5,0.5]])