import numpy as np
SCCList =  np.array([[0,0,0]])
BCCList =  np.append(SCCList, np.array([[0.5,0.5,0.5]]),axis=0)
FCCList =  np.append(SCCList, np.array([[0,0.5,0.5],[0.5,0,0.5],[0.5,0.5,0]]),axis=0) 
FCCCell = np.array([[0,0,0], [1,0,0],[0,1, 0],[1,1,0],[0.5,0.5,0], [0,0.5,0.5],[1,0.5, 0.5],[0.5,1,0.5],[0.5,0,0.5],[0,1,1], [1,0,1], [1,1,1],[0,0,1],[0.5,0.5,1]]) 
FCCCell2 = np.array(FCCCell - np.array([0.5,0.5,0.5]))
HCPList = np.array([[0,0,-1/2],[1,0,-1/2], [0,1,-1/2],[1/3,1/3,0],[0,0,1/2],[1,0,1/2], [0,1,1/2]])
   
HCPBasisVectors = np.array([[1,0,0],[1/2,np.sqrt(3)/2,0],[0,0,1]]) 