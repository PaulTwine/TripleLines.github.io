import numpy as np

intDirs = 25
strDirectory = '/home/p17992pt/LAMMPSData/'
for j in range(intDirs):
    fIn = open(strDirectory +  str(j) + '/read0.dat', 'rt')
    fData = fIn.readline()
    fData = fData.startswith('[')
    print(fData)
    fIn.close()

