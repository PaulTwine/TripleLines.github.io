import sys
import numpy as np

def UpdateTemplate(lstOriginal: list, lstNew: list, strOldFilename: str,strNewFilename: str):
    fIn = open(strOldFilename, 'rt')
    fData = fIn.read()
    for j in range(len(lstOriginal)):
        fData = fData.replace(lstOriginal[j], lstNew[j])
    fIn.close()
    fIn = open(strNewFilename, 'wt')
    fIn.write(fData)
    fIn.close()
    