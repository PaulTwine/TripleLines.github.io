from typing import Dict, List

def solve(input: str) -> Dict[str, List[List[str]]]:
    dctWords = dict()
    strValue = input[:]
    lstValues = strValue.split('#')
    lstDict = lstValues[0]
    lstString = lstValues[1]
    lstDictWords = lstDict.split('\n')
    lstDictWordsSorted = list(map(lambda x : sorted(x.strip()),lstDictWords))
    lstWords = lstString.split('\n')
    for j in range(len(lstWords)):
        for k in range(len(lstWords)):
            if sorted(lstWords[j].strip()) == lstDictWordsSorted[k]:
                dctWords[lstWords[j]] = lstDictWords[k]
    return dctWords

print(solve("""IS\n THIS\n SPARTA\n # ATRAPS\n ATRAPS SI\n THIS IS SPARTA\n#"""))
