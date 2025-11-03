'''
Code written by Nicholas J. Cooper.
Released under the MIT license, see the GitHub page for the full legal boilerplate.
tldr: you freely can do whatever you like with this code, but please cite the GitHub at: https://github.com/Thegolfingocto/KDwoCE.git
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
'''

import torch
import numpy as np
import json
import copy
import numbers

def GetInput(strPrompt: str) -> bool:
    chC = input(strPrompt)
    return chC == "Y"

def DictFlatten(d: dict, ret: dict = None) -> dict:
    if ret is None: ret = {}
    for k in d.keys():
        if isinstance(d[k], dict): ret = DictFlatten(d[k], ret)
        elif k not in ret.keys(): ret[k] = d[k]
    return ret

def DictFilterByKey(strFilter: str, d: dict, ret: dict = None, bRecursive: bool = False) -> dict:
    '''
    Returns a subdict consisting of KVPs where strFilter is a substring of K
    '''
    if ret is None: ret = {}
    vecKeys = list(d.keys())
    if bRecursive:
        for k in vecKeys:
            if strFilter in k:
                ret[k] = d[k]
                continue
            if isinstance(ret[k], dict):
                ret[k] = DictFilterByKey(strFilter, ret[k], ret, True)
    else:
        for k in vecKeys:
            if strFilter in k:
                ret[k] = d[k]

    return ret

def IsDictSubset(d1: dict, d2: dict) -> bool:
    '''
    Checks if d1 is a sub-dict of d2
    '''
    for key in d1.keys():
        if key not in d2.keys():
            return False
        if d1[key] != d2[key]:
            return False
        
    return True

def DictEquals(d1: dict, d2: dict) -> bool:
    return IsDictSubset(d1, d2) and IsDictSubset(d2, d1)

def PrintDictDiff(d1: dict, d2: dict) -> None:
    vecNewKeys = []
    vecDelKeys = []
    vecModKeys = []

    for key in d1.keys():
        if key not in d2.keys():
            vecDelKeys.append(key)
        elif d1[key] != d2[key]:
            vecModKeys.append(key)

    for key in d2.keys():
        if key not in d1.keys():
            vecNewKeys.append(key)

    print("Dict Diff:")
    print("-------------------------------------")
    print("New KVPs:")
    for nk in vecNewKeys: print("{}: {}".format(nk, d2[nk]))
    print("-------------------------------------")
    print("Missing KVPs:")
    for dk in vecDelKeys: print("{}: {}".format(dk, d1[dk]))
    print("-------------------------------------")
    print("Modified KVPs:")
    for mk in vecModKeys: print("{}: {} -> {}".format(mk, d1[mk], d2[mk]))
    print("-------------------------------------")

    return

def ReadFile(strPath: str) -> object:
    '''
    Generic multi-extension file opener
    '''
    if ".json" in strPath:
        with open(strPath, "r") as f:
            return json.load(f)
    elif ".pkl" in strPath:
        with open(strPath, "rb") as f:
            return torch.load(f)

    print("ReadFile() Found Unsupported Format: {}".format(strPath.split(".")[-1]))
    return None

def WriteFile(oData: object, strPath: str) -> None:
    '''
    Generic multi-extension file writer
    '''
    if ".json" in strPath:
        assert type(oData) == dict or type(oData) == list, "Invalid object of type {} passed for .json writing!".format(type(oData))
        with open(strPath, "w") as f:
            json.dump(oData, f, indent = 2) #pretty formatting
    elif ".pkl" in strPath:
        assert type(oData) == torch.tensor, "Invalid object of type {} passed for .pkl writing!".format(type(oData))
        with open(strPath, "wb") as f:
            torch.save(oData, f)
    else:
        print("WriteFile() Found Unsupported Format: {}".format(strPath.split(".")[-1]))
    return None

def GetRandomSubset(Y: torch.tensor, iN: int) -> torch.tensor:
    if len(Y.shape) > 1:
        iC = Y.shape[1]
        YA = torch.argmax(Y, dim = 1)
    else:
        iC = torch.max(Y) + 1
        YA = Y

    n = iN // iC
    vecIdx = [torch.where(YA == i)[0] for i in range(iC)]
    rng = np.random.default_rng()
    
    idx = torch.zeros((0), dtype=torch.int32)
    for i in range(iC):
        Idx = rng.permutation(vecIdx[i].shape[0])[:n]
        idx = torch.cat((idx, vecIdx[i][Idx]), dim = 0)
    
    return idx

def SplitLabelsByClass(Y: torch.Tensor) -> list[torch.Tensor]:
    '''
    Returns a list of index tensors based on the class labels in Y.
    Compatible with both one-hot encoded and integer label schemes.
    '''
    if len(Y.shape) == 2:
        c = Y.shape[1]
        Y = torch.argmax(Y, dim = 1).to("cpu")
    elif len(Y.shape) == 1:
        c = torch.max(Y) + 1
    
    vecIdx = [torch.where(Y == i)[0] for i in range(c)]

    return vecIdx

def SplitTrainFeaturesByClass(F: torch.tensor, Y: torch.tensor) -> list[torch.tensor]:
    vecIdx = SplitLabelsByClass(Y)

    return [F[idx,...] for idx in vecIdx]

def ListEquals(vecX: list, vecY: list) -> bool:
    if len(vecX) != len(vecY): return False
    vecT = copy.deepcopy(vecY)
    for i in range(len(vecX)):
        if vecX[i] not in vecT: return False
        vecT.remove(vecX[i])

    return True

def ListPermuataion(vecX: list, vecY: list) -> list[int]:
    '''
    Returns a permutation which describes how to map vecX bijectively to vecY
    '''
    #assert ListEquals(vecX, vecY), "ListPermutation requires inputs to be equal up to a bijection. X: {}, Y: {}".format(vecX, vecY)
    if not ListEquals(vecX, vecY): return False
    vecT = copy.deepcopy(vecX)
    vecPerm = []
    for i in range(len(vecY)):
        idxT = vecT.index(vecY[i])
        vecPerm.append(idxT)
        vecT[idxT] = None

    return vecPerm

def ListIntersection(vecX: list, vecY: list) -> list:
    '''
    Returns the intersection of two lists, with the sub-ordering from vecX.
    '''
    return [x for x in vecX if x in vecY]


def ListAvgDict(vecResults: list[dict]) -> dict:
    '''
    Assumes a list of flat dictionaries. Returns a dict containing avg/median/std for each numerical-valued key 
    '''
    vecTrackedKeys = []
    dStats = {}
    for strKey in vecResults[0].keys():
        if isinstance(vecResults[0][strKey], numbers.Number):
            vecTrackedKeys.append(strKey)
            dStats[strKey] = []

    for i in range(len(vecResults)):
        dR = vecResults[i]
        for strKey in vecTrackedKeys:
            dStats[strKey].append(dR[strKey])

    dRet = {}
    for strKey in vecTrackedKeys:
        nKey = np.array(dStats[strKey])
        dRet["Avg_" + strKey] = np.mean(nKey)
        dRet["Std_" + strKey] = np.std(nKey)
        dRet["Median_" + strKey] = np.median(nKey)
        dRet["Min_" + strKey] = np.min(nKey)
        dRet["Max_" + strKey] = np.max(nKey)

    return dRet

def DisplayDictFilter(dData: dict, strKeyFilter: str = "") -> None:
    for strKey in dData.keys():
        if strKeyFilter in strKey: print(strKey, ": ", dData[strKey])

    return



if __name__ == "__main__":
    X = [32, 32, 3]
    Y = [3, 32, 32]

    print(ListPermuataion(X, Y))

    quit()


    #Dict util tests
    dTest = {
    "Dataset": "CIFAR10",
    "Dataset:CIFAR100S:Params": {"Subset": 7},

    "Model": "ResNet9",
    "Model:Params": {
        "PreTrained": True,
        "PreTrained:Params": {"BackboneLRMultiplier": -1},
        "BatchNorm": True
    },
    "Model:ViT,TcT:Params": {"AvgTokens": True},

    "CustomReLU": False,
    "CustomReLU:Params": {"Kn": 0.236, "Kp": 1.0},

    "FeatureKD": True,
    "FeatureKD:Params": "...",

    "LayerMappingMethod": "One2One",
    "LayerMappingMethod:PreDefined:Params": {"LayerMap": "...", "LayerMapWeights": "..."},

    ":": ["vecIgnoredFields"]
    }

    print(DictFlatten(dTest))
    print(DictFilterByKey(":", dTest))