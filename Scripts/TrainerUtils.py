'''
Code written by Nicholas J. Cooper.
Released under the MIT license, see the GitHub page for the full legal boilerplate.
tldr: you freely can do whatever you like with this code, but please cite the GitHub at: https://github.com/Thegolfingocto/KDwoCE.git
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
'''


import os
import copy
import json
import matplotlib.cm as cm
import matplotlib.colors as colors
from typing import Callable

from FeatureUtils import *
from Arch.Analysis.MathUtils import *
from Arch.Analysis.Analysis import *
from Trainer import KDTrainer
from KDUtils import *

from Arch.ITrainer import ITrainer
from Arch.Analysis.IAnalyzers import *
from Arch.Utils import *
from Arch.Logger import *


vecStatNames = ["LinearSeperability", "PCAIntrinsicDimension", "DistMtxIntrinsicDimensions", "TDA", "CovarianceStats", "SlicedMutualInformation", "DistanceStats", "Tensors/ClassPCAID", "NumOrthants", "HODT"]
vecReadableStatNames = ["Linear Separability", "PCA Intrinsic Dimension", "Other IDs", "Persistent Entropy", "DNC Stats", "Sliced Mutual Information", "Distance Stats", "Class-wise Embedding Dimension", "Distinct Orthants"]
vecClassStyles = ["dashed", "dashdot", "dotted"]
vecShadesOfBlue = ["midnightblue", "darkblue", "mediumblue", "royalblue", "dodgerblue", "deepskyblue", "lightskyblue"]


def AutoAddTrainers(T: KDTrainer, strKey: str, vecValues: list[str]) -> list[KDTrainer]:
    if T.GetValue(strKey) in vecValues: vecValues.remove(T.GetValue(strKey))
    vecT = [T]
    for strValue in vecValues:
        newCfg = copy.deepcopy(T.dCfg)
        newCfg[strKey] = strValue
        newCfg["MaxLR"] = LookupModelMaxLR(newCfg["Model"], newCfg["Dataset"], newCfg["ProjectorArchitecture"] if newCfg["FeatureKD"] else "")
        vecT.append(KDTrainer("../KDTrainer", newCfg))

    return vecT

def TComputePerLayerStats(T: KDTrainer, vecStats: list[int], vecClasses: list[int] = None, iStartLayer: int = 0, iMaxDimension: int = 262144,
                          fDataFrac: float = 1.0) -> None:
    #generate all features first
    if T.tModel is None: T.ILoadModel()
    vecL = [i for i in range(T.tModel.iGenLayers)]
    vecL = vecL[iStartLayer:]
    T.GenPerLayerFeatures(vecL)
    #if T.dlTrain is None: T.ILoadDataset()
    
    strDir = T.GetCurrentFolder() + "Features/"
    with open(strDir + T.GenFeatureStringLabel(bTeacher = False) + ".pkl", "rb") as f:
        Y = torch.load(f)

    YA = torch.argmax(Y, dim = 1)
    
    strDir = T.GetCurrentFolder() + "Analysis/"
    if not os.path.isdir(strDir): os.mkdir(strDir)

    strF = "" if fDataFrac == 1 else str(fDataFrac)
    
    vecPaths = [strDir + strN + strF + (".json" if "/" not in strN else "") for strN in vecStatNames]
    #print(vecPaths)
    vecResults = [None for _ in vecStatNames]
    
    if vecClasses is not None:
        vecClassPaths = [[strDir + strN + "_Class" + str(c) + ".json" for strN in vecStatNames] for c in vecClasses]
        vecClassResults = [[None for _ in range(len(vecStatNames))] for _ in vecClasses]
    
    for iS in vecStats:
        if os.path.exists(vecPaths[iS]):
            printf("Found existing {} data for config {}".format(vecStatNames[iS], T.IGenHash()))
            if not GetInput("Overwrite? (Y/X)"):
                printf("Quitting TComputePerLayerStats()")
                return
        vecResults[iS] = []
        
        if vecClasses is not None:
            if iS == 0: continue
            for c in range(len(vecClasses)): vecClassResults[c][iS] = []
    
    #it is intractable to compute ID/PD across all samples. In practice, 5-10k is enough to get an accurate estimate
    tIdx = GetRandomSubset(Y, 5000)
    tIdx2 = GetRandomSubset(Y, min([500, Y.shape[0] // Y.shape[1]]) * Y.shape[1])
    
    if vecClasses is not None or 6 in vecStats:
        vecIdx = [torch.where(torch.argmax(Y[tIdx,...], dim = 1) == c)[0] for c in range(Y.shape[1])]    

    #loop thru the features
    for iL in vecL:
        strF = T.GetCurrentFolder() + "Features/" + T.GenFeatureString(iL, bTeacher = False) + ".pkl"
        printf("Loading Features: {}".format(strF), INFO)
        with open(strF, "rb") as f:
            F = torch.load(f)
        
        #sample features and center
        if 7 not in vecStats and 8 not in vecStats:
            F = F[tIdx,...]
            bSampled = True
        else: bSampled = False
        #F -= torch.mean(F, dim = 0)

        dim = 1
        for i in range(1, len(F.shape)): dim *= F.shape[i]
        if dim > iMaxDimension:
            printf("Warning! Features are too large, downsampling...\nIf this is unacceptable, increase iMaxDimension ({})".format(iMaxDimension), 
                   WARNING)
            if len(F.shape) == 3:
                print("NOT IMPLEMENTED YET!")
            elif len(F.shape) == 4:
                iPoolSize = math.ceil(np.sqrt(dim / iMaxDimension))
                F = torch.nn.functional.max_pool2d(F, (iPoolSize, iPoolSize))

        #do stuff with these features
            
        if vecResults[0] is not None:
            if len(F.shape) == 4 and F.shape[2] > 16:
                with torch.no_grad():
                    tF = torch.zeros((F.shape[0], F.shape[1], 16, 16))
                    iBatchSize = 2000
                    iN = F.shape[0] // iBatchSize
                    if F.shape[0] % iBatchSize != 0: iN += 1
                    for i in range(iN):
                        iE = (i+1)*iBatchSize
                        if iE > F.shape[0]: iE = F.shape[0]
                        tF[i*iBatchSize:iE,...] = torch.nn.functional.adaptive_max_pool2d(F[i*iBatchSize:iE,...], (16, 16))
                acc, _ = IMeasureSeperability(tF, Y)
                del tF
            else:
                acc, _ = IMeasureSeperability(F, Y)
            print("Layer {} linear seperability: {}".format(iL, acc))
            vecResults[0].append(acc)
        
        if vecResults[1] is not None:
            dPCAID = IComputePCAID(F[tIdx,...] if not bSampled else F)
            print("Layer {} PCAID:".format(iL), dPCAID["PCAID"])
            vecResults[1].append(dPCAID)
            
            if vecClasses is not None:
                for c in range(len(vecClasses)):
                    dPCAIDC = IComputePCAID(F[vecIdx[c],...])
                    print("Layer {} Class {} PCAID:".format(iL, vecClasses[c]), dPCAIDC["PCAID"])
                    vecClassResults[c][1].append(dPCAIDC)
        
        if vecResults[2] is not None:
            dID = IComputeDistMtxIDs(F[tIdx,...] if not bSampled else F)
            print("Layer {} IDs:".format(iL), dID)
            vecResults[2].append(dID)
            
            if vecClasses is not None:
                for c in range(len(vecClasses)):
                    dIDC = IComputeDistMtxIDs(F[vecIdx[c],...])
                    print("Layer {} Class {} IDs:".format(iL, vecClasses[c]), dIDC)
                    vecClassResults[c][2].append(dIDC)
            
        if vecResults[3] is not None:
            PD = ComputePD(F[tIdx,...] if not bSampled else F)
            vecES, PE, nL = ComputePDES(PD)
            print("Layer {} PE: {}, Num Bars: {}".format(iL, PE, nL))
            vecResults[3].append({"EntropySummary": vecES, "PersistentEntropy": PE, "NumBars": nL})
            
            if vecClasses is not None:
                for c in range(len(vecClasses)):
                    PD = ComputePD(F[vecIdx[c],...])
                    vecESC, PEC, _ = ComputePDES(PD)
                    print("Layer {} Class {} PE:".format(iL, vecClasses[c]), PEC)
                    vecClassResults[c][3].append({"EntropySummary": vecESC, "PersistentEntropy": PEC})
            
        if vecResults[4] is not None:
            dCov = IComputeCovarianceStats(F[tIdx,...] if not bSampled else F, Y[tIdx,...])
            print("Layer {} CovStats: {}".format(iL, dCov))
            vecResults[4].append(dCov)

        if vecResults[5] is not None:
            fSMI = IComputeSlicedMI(F[tIdx,...] if not bSampled else F, YA[tIdx,...]) #5k should be plenty for a decent estimate according to: https://proceedings.mlr.press/v206/wongso23a/wongso23a.pdf
            print("Layer {} Sliced MI: {}".format(iL, fSMI))
            vecResults[5].append(fSMI)

        if vecResults[6] is not None:
            tF = F[tIdx,...] if not bSampled else F
            
            dDistMtxStats = IComputeDistMtxStats(tF, vecIdx)

            print("Layer {} Compression: ".format(iL), dDistMtxStats["ClassCompression"])
            print("Layer {} Separation: ".format(iL), dDistMtxStats["ClassSeparation"])

            vecResults[6].append(dDistMtxStats)

        if vecResults[7] is not None:
            tPCA = ComputeClassPCAID(F[tIdx2,...], Y[tIdx2,...])
            vecResults[7].append(tPCA)

        if vecResults[9] is not None:
            tF = F
            f = tF.view(tF.shape[0], -1).to(device)

            vecAvg = []
            vecStd = []

            fMin, fMax = torch.min(tF), torch.max(tF)
            fRange = fMax - fMin
            iSS = 512
            for i in range(iSS):
                fT = fMin + (i / (iSS - 1))*fRange
                tOrths = torch.sum(torch.where(f > fT, 1, 0), dim = 1).float()
                vecAvg.append(torch.mean(tOrths).to("cpu").item())
                vecStd.append(torch.std(tOrths).to("cpu").item())

            dOrths = {
                "Avgs": vecAvg,
                "Stds": vecStd,
                "AD": f.shape[1]
            }
            vecResults[9].append(dOrths)

        del F #python bad
    
    #save the results
    for iS in vecStats:
        strPath = vecPaths[iS]
        if ".json" in strPath:
            with open(strPath, "w") as f:
                json.dump(vecResults[iS], f)
        elif "/" in strPath:
            strSubDir = strDir + strPath.split("/")[-2] + "/"
            strN = strPath.split("/")[-1]
            #print(strSubDir, strN)
            if not os.path.exists(strSubDir): os.mkdir(strSubDir)
            for l in range(len(vecResults[iS])):
                #todo: consider adding type logic for different file formats
                strSave = strSubDir + strN + "_L" + str(l) + ".pkl"
                with open(strSave, "wb") as f:
                    torch.save(vecResults[iS][l], f)
            
        if vecClasses is not None:
            for c in range(len(vecClasses)):
                with open(vecClassPaths[c][iS], "w") as f:
                    json.dump(vecClassResults[c][iS], f)
    
    return





def TGetPerLayerStats(T: KDTrainer, vecStats: list[int], vecClasses: list[int] = None, iStartLayer: int = 0, fDataFrac: float = 1.0) -> any:
    strDir = T.GetCurrentFolder() + "Analysis/"
    if not os.path.isdir(strDir):
        printf("Error! No Analysis directory for config {}!".format(T.IGenHash()))
    
    strF = "" if fDataFrac == 1 else str(fDataFrac)

    #load in the analysis results
    vecPaths = [strDir + strN + strF + (".json" if "/" not in strN else "") for strN in vecStatNames]
    vecResults = [None for _ in vecStatNames]
    
    vecClassResults = None
    if vecClasses is not None:
        vecClassPaths = [[strDir + strN + "_Class" + str(c) + ".json" for strN in vecStatNames] for c in vecClasses]
        vecClassResults = [[None for _ in range(len(vecStatNames))] for _ in vecClasses]
    
    vecMissing = []
    vecMissingCls = None if vecClasses is None else []
    
    for iS in vecStats:
        strQ = vecStatNames[iS]
        if "/" in strQ:
            strQ = strDir + strQ.split("/")[-2]
            if not os.path.isdir(strQ): vecMissing.append(iS)
        elif not os.path.exists(vecPaths[iS]):
            #print(vecPaths[iS])
            vecMissing.append(iS)
        if vecClasses is not None:
            if iS in [0, 4, 5, 6, 7]: continue
            for c in range(len(vecClasses)):
                if not os.path.exists(vecClassPaths[c][iS]):
                    if vecClasses[c] not in vecMissingCls: vecMissingCls.append(vecClasses[c])
                    if iS not in vecMissing: vecMissing.append(iS)
    if len(vecMissing) > 0 or (vecMissingCls is not None and len(vecMissingCls) > 0):
        printf("Generating missing stats: {}".format(vecMissing))       
        TComputePerLayerStats(T, vecMissing, vecClasses = vecMissingCls, iStartLayer = iStartLayer, fDataFrac = fDataFrac) 
        
    if T.tModel is None: T.ILoadModel()
    L = T.tModel.iGenLayers

    for iS in vecStats:
        strPath = vecPaths[iS]
        if ".json" in strPath:
            with open(strPath, "r") as f:
                vecResults[iS] = json.load(f)
                if len(vecResults[iS]) > L - iStartLayer:
                    vecResults[iS] = vecResults[iS][iStartLayer:]
        elif "/" in strPath:
            strSubDir = strPath.split("/")[-2] + "/"
            strN = strPath.split("/")[-1]
            vecResults[iS] = []
            for l in range(iStartLayer, L):
                with open(strDir + strSubDir + strN + "_L" + str(l) + ".pkl", "rb") as f:
                    vecResults[iS].append(torch.load(f))
        
            
        if vecClasses is not None:
            if iS in [0, 4, 5, 6]: continue
            for c in range(len(vecClasses)):
                with open(vecClassPaths[c][iS], "r") as f:
                    vecClassResults[c][iS] = json.load(f)
                    if len(vecClassResults[c][iS]) > L - iStartLayer:
                        vecClassResults[c][iS] = vecClassResults[c][iS][iStartLayer:]

    return vecResults, vecClassResults

def TPlotPerLayerStats(T: KDTrainer, vecStats: list[int], vecClasses: list[int] = None, iStartLayer: int = 0,
                       bNormalize: bool = True, bNormalizeByHostDim: bool = False, bNormalizeLinSep: bool = False) -> None:
    if T.GetValue("Dataset") != "CIFAR100S":
        strT = T.GetValue("Model") + " Per-Layer Complexity of " + T.GetValue("Dataset") + " Representations"
    else:
        strT = T.GetValue("Dataset") + " Subset " + str(T.GetValue("Subset")) + " " +  T.GetValue("Model")
        
    vecResults, vecClassResults = TGetPerLayerStats(T, vecStats, vecClasses, iStartLayer = iStartLayer)
    
    if T.tModel is None: T.ILoadModel()
    L = [i for i in range(T.tModel.iGenLayers)]
    L = L[iStartLayer:]
    vecPlots = []
    
    #per case plotting
    if 0 in vecStats:
        vecPlots.append((vecResults[0], "blue", 3, "solid", "Linear Separability (Normalized)" if bNormalizeLinSep else "Linear Separability"))
        
    if 1 in vecStats:
        vecPlots.append(([d["PCAID"] / 5000 for d in vecResults[1]], "red", 3, "solid", "PCA Intrinsic Dimension"))
        #vecPlots.append(([d["HostDim"] for d in vecResults[1]], "black", 3, "solid", "Ambient Dimension"))
        #vecHD = vecPlots[-1][0]
        
        if vecClasses is not None:
            for c in range(len(vecClasses)):
                vecPlots.append(([d["PCAID"] for d in vecClassResults[c][1]], "red", 3, vecClassStyles[c % len(vecClassStyles)], "Class " + str(vecClasses[c]) + " PCA Intrinsic Dimension"))
        
    if 2 in vecStats:
        vecPlots.append(([d["TwoNNID"] for d in vecResults[2]], "orange", 3, "solid", "TwoNN Intrinsic Dimension"))
        vecPlots.append(([d["AvgVariance"] for d in vecResults[2]], "grey", 3, "solid", "Avg. Variance"))
        vecPlots.append(([d["AvgVariance"] / d["VarID"] for d in vecResults[2]], "midnightblue", 3, "solid", "Var. Variance"))
        if 1 not in vecStats:
            vecPlots.append(([d["HostDim"] for d in vecResults[2]], "black", 3, "solid", "Host Dimension"))
            
        if vecClasses is not None:
            for c in range(len(vecClasses)):
                vecPlots.append(([d["TwoNNID"] for d in vecClassResults[c][2]], "orange", 3, vecClassStyles[c % len(vecClassStyles)], "Class " + str(vecClasses[c]) + " TwoNN Intrinsic Dimension"))
        
        vecPlots.append(([d["VarID"] for d in vecResults[2]], "maroon", 3, "solid", "Avg. Variance Intrinsic Dimension"))   
        if vecClasses is not None:
            for c in range(len(vecClasses)):        
                vecPlots.append(([d["VarID"] for d in vecClassResults[c][2]], "maroon", 3, vecClassStyles[c % len(vecClassStyles)], "Class " + str(vecClasses[c]) + " Avg. Variance Intrinsic Dimension"))
                vecPlots.append(([d["AvgVariance"] for d in vecClassResults[c][2]], "grey", 3, vecClassStyles[c % len(vecClassStyles)], "Class " + str(vecClasses[c]) + " Avg. Variance"))
                vecPlots.append(([d["AvgVariance"] / d["VarID"] for d in vecClassResults[c][2]], "midnightblue", 3, vecClassStyles[c % len(vecClassStyles)], "Class " + str(vecClasses[c]) + " Var. Variance"))
            
    if 3 in vecStats:
        vecPlots.append(([d["PersistentEntropy"] for d in vecResults[3]], "limegreen", 3, "solid", "Persistent Entropy"))
        #vecPlots.append(([d["NumBars"] for d in vecResults[3]], "midnightblue", 3, "solid", "Number of Lifetimes"))
        
        if vecClasses is not None:
            for c in range(len(vecClasses)):
                vecPlots.append(([d["PersistentEntropy"] for d in vecClassResults[c][3]], "limegreen", 3, vecClassStyles[c % len(vecClassStyles)], "Class " + str(vecClasses[c]) + " Persistent Entropy"))
    
    if 4 in vecStats:
        #vecPlots.append(([d["Compression"] for d in vecResults[4]], "dodgerblue", 3, "solid", "Class Compression"))
        #vecPlots.append(([d["Discrimination"] for d in vecResults[4]], "midnightblue", 3, "solid", "Class Discrimination"))
        #vecPlots.append(([d["AverageClassCenterEntropy"] / d["Log(HostDim)"] for d in vecResults[4]], "cyan", 3, "solid", "Entropy of Class Center Co-ords"))
        
        vecPlots.append(([np.mean(np.array(d["MinIntraClassDPS"])) for d in vecResults[4]], "red", 3, "dotted", "Average Min Class DPS"))
        vecPlots.append(([np.mean(np.array(d["AvgIntraClassDPS"])) for d in vecResults[4]], "red", 3, "solid", "Average Avg Intra Class DPS"))
        vecPlots.append(([np.mean(np.array(d["MinIntraClassDPS"])) for d in vecResults[4]], "red", 3, "dashed", "Average Min Intra Class DPS"))

        vecPlots.append(([np.mean(np.array(d["AvgInterClassDPS"])) for d in vecResults[4]], "orange", 3, "solid", "Average Avg Inter Class DPS"))
        vecPlots.append(([np.mean(np.array(d["MaxInterClassDPS"])) for d in vecResults[4]], "orange", 3, "dashed", "Average Max Inter Class DPS"))
        #vecPlots.append(([np.mean(np.array(d["MaxClassDPS"])) for d in vecResults[4]], "red", 3, "dashed", "Average Max Class DPS"))

        vecPlots.append(([np.mean(np.array(d["AvgClassNorm"])) for d in vecResults[4]], "black", 3, "solid", "Average Avg Class Norm"))

    if 5 in vecStats:
        vecPlots.append(([smi for smi in vecResults[5]], "cyan", 3, "solid", "Sliced Mutual Information"))

    if 6 in vecStats:
        #vecPlots.append(([d["DistanceEntropy"] for d in vecResults[6]], "limegreen", 3, "solid", "Distance Entropy"))
        #vecPlots.append(([min(d["ClassCompression"]) for d in vecResults[6]], "limegreen", 3, "dotted", "Minimum Class Compression"))
        #vecPlots.append(([np.mean(np.array(d["ClassCompression"])) for d in vecResults[6]], "limegreen", 3, "solid", "Average Class Compression"))
        #vecPlots.append(([max(d["ClassCompression"]) for d in vecResults[6]], "limegreen", 3, "dashed", "Maximum Class Compression"))

        #vecPlots.append(([min(d["ClassSeparation"]) for d in vecResults[6]], "midnightblue", 3, "dotted", "Minimum Class Separation"))
        #vecPlots.append(([np.mean(np.array(d["ClassSeparation"])) for d in vecResults[6]], "midnightblue", 3, "solid", "Average Class Separation"))
        #vecPlots.append(([max(d["ClassSeparation"]) for d in vecResults[6]], "midnightblue", 3, "dashed", "Maximum Class Separation"))


        #vecPlots.append(([min(d["MaxIntraClassDist"]) for d in vecResults[6]], "cyan", 3, "dotted", "Minimum Max Intra Class Distance"))
        #vecPlots.append(([np.mean(np.array(d["MaxIntraClassDist"])) for d in vecResults[6]], "cyan", 3, "solid", "Average Max Intra Class Distance"))
        #vecPlots.append(([max(d["MaxIntraClassDist"]) for d in vecResults[6]], "cyan", 3, "dashed", "Maximum Max Intra Class Distance"))

        #vecPlots.append(([min(d["AvgIntraClassDist"]) for d in vecResults[6]], "deepskyblue", 3, "dotted", "Minimum Avg Intra Class Distance"))
        #vecPlots.append(([np.mean(np.array(d["AvgIntraClassDist"])) for d in vecResults[6]], "deepskyblue", 3, "solid", "Average Avg Intra Class Distance"))
        #vecPlots.append(([max(d["AvgIntraClassDist"]) for d in vecResults[6]], "deepskyblue", 3, "dashed", "Maximum Avg Intra Class Distance"))

        #vecPlots.append(([np.mean(np.array(d["MinIntraClassDist"])) for d in vecResults[6]], "red", 3, "solid", "Average Min Intra Class Distance"))

        #vecPlots.append(([min(d["MinInterClassDist"]) for d in vecResults[6]], "maroon", 3, "dotted", "Minimum Min Inter Class Distance"))
        #vecPlots.append(([np.mean(np.array(d["MinInterClassDist"])) for d in vecResults[6]], "maroon", 3, "solid", "Average Min Inter Class Distance"))
        #vecPlots.append(([max(d["MinInterClassDist"]) for d in vecResults[6]], "maroon", 3, "dashed", "Maximum Min Inter Class Distance"))

        #vecPlots.append(([min(d["AvgInterClassDist"]) for d in vecResults[6]], "red", 3, "dotted", "Minimum Avg Inter Class Distance"))
        #vecPlots.append(([np.mean(np.array(d["AvgInterClassDist"])) for d in vecResults[6]], "red", 3, "solid", "Average Avg Inter Class Distance"))
        #vecPlots.append(([max(d["AvgInterClassDist"]) for d in vecResults[6]], "red", 3, "dashed", "Maximum Avg Inter Class Distance"))

        vecPlots.append(([np.mean(np.array(d["MinClassNorm"])) for d in vecResults[6]], "black", 3, "dotted", "Average Min Class Norm"))
        vecPlots.append(([np.mean(np.array(d["AvgClassNorm"])) for d in vecResults[6]], "black", 3, "solid", "Average Avg Class Norm"))
        vecPlots.append(([np.mean(np.array(d["MaxClassNorm"])) for d in vecResults[6]], "black", 3, "dotted", "Average Max Class Norm"))
        #vecPlots.append(([np.mean(np.array(d["MaxClassNorm"])) - np.mean(np.array(d["MinClassNorm"])) for d in vecResults[6]], "black", 3, "dashed", "Average Norm Difference"))

    TPlotAnalyses(strT, vecPlots, [L], bNormalize, bNormalizeLinSep, vecNormalize = None)
    
    return

def TPlotClassificationEfficiency(T: KDTrainer, iStartLayer: int = 0, bNormalize: bool = True, fDataFrac: float = 1.0) -> None:
    strT = T.GetValue("Model") + " Layerwise Knowledge Quality on " + T.GetValue("Dataset")

    if T.dsData is None: T.dsData = T.InitDataset()
    if T.tModel is None: T.ILoadModel()
    L = [i for i in range(T.tModel.iGenLayers)]
    L = L[iStartLayer:]

    vecResults, _ = TGetPerLayerStats(T, [1, 4, 6, 7], iStartLayer = iStartLayer, fDataFrac = fDataFrac)

    vecPCAID = []
    vecPCAGID = []
    vecSVDE = []
    varThreshold = 0.95
    vecEVRs = [np.array(d["EVRatio"]) for d in vecResults[1]]
    for i in range(len(L)):
        tSVs = vecResults[7][i][:,:-1]
        fPCAID = 0
        #this computes the class-wise average embedding dimension w/ the specified threshold
        for c in range(tSVs.shape[0]):
            cnt = 0
            while torch.sum(tSVs[c, :cnt]) < varThreshold and cnt < tSVs.shape[1] - 1: cnt += 1
            fPCAID += cnt
        vecPCAID.append(fPCAID / tSVs.shape[0])

        tSVDE = tSVs[:, :int(fPCAID)]
        tSVDE = torch.where(tSVDE == 0, 1e-127, tSVDE)
        tSVDE /= torch.sum(tSVDE, dim = 1, keepdim = True)
        tSVDE = -1 * tSVDE * torch.log(tSVDE)
        tSVDE = torch.sum(tSVDE, dim = 1)
        #print(tSVDE)
        tSVDE[tSVDE != tSVDE] = 0 #replaces nans with zeros
        vecSVDE.append(torch.mean(tSVDE)) #average class-wise SVD entropy

        vecEVRs[i] = vecEVRs[i] / np.sum(vecEVRs[i])
        vecEVRs[i][np.where(vecEVRs[i] == 0)[0]] = 1e-127
        vecPCAGID.append(np.sum(np.where(vecEVRs[i] != 0, 1, 0)))

    vecSVDE = np.array(vecSVDE)
    vecPCAID = np.array(vecPCAID)
    #print(vecPCAID, vecSVDE)
    vecSVDENormalized = vecSVDE / np.log(vecPCAID)
    vecSVDENormalized2 = vecSVDE / np.log(5) #right now, all datasets are capped at 500 samples/class
    vecNormalizedPCAID = np.log(vecPCAID) / np.log(5)

    vecAvgIntraDist = np.array([np.mean(np.array(d["AvgIntraClassDist"])) for d in vecResults[6]])
    vecAvgInterDist = np.array([np.mean(np.array(d["AvgInterClassDist"])) for d in vecResults[6]])
    vecAvgMinInterDist = np.array([np.mean(np.array(d["MinInterClassDist"])) for d in vecResults[6]])

    vecAvgMaxNorm = np.array([np.mean(np.array(d["MaxClassNorm"])) for d in vecResults[4]])
    vecAvgNorm = np.array([np.mean(np.array(d["AvgClassNorm"])) for d in vecResults[4]])
    vecAvgMinNorm = np.array([np.mean(np.array(d["MinClassNorm"])) for d in vecResults[4]])
    vecAvgIntraDPS = np.array([np.mean(np.array(d["AvgIntraClassDPS"])) for d in vecResults[4]])
    vecAvgMinIntraDPS = np.array([np.mean(np.array(d["MinIntraClassDPS"])) for d in vecResults[4]])
    vecAvgMaxIntraDPS = np.array([np.mean(np.array(d["MaxIntraClassDPS"])) for d in vecResults[4]])
    vecAvgMaxInterDPS = np.array([np.mean(np.array(d["MaxInterClassDPS"])) for d in vecResults[4]])
    vecAvgInterDPS = np.array([np.mean(np.array(d["AvgInterClassDPS"])) for d in vecResults[4]])

    vecK = (T.dsData.Size() / np.pi)**(1/(np.array(vecPCAGID)))

    vecC1 = (vecAvgIntraDPS - vecAvgInterDPS) / (vecAvgMaxNorm - vecAvgMinNorm)

    vecC2 = (vecAvgIntraDPS - vecAvgInterDPS) * (vecAvgNorm * (1 - vecAvgMinIntraDPS)) / (vecAvgMaxNorm - 2*vecAvgMinInterDist)
    

    #vecSeparation = (vecAvgIntraDPS / vecAvgInterDPS)
    vecSeparation = (vecAvgIntraDPS - vecAvgInterDPS)

    #vecInformation = ((1 - vecAvgMinIntraDPS) / (vecAvgMaxInterDPS)) * vecSVDENormalized
    vecInformation = ((1 - vecAvgMinIntraDPS)) * vecSVDENormalized
    vecInformation2 = ((1 - vecAvgMinIntraDPS)) * vecSVDENormalized2
    vecInformation3 = ((1 - vecAvgMinIntraDPS)) * vecNormalizedPCAID

    #vecAvgMinInterDist + 
    vecEfficiencyM = (2*vecAvgMinInterDist) / (vecAvgNorm)
    vecEfficiencyScaled = vecEfficiencyM * vecK
    
    # * (5000 / np.pi)**(1/vecPCAID[i])
    # * (50000 / np.pi)**(1/vecPCAID[i])
    #(1 - vecAvgMinIntraDPS[i]) * 
    #(vecAvgIntraDPS[i] - vecAvgInterDPS[i]) * 
    #vecAvgNorm[i] * 
    vecPlots = []
    #vecPlots.append((vecC1, "cyan", 3, "solid", "Average Separation Efficiency"))
    #vecPlots.append((vecC2, "orange", 3, "solid", "C2"))
    
    #vecPlots.append((vecEfficiencyM, "blue", 3, "solid", "Efficiency"))

    vecPlots.append((vecSeparation, "midnightblue", 4, "solid", "Separation"))
    #vecPlots.append((vecInformation, "cyan", 4, "solid", "Information"))
    vecPlots.append((vecInformation2, "cyan", 4, "solid", "Information"))
    vecPlots.append((vecEfficiencyScaled, "red", 4, "solid", "Efficiency"))

    #vecPlots.append((vecInformation2, "cyan", 3, "dashed", "Information 2"))
    #vecPlots.append((vecSVDENormalized, "dodgerblue", 3, "solid", "SVDEN"))
    #vecPlots.append((vecSVDENormalized2, "dodgerblue", 3, "solid", "SVDEN-2"))
    #vecPlots.append((vecNormalizedPCAID, "dodgerblue", 3, "dashed", "PCA-ID"))


    vecQualityAM = vecSeparation**(1/2) + (vecInformation * vecEfficiencyScaled)
    vecQualityGM = (vecSeparation * vecInformation * vecEfficiencyScaled)**(1/3)

    vecSqrtIE = (vecInformation * vecEfficiencyScaled)**(1/2)
    vecQuality = vecSeparation + vecSqrtIE

    vecSqrtIE2 = (vecInformation2 * vecEfficiencyScaled)**(1/2)
    vecQuality2 = vecSeparation + vecSqrtIE2
    
    #vecPlots.append((vecQuality, "tab:orange", 5, "solid", "Quality"))
    #vecPlots.append((vecQualityGM, "tab:orange", 3, "dashed", "Geometric Mean Quality"))
    #vecPlots.append((vecQualityAM, "tab:orange", 3, "dotted", "Arithmetic Mean Quality"))
    #vecPlots.append(([1 for _ in range(len(L))], "black", 3, "dashed", "Y = 1"))
    #vecPlots.append((vecSqrtIE, "tab:purple", 3, "solid", "SqrtIE"))
    #vecPlots.append((vecQuality, "dimgrey", 4, "solid", "Knowledge Quality"))

    vecPlots.append((vecQuality2, "dimgrey", 4, "solid", "Knowledge Quality"))
    #vecPlots.append((vecPCAID / 500, "black", 5, "solid", "ED"))
    #vecPlots.append((vecQuality3, "tab:green", 3, "dashed", "Quality 3"))


    TPlotAnalyses(strT, vecPlots, [L], bNormalize = bNormalize)

    return {
        "Quality": vecQuality,
        "QualityAM": vecQualityAM,
        "QualityGM": vecQualityGM,
        "Quality2": vecQuality2,
        "Separation": vecSeparation,
        "Information2": vecInformation2,
        "Efficiency": vecEfficiencyScaled
    }

def TPlotPCADetails(T: KDTrainer, vecThresholds: list[float] = [0.5, 0.6, 0.7, 0.8, 0.9], iStartLayer: int = 0, bNormalize: bool = True) -> None:
    strT = T.GetValue("Model") + " Per-Layer Complexity of " + T.GetValue("Dataset") + " Representations"
    vecResults, _ = TGetPerLayerStats(T, [1])
    vecEVRs = [np.array(d["EVRatio"]) for d in vecResults[1]]
    if T.tModel is None: T.ILoadModel()
    X = [i for i in range(iStartLayer, T.tModel.iGenLayers)]

    vecPlots = []
    vecPlots.append(([d["HostDim"] for d in vecResults[1]], "black", 3, "solid", "Host Dimension"))
    
    vecPCAEntropy = []
    vecPCAEntropyBound = []
    for i in range(len(X)):
        vecEVRs[i] = vecEVRs[i] / np.sum(vecEVRs[i])
        vecEVRs[i][np.where(vecEVRs[i] == 0)[0]] = 1e-127
        vecPCAEntropy.append(-1 * np.sum(vecEVRs[i] * np.log(vecEVRs[i])))
        vecPCAEntropyBound.append(np.log(vecEVRs[i].shape[0]))
    vecPlots.append((vecPCAEntropy, "red", 3, "solid", "SVD Entropy"))
    vecPlots.append((vecPCAEntropyBound, "orange", 3, "solid", "SVD Entropy Upper Bound"))

    sf = 1
    for i in range(len(vecThresholds)):
        t = vecThresholds[i]
        vecPCAID = []
        for j in range(len(X)):
            cnt = 0
            while np.sum(vecEVRs[j][:cnt]) < t and cnt < vecEVRs[j].shape[0]: cnt += 1
            vecPCAID.append(cnt)
        if max(vecPCAID) > sf: sf = max(vecPCAID)
        vecPlots.append((vecPCAID, vecShadesOfBlue[i % len(vecShadesOfBlue)], 3, "solid", "PCA-ID (" + str(t * 100) + "% EV)"))

    for i in range(3, len(vecPlots)):
        vecID, color, width, style, name = vecPlots[i]
        vecPlots[i] = ([pca * max(vecPCAEntropyBound) / sf for pca in vecID], color, width, style, name)

    vecPlots.append(([max(vecPCAEntropyBound) * vecPCAEntropy[i] / np.log(vecPlots[-1][0][i]) for i in range(len(vecPCAEntropy))], "limegreen", 3, "solid", "Normalized SVD Entropy"))

    TPlotAnalyses(strT, vecPlots, [X], bNormalize = bNormalize)

    return


def TPlotPCASurface(T: KDTrainer, iStartIdx: int = 0, iStopIdx: int = 512, bCumulative: bool = True, iStartLayer: int = 0) -> None:
    vecResults, _ = TGetPerLayerStats(T, [1], [0, 1], iStartLayer = iStartLayer)
    nEVRs = np.array([d["EVRatio"][iStartIdx:iStopIdx] for d in vecResults[1]])
    if bCumulative:
        nEVRs = np.cumsum(nEVRs, axis = 1)

    if T.tModel is None: T.ILoadModel()
    x = np.array([i for i in range(iStartLayer, T.tModel.iGenLayers)])
    y = np.arange(nEVRs.shape[1])
    #print(np.min(nEVRs), np.max(nEVRs))
    X, Y = np.meshgrid(y, x)
    #print(X.shape, Y.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    mplColorNorm = colors.LogNorm(np.min(nEVRs), np.max(nEVRs), clip = False)
    ax.plot_surface(X, Y, nEVRs, rstride=15, cstride=15, edgecolor='royalblue', linewidth=1, antialiased=False, alpha = 0.3)

    plt.show()
    plt.close()

    return

def TPlotHODSurface(T: KDTrainer) -> None:
    vecResults, _ = TGetPerLayerStats(T, [9])
    nHODs = np.array([d["Avgs"] for d in vecResults[9]])
    nAD = np.array([d["AD"] for d in vecResults[9]])
    nAD = nAD[:, np.newaxis]
    nHODs /= nAD
    print(nHODs.shape, nAD.shape)

    if T.tModel is None: T.ILoadModel()
    x = np.array([i for i in range(T.tModel.iGenLayers)])
    y = np.arange(nHODs.shape[1])
    X, Y = np.meshgrid(y, x)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    mplColorNorm = colors.LogNorm(np.min(nHODs), np.max(nHODs), clip = False)
    ax.plot_surface(X, Y, nHODs, rstride=15, cstride=15, edgecolor='royalblue', linewidth=1, antialiased=False, alpha = 0.3)

    plt.show()
    plt.close()

    return

def TPlotPCASurfaceCollapsed(T: KDTrainer, iStartIdx: int = 0, iStopIdx: int = 512, bCumulative: bool = True, iStartLayer: int = 0) -> None:
    vecResults, vecClassResults = TGetPerLayerStats(T, [1], [0, 1], iStartLayer = iStartLayer)
    nEVRs = np.array([d["EVRatio"][iStartIdx:iStopIdx] for d in vecResults[1]])
    if bCumulative:
        nEVRs = np.cumsum(nEVRs, axis = 1)

    if T.tModel is None: T.ILoadModel()
    vecShades = [(i / (2 * (T.tModel.iGenLayers - iStartLayer)), i / (2 * (T.tModel.iGenLayers - iStartLayer)), i / (T.tModel.iGenLayers - iStartLayer)) for i in range(iStartLayer, T.tModel.iGenLayers)]
    #print(vecShades)
    for i in range(nEVRs.shape[0]):
        plt.plot([x for x in range(iStartIdx, iStopIdx)], nEVRs[i,:], color = vecShades[i], linewidth = 2)

    plt.show()
    plt.close()

    return

def TPlotDistHist(T: KDTrainer, iStartLayer: int = 0) -> None:
    vecResults, _ = TGetPerLayerStats(T, [6], iStartLayer = iStartLayer)
    if T.tModel is None: T.ILoadModel()
    vecShades = [(i / (2 * (T.tModel.iGenLayers - iStartLayer)), i / (2 * (T.tModel.iGenLayers - iStartLayer)), i / (T.tModel.iGenLayers - iStartLayer)) for i in range(T.tModel.iGenLayers - iStartLayer)]
    for i in range(T.tModel.iGenLayers - iStartLayer):
        plt.plot(vecResults[6][i]["DistanceHistValues"][1:], 
                 vecResults[6][i]["DistanceHistCounts"], color = vecShades[i], linewidth = 3)

    plt.show()
    plt.close()

    return

def TPlotStatvsModels(vecT: list[KDTrainer], iStat: int, bNormalize: bool = False) -> None:
    strData = vecT[0].GetValue("Dataset")
    strModels = "["
    vecVResults = []
    for T in vecT:
        vecTemp, _ = TGetPerLayerStats(T, [iStat])
        vecVResults.append(vecTemp)
        if T.GetValue("Dataset") != strData:
            print("Error! Expected all trainers to use the same dataset!")
            return
        strModels += T.GetValue("Model") + ", "
    strModels = strModels[:-2]
    strModels += "]"
    
    vecPlots = []
    vecX = []
    iT = 0
    for vecRes in vecVResults:
        if iStat == 0:
            print("TODO!")
        elif iStat == 1:
            vecPlots.append(([d["PCAID"] for d in vecRes[1]], "red", 3, vecClassStyles[iT % len(vecClassStyles)], 
                             vecT[iT].GetValue("Model") + " PCA Intrinsic Dimension"))
        elif iStat == 2:
            print("TODO")
        elif iStat == 3:
            vecPlots.append(([d["PersistentEntropy"] for d in vecRes[3]], "limegreen", 3, vecClassStyles[iT % len(vecClassStyles)], 
                             vecT[iT].GetValue("Model") + " Persistent Entropy"))
        elif iStat == 4:
            print("TODO")
        elif iStat == 5:
            vecPlots.append(([smi for smi in vecRes[5]], "cyan", 3, vecClassStyles[iT % len(vecClassStyles)], 
                             vecT[iT].GetValue("Model") + " Sliced Mutual Information"))
        
        X = [i for i in range(len(vecPlots[-1][0]))]
        vecX.append([x / max(X) for x in X])
        iT += 1

    strT = vecReadableStatNames[iStat] + " For " + strModels + " On " + strData
    TPlotAnalyses(strT, vecPlots, vecX, bNormalize)

    return

def TPlotAnalyses(strT: str, vecPlots, vecX: list[float], bNormalize: bool = True, bNormalizeLinSep: bool = True, vecNormalize: list[float] = None) -> None:
    '''
    TODO: figure out a better way of managing relative depth
    '''
    bSeparateMode = len(vecX) > 1
    fMaX = 0
    fMaY = 0
    for n in range(len(vecPlots)):
        Y, color, width, style, name = vecPlots[n]
        Yhd = None
        if vecNormalize is not None and name not in ["Ambient Dimension", "Linear Seperability"]:
            Y = [Y[i] / vecNormalize[i] for i in range(len(vecX[n] if bSeparateMode else vecX[0]))]
        if name == "Ambient Dimension": #or (bNormalize and not bNormalizeLinSep and name != "Linear Separability") or (bNormalize and bNormalizeLinSep):
            # if max(Y) != min(Y):
            #     Y = [(y - min(Y)) / (max(Y) - min(Y)) for y in Y]
            # else:
            #     Y = [y / max(Y) for y in Y]
            print(Y)
            Y = [y / max(Y) for y in Y]
            if Yhd is not None:
                Yhd = [(y - min(Yhd)) / (max(Yhd) - min(Yhd)) for y in Yhd]
        
        if bSeparateMode and max(vecX[n]) > fMaX:
            fMaX = max(vecX[n])

        if not bNormalize and max(Y) > fMaY:
            fMaY = max(Y)

        #if "Host Dimension" in name: continue #comment in to remove the solid black HD line
        '''if 0:
            #random temp stuff
            vecD = [0 for _ in range(len(Y))]
            vecD2 = [0 for _ in range(len(Y))]
            for i in range(1, len(Y)):
                vecD[i] = Y[i] - Y[i - 1]
            for i in range(2, len(Y)):
                vecD2[i] = (Y[i] - Y[i - 1]) - (Y[i - 1] - Y[i - 2])
            #plt.plot(L, vecD, label = "Delta", linewidth = 2, color = "black")
            #plt.plot(L, vecD2, label = "Delta2", linewidth = 2, color = "black", linestyle = "dashed")

            #plt.scatter(9, (Y[9] - min(Y)) / (max(Y) - min(Y)), color = "blue", s = 200)
            if "Class" not in name:
                idx = torch.argmax(torch.tensor(Y)).item()
                print(idx)
                plt.fill_between(L[:idx+1], [0 for _ in range(len(L[:idx+1]))], Y[:idx+1], alpha = 0.5, facecolor = "blue")
                plt.fill_between(L[idx:], [0 for _ in range(len(L[idx:]))], Y[idx:], alpha = 0.5, facecolor = "orange")

                #plt.scatter(L[12:17], Y[12:17], s = 320, color = "black")'''

        plt.plot(vecX[n] if bSeparateMode else vecX[0], Y, color = color, linewidth = width, linestyle = style, label = name)

    plt.legend(fontsize = 24)
    plt.title(strT, fontsize = 36)
    plt.xlabel("Relative Depth" if bSeparateMode else "Layer Index", fontsize=32)
    strY = "Metrics"
    if bNormalize: strY += " (Normalized)"
    plt.ylabel(strY, fontsize=32)
    plt.xticks(np.linspace(0, fMaX, 20) if bSeparateMode else vecX[0], fontsize = 26)
    #plt.yticks(np.linspace(0, fMaY, 10) if not bNormalize else [i/10 for i in range(11)], fontsize = 16)
    plt.yticks(np.arange(0, 2.2, 0.2), fontsize = 26)
    #plt.grid(color="black", linestyle="dashed", linewidth = 1)
    plt.grid()
    
    plt.show()
    plt.close()

    return

def TPlotFeaturePCA(T: ITrainer, iDataset: int = 0, iLayer: int = -1, iComps: int = 2) -> None:
    T.LoadTrainedModel()
    T.tModel.eval()
    if T.GetValue("Dataset") != "CIFAR100S":
        strT = T.GetValue("Dataset") + " " + T.GetValue("Model")
    else:
        strT = T.GetValue("Dataset") + " Subset " + str(T.GetValue("Subset")) + " " +  T.GetValue("Model")
    return IPlotFeaturePCA(T.tModel, iDataset, iLayer, iComps, strT)

def TMeasureFeatureSeperability(T: ITrainer, iLayer: int = -1, cTransform: Callable[[torch.tensor], torch.tensor] = None) -> float:
    T.LoadTrainedModel()
    T.tModel.eval()
    if iLayer < 0:
        iLayer += T.tModel.iHeadIdx
    T.GenPerLayerFeatures([iLayer])
    if T.dlTrain is None: T.ILoadDataset()

    strF = T.GetCurrentFolder() + "Features/" + T.GenFeatureString(iLayer, bTeacher = False) + ".pkl"
    printf("Loading Features: {}".format(strF), INFO)
    with open(strF, "rb") as f:
        F = torch.load(f)

    if cTransform is not None:
        F = cTransform(F)

    return IMeasureSeperability(F, T.dlTrain[1])

def TGenerateAndSaveFeatures(T: ITrainer, vecFeatureLayers: list[int] = [-1]) -> None:
    T.LoadTrainedModel()
    T.ILoadDataset()
    T.tModel.eval()
    X, Y = T.dlTrain
    
    vecF = IGenerateFeatures(T.tModel, X, Y, vecFeatureLayers = vecFeatureLayers)
    
    strSaveDir = T.GetCurrentFolder() + "Features/"
    if not os.path.isdir(strSaveDir):
        os.mkdir(strSaveDir)
    
    for i in range(len(vecF)):
        tF = vecF[i]
        iL = vecFeatureLayers[i]
        strSavePath = strSaveDir + T.GenFeatureString(iL) + ".pkl"
        with open(strSavePath, "wb") as f:
            torch.save(tF, f)
        
    return

def TGenerateAndSaveLogits(T: ITrainer, iFeatureLayer: int = -1, dTemperature: float = 2.) -> None:
    T.LoadTrainedModel()
    T.tModel.eval()
    if T.dlTrain is None: T.ILoadDataset()
    
    strFeaturePath = T.GetCurrentFolder() + "Features/"
    strFeaturePath += T.GetValue("Dataset")
    if "CIFAR100" in strFeaturePath:
        strFeaturePath = strFeaturePath[:-1] #strip the CIFAR100 indentifier
    strLogitPath = strFeaturePath + "_Layer" + str(iFeatureLayer) + "_Logits_T" + str(dTemperature) + ".pkl"
    strFeaturePath += "_Layer" + str(iFeatureLayer) + "_Features.pkl"
    
    if not os.path.exists(strFeaturePath):
        printf("Could not find file {}".format(strFeaturePath), INFO)
        printf("Generating Features...", INFO)
        F = IGenerateFeatures(T.tModel, T.dlTrain[0], T.dlTrain[1], bPerClass = False, iFeatureLayer = iFeatureLayer)
    
        printf("Saving {} features (size {}) from layer {}".format(T.IGenHash(), F.shape[1], iFeatureLayer))
        with open(strFeaturePath, "wb") as f:
            torch.save(F, f)
    else:
        with open(strFeaturePath, "rb") as f:
            F = torch.load(f)
            
    L = IGenerateSoftenedLogits(F, T.dlTrain[1], dTemperature)
    printf("Saving {} logits (T = {:.2f}) from layer {}".format(T.IGenHash(), dTemperature, iFeatureLayer))
    with open(strLogitPath, "wb") as f:
        torch.save(L, f)
        
    return

def TComputeLRSweep(T: ITrainer, iMaxIter: int = 5000, vecLRs: list[float] = [0.0001, 0.0005, 0.001, 0.0025, 0.005, 0.0075,
                                                                             0.01]) -> list[float]:
                                                                             
    if T.dlTrain is None: T.ILoadDataset()
    T.Cfg["Scheduler"] = "None"

    strSavePath = T.GetCurrentFolder() + "LRSweep.json"
    
    if os.path.exists(strSavePath):
        print("Warning! Found existing learning rate sweep result for config {}".format(T.IGenHash()))
        if not GetInput("Overwrite? (Y/X)\n"):
            return None
    
    iIterPerEpoch = T.dlTrain[0].shape[0] // T.GetValue("BatchSize")
    
    vecRes = []
    
    for lr in vecLRs:
        T.Cfg["LearningRate"] = lr
        T.ILoadComponents()
        T.tModel.train()
        print("Training Model w/ LR {} for {} epochs".format(lr, (iMaxIter // iIterPerEpoch) + 1))
        for n in range((iMaxIter // iIterPerEpoch) + 1):
            T.Train()
        vecRes.append(T.Eval()[T.iPerfIdx])
    T.Cfg["LearningRate"] = -1
    T.UpdateCacheMap()
    
    with open(strSavePath, "w") as f:
        json.dump({"TestMetric": vecRes, "LRs": vecLRs}, f)
        
    return vecRes

def TPlotLRSweep(T: ITrainer) -> None:
    T.Cfg["Scheduler"] = "None"
    T.Cfg["LearningRate"] = -1
    strSavePath = T.GetCurrentFolder() + "LRSweep.json"
    
    if not os.path.exists(strSavePath):
        T.PrintConfigSummary()
        print("No LR Sweep found for config {}".format(T.IGenHash()))
        if GetInput("Generate? (Y/X)\n"):
            TComputeLRSweep(T)
        else:
            return   
        
    with open(strSavePath, "r") as f:
        dRes = json.load(f)
        
    plt.plot(dRes["LRs"], dRes["TestMetric"])
    plt.scatter(dRes["LRs"], dRes["TestMetric"])
    plt.grid()
    
    plt.title(T.Cfg["Model"] + " " + T.Cfg["Dataset"] + " LR Sweep", fontsize = 24)
    
    plt.show()
    plt.close()
    
    return    