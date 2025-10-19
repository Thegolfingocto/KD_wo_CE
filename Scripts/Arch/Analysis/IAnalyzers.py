'''
Code written by Nicholas J. Cooper.
Released under the MIT license, see the GitHub page for the full legal boilerplate.
tldr: you freely can do whatever you like with this code, but please cite the GitHub at: https://github.com/Thegolfingocto/KDwoCE.git
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
'''









import os
import matplotlib.pyplot as plt
import math
import numpy as np
from inspect import ismethod
from dataclasses import dataclass
from typing import Iterable
import json

import torch

try:
    from Arch.ICache import ICache
    from Arch.Logger import *
    import Arch.Logger as Logger
    import Arch.ITrainer as ITrainer
    import Arch.Utils.Utils as Utils
except:
    from ICache import ICache
    from Logger import *
    import Logger
    import ITrainer
    import Utils.Utils as Utils


@dataclass
class mplPlot():
    Y: Iterable

    label: str
    color: str
    width: int
    style: str


class IStat(object):
    def __init__(self, strName: str, strDisplayName: str = None, strFormat: str = ".json") -> None:
        '''
        Generic stat class for managing different analyses. Pass the (display) name and format to benefit from subdirectory management features.
        If subfolder structure is provided in strName, the corresponding folders will be created. Currently only 1 level sub-folders are supported.
        '''
        self.strName = strName
        self.strDisplayName = strDisplayName if strDisplayName is not None else strName
        self.strFormat = strFormat

        self.bSubFolder = "/" in strName
        self.strSubAnalysisDir = None

    def HasMethod(self, strQ: str) -> bool:
        return (hasattr(self, strQ) and ismethod(getattr(self,strQ)))
    
    def SetSubAnalysisDir(self, strSubAnalysisDir: str) -> None:
        self.strSubAnalysisDir = strSubAnalysisDir
    
    def GenPath(self, Trainer: ITrainer.ITrainer) -> str:
        strPath = Trainer.GetCurrentFolder() + "Analysis/"
        if self.strSubAnalysisDir is not None:
            strPath += self.strSubAnalysisDir
            if strPath[-1] != "/": strPath += "/"
        if self.bSubFolder: strPath += self.strName.split("/")[0] + "/" + self.strName.split("/")[1]
        else: strPath += self.strName + self.strFormat

        return strPath
    
    def Get(self, Trainer: ITrainer.ITrainer) -> object:
        '''
        This is the main entry point for all analysis access. Will create necessary directories if missing.
        '''
        strBaseDir = Trainer.GetCurrentFolder()
        assert os.path.isdir(strBaseDir), "Config Directory {} Missing!".format(strBaseDir)

        strBaseDir += "Analysis/"
        if not os.path.isdir(strBaseDir): os.mkdir(strBaseDir)

        if self.strSubAnalysisDir is not None:
            strBaseDir += self.strSubAnalysisDir
            if strBaseDir[-1] != "/": strBaseDir += "/"
            if not os.path.isdir(strBaseDir): os.mkdir(strBaseDir)

        if not self.bSubFolder:
            if os.path.exists(strBaseDir + self.strName + self.strFormat):
                return Utils.ReadFile(strBaseDir + self.strName + self.strFormat)

            return None
            
        if not os.path.isdir(strBaseDir + self.strName.split("/")[0] + "/"): os.mkdir(strBaseDir + self.strName.split("/")[0] + "/")

    def Save(self, oData: object, Trainer: ITrainer.ITrainer) -> None:
        #No need for path checks, everything goes through IGet() first
        strPath = self.GenPath(Trainer)
        Utils.WriteFile(oData, strPath)

        return

    def ICompute(self, tData: torch.tensor, **kwargs) -> object:
        assert self.HasMethod("Compute"), "Missing Compute() Function!"

        #pass to user defined computation
        return self.Compute(tData, **kwargs)

    def IPlot(self, vecData: list[object]) -> list[mplPlot]:
        assert self.HasMethod("Plot"), "Missing Plot() Function!"

        return self.Plot(vecData)

class IAnalyzer(object):
    '''
    Generic interface class for managing and plotting stats computed from ITrainers. Pass in a list of trainers at construction time, 
    then add desired stats with AddStat().
    '''
    def __init__(self, vecTrainers: list[ITrainer.ITrainer]) -> None:
        self.vecTrainers = vecTrainers

        self.vecStats: list[IStat] = []
        self.vecData: list[object] = None

        self.strTitle = None
        self.vecDisplayTitles = []
        vecAllowedFixedKeys = ["Dataset", "Model"]
        if len(self.vecTrainers) > 1:
            for strKey in vecAllowedFixedKeys:
                bFound = True
                strV = self.vecTrainers[0].GetValue(strKey)
                for t in self.vecTrainers[1:]:
                    #print(strV, t.GetValue(strKey))
                    bFound = bFound and (t.GetValue(strKey) == strV)
                if bFound:
                    self.strTitle = strKey
                else: self.vecDisplayTitles.append(strKey)
        #print(self.vecDisplayTitles)
        self.strBackupDisplayKey = "Model"
        self.strBackupTitleKey = "Dataset"

        return

    def HasMethod(self, strQ: str) -> bool:
        return (hasattr(self, strQ) and ismethod(getattr(self,strQ)))

    def AddStat(self, Stat: IStat) -> None:
        self.vecStats.append(Stat)
        return
    
    def IGetStats(self, **kwargs) -> None:
        assert self.HasMethod("GetStats"), "GetStats() Not Implemented!"

        return self.GetStats(**kwargs)
    
    def IPlotStats(self, **kwargs) -> None:
        assert self.HasMethod("PlotStats"), "PlotStats() Not Implemented!"
        self.IGetStats(**kwargs)
        #print(kwargs)
        return self.PlotStats(**kwargs)


class PerLayerFeatureAnalyzer(IAnalyzer):
    '''
    This is a specific analyzer which computes and plots stats computed from the latent features across the layers of ITrainer Models. 
    Packages the stats computed into lists of the form [layer1 stat, layer2 stat, etc...].
    '''
    def __init__(self, vecTrainers: list[ITrainer.ITrainer], iStartLayer: int = 0, bConsolidateTrainerPlots: bool = True,
                 strDisplayKey: str = None, strTitleKey: str = None, vecNormalizeNames: list[str] = [],
                 iPlotsPerRow: int = 2) -> None:
        super().__init__(vecTrainers)

        self.iStartLayer = iStartLayer
        self.bConsolidateTrainerPlots = bConsolidateTrainerPlots
        if strDisplayKey is not None: self.strBackupDisplayKey = strDisplayKey
        if strTitleKey is not None: self.strBackupTitleKey = strTitleKey
        self.vecNormalizeNames = vecNormalizeNames
        self.iPlotsPerRow = iPlotsPerRow

        return
    
    def IsNormalized(self, strQ: str):
        bRet = False
        for strName in self.vecNormalizeNames:
            bRet = bRet or strName in strQ
        return bRet

    def GetStats(self, **kwargs) -> None:
        self.vecData = [[] for _ in range(len(self.vecTrainers))]

        for i in range(len(self.vecTrainers)):
            Trainer = self.vecTrainers[i]
            if Trainer.tModel is None: Trainer.ILoadModel()
            iNumLayers = Trainer.tModel.iGenLayers - self.iStartLayer

            vecMissingStats = []
            for Stat in self.vecStats:
                oData = Stat.Get(Trainer)
                if oData is None: vecMissingStats.append(Stat)
                else:
                    if len(oData) == iNumLayers: self.vecData[i].append(oData)
                    elif len(oData) > iNumLayers: self.vecData[i].append(oData[self.iStartLayer:])
                    else:
                        print("Something got messed up with iStartLayer. Running ReComputeStats() may solve the issue.")
                        return

            if len(vecMissingStats) > 0:
                iSS = 5000
                if "SampleSize" in kwargs: iSS = kwargs.get("SampleSize")
                self.ComputePerLayerStats(Trainer, vecMissingStats, iSampleSize = iSS)
                return self.GetStats()

    def PlotStats(self, **kwargs) -> None:
        #print(kwargs)
        vecPlots = [[] for _ in range(len(self.vecTrainers))]
        vecX = []
        fMaxY = -1
        bNormalizeX = False
        for iT in range(len(self.vecTrainers)):
            X = np.arange(self.vecTrainers[iT].tModel.iGenLayers - self.iStartLayer) + 1.0
            if len(vecX) > 0 and len(X) != len(vecX[-1]): bNormalizeX = True #adjust X axes if different numbers of layers are present
            vecX.append(X)
            for iS in range(len(self.vecStats)):
                vecPlots[iT] += self.vecStats[iS].IPlot(self.vecData[iT][iS])

        if bNormalizeX:
            for i in range(len(vecX)):
                vecX[i] /= vecX[i][-1]

        if self.bConsolidateTrainerPlots or len(self.vecTrainers) == 1:
            for iT in range(len(self.vecTrainers)):
                for plot in vecPlots[iT]:
                    if self.IsNormalized(plot.label): plot.Y = [y / max(plot.Y) for y in plot.Y]
                    if len(self.vecTrainers) > 1 and len(plot.label) > 0: plot.label = self.vecTrainers[iT].GetValue(self.strDisplayKey) + ": " + plot.label
                    if max(plot.Y) > fMaxY: fMaxY = max(plot.Y)

                    if iT == 0: plt.plot(vecX[iT], plot.Y, label = plot.label, color = plot.color, linewidth = plot.width, linestyle = plot.style)
                    else: plt.plot(vecX[iT], plot.Y, label = plot.label, linewidth = plot.width, linestyle = plot.style)
            plt.legend(fontsize = 24)

            if "strTitle" in kwargs:
                plt.title(kwargs.get("strTitle"), fontsize = 26)
            else:
                plt.title("Layerwise Analysis of " + str(self.vecTrainers[0].GetValue(self.strBackupTitleKey if self.strTitle is None else self.strTitle)), fontsize = 26)

            plt.xlabel("Relative Depth" if len(self.vecTrainers) > 1 else "Layer Index", fontsize=26)
            plt.ylabel("Metrics", fontsize = 26)

            plt.xticks(np.linspace(0, vecX[0][-1], 20) if len(self.vecTrainers) > 1 else vecX[0], fontsize = 20)
            plt.yticks(np.linspace(0, fMaxY, 20), fontsize = 20)
            plt.grid()

        else:
            iRows = math.ceil(len(self.vecTrainers) / self.iPlotsPerRow)
            fig, axes = plt.subplots(iRows, self.iPlotsPerRow)
            if iRows == 1: axes = [axes]
            if self.iPlotsPerRow == 1:
                for i in range(iRows): axes[i] = [axes[i]]

            for iT in range(len(self.vecTrainers)):
                iR = iT // self.iPlotsPerRow
                iC = iT - (iR * self.iPlotsPerRow)
                fMaxY = -1
                for plot in vecPlots[iT]:
                    if self.IsNormalized(plot.label): plot.Y = [y / max(plot.Y) for y in plot.Y]
                    if max(plot.Y) > fMaxY: fMaxY = max(plot.Y)

                    axes[iR][iC].plot(vecX[iT], plot.Y, label = plot.label, color = plot.color, linewidth = plot.width, linestyle = plot.style)
                
                strTK = self.strBackupDisplayKey if len(self.vecDisplayTitles) == 0 else self.vecDisplayTitles[0]
                axes[iR][iC].set_title(strTK + ": " + str(self.vecTrainers[iT].GetValue(strTK)), 
                                       fontsize = 14)

                axes[iR][iC].set_xticks(np.linspace(0, vecX[0][-1], 20) if len(self.vecTrainers) > 1 else vecX[0])
                axes[iR][iC].set_yticks(np.linspace(0, fMaxY, 20))
                axes[iR][iC].grid()

            vecLines = [] 
            vecLabels = [] 
            
            for ax in fig.axes: 
                Lines, Labels = ax.get_legend_handles_labels()

                for i in range(len(Labels)):
                    if Labels[i] not in vecLabels:
                        vecLines.append(Lines[i]) 
                        vecLabels.append(Labels[i]) 
            
            fig.legend(vecLines, vecLabels, loc='upper right', fontsize = 18) 

            if "strTitle" in kwargs:
                fig.suptitle(kwargs.get("strTitle"), fontsize = 26)
            else:
                fig.suptitle("Layerwise Analysis of " + self.vecTrainers[0].GetValue(self.strBackupTitleKey if self.strTitle is None else self.strTitle), fontsize = 26)        

        plt.show()
        plt.close()

        return

    def ComputePerLayerStats(self, Trainer: ITrainer.ITrainer, vecMissingStats: list[IStat], 
                             iSampleSize: int = 5000, iMaxDimension: int = 262144) -> None:
        #safety first!
        for Stat in vecMissingStats:
            if os.path.exists(Stat.GenPath(Trainer)):
                print("WARNING! Found existing data for stat: {}".format(Stat.strDisplayName))
                if not Utils.GetInput("Overwrite? (Y/X)"):
                    print("Quitting...")
                    return
                
        #store the results
        vecMissingData = [[] for _ in range(len(vecMissingStats))]

        #generate all features first
        if Trainer.tModel is None: Trainer.ILoadModel()
        vecL = [i for i in range(Trainer.tModel.iGenLayers)]
        vecL = vecL[self.iStartLayer:]
        Trainer.GenPerLayerFeatures(vecL)

        #grab the labels
        strFeatureDir = Trainer.GetCurrentFolder() + "Features/"
        Y = Utils.ReadFile(strFeatureDir + Trainer.GenFeatureStringLabel() + ".pkl")
        if len(Y.shape) == 2:
            iC = Y.shape[1]
            YA = torch.argmax(Y, dim = 1)
        else:
            iC = torch.max(Y) + 1
            YA = Y

        #subsample
        tIdx = Utils.GetRandomSubset(Y, iSampleSize)
        vecIdx = [torch.where(YA[tIdx,...] == c)[0] for c in range(iC)]
        #tIdx2 = Utils.GetRandomSubset(Y, min([iSampleSize // 10, Y.shape[0] // Y.shape[1]]) * Y.shape[1])

        #loop thru the features
        for iL in vecL:
            strF = Trainer.GetCurrentFolder() + "Features/" + Trainer.GenFeatureString(iL) + ".pkl"
            printf("Loading Features: {}".format(strF), INFO)
            with open(strF, "rb") as f:
                F = torch.load(f)

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

            for iS in range(len(vecMissingStats)):
                oData = vecMissingStats[iS].ICompute(F, tSampleIdx = tIdx, vecIdx = vecIdx, Trainer = Trainer, YA = YA)
                print("Layer: {}, Stat: {}".format(iL, vecMissingStats[iS].strDisplayName))
                print(oData)
                vecMissingData[iS].append(oData)

        #Save the results
        for iS in range(len(vecMissingStats)):
            vecMissingStats[iS].Save(vecMissingData[iS], Trainer)

        return
    


class LogitTrajectoryAnalyzer(IAnalyzer):
    def __init__(self, vecTrainers: list[ITrainer.ITrainer]) -> None:
        super().__init__(vecTrainers)

    def AddStat(self, Stat: IStat) -> None:
        Stat.SetSubAnalysisDir("LogitTrajectory")
        self.vecStats.append(Stat)
        return

    def GetStats(self, **kwargs) -> None:
        self.vecData = [[] for _ in range(len(self.vecTrainers))]

        for i in range(len(self.vecTrainers)):
            Trainer = self.vecTrainers[i]
            if Trainer.tModel is None: Trainer.ILoadModel()

            vecMissingStats = []
            for Stat in self.vecStats:
                oData = Stat.Get(Trainer)
                if oData is None: vecMissingStats.append(Stat)
                else: self.vecData[i].append(oData)

            if len(vecMissingStats) > 0:
                iSS = 5000
                if "SampleSize" in kwargs: iSS = kwargs.get("SampleSize")
                self.ComputeCheckpointStats(Trainer, vecMissingStats, iSampleSize = iSS)
                return self.GetStats()
    
    def ComputeCheckpointStats(self, Trainer: ITrainer.ITrainer, vecMissingStats: list[IStat], 
                             iSampleSize: int = 5000) -> None:
        #safety first!
        for Stat in vecMissingStats:
            if os.path.exists(Stat.GenPath(Trainer)):
                print("WARNING! Found existing data for stat: {}".format(Stat.strDisplayName))
                if not Utils.GetInput("Overwrite? (Y/X)"):
                    print("Quitting...")
                    return
                
        #store the results
        vecMissingData = [[] for _ in range(len(vecMissingStats))]

        vecCkpts = Trainer.GetCheckpointIntervals()
        if len(vecCkpts) == 0:
            print("ERROR! Found zero checkpoints for config {}".format(Trainer.IGenHash()))
            return

        #grab dataset
        if not Trainer.dsData.Loaded(): Trainer.ILoadDataset() #only load dataset if necessary
        X, Y = Trainer.dsData.X, Trainer.dsData.Y

        if len(Y.shape) == 2:
            iC = Y.shape[1]
            YA = torch.argmax(Y, dim = 1)
        else:
            iC = torch.max(Y) + 1
            YA = Y

        #subsample
        tIdx = Utils.GetRandomSubset(Y, iSampleSize)
        vecIdx = [torch.where(YA[tIdx,...] == c)[0] for c in range(iC)]
        #tIdx2 = Utils.GetRandomSubset(Y, min([iSampleSize // 10, Y.shape[0] // Y.shape[1]]) * Y.shape[1])

        #storage
        tLogits = torch.zeros((X.shape[0], iC))

        #loop thru the features
        for k in range(len(vecCkpts)):
            Trainer.LoadTrainedModel(iCkpt = int(vecCkpts[k]))

            iN = (X.shape[0] // 100) + 1
            with torch.no_grad():
                for i in range(iN):
                    iE = min([(i+1) * 100, X.shape[0]])
                    idx = torch.arange(i*100, iE, 1)
                    if idx.shape[0] == 0: break
                    tL = Trainer.tModel(X[idx].to(Trainer.device))
                    tLogits[i*100:iE,...] = tL.to("cpu")

            for iS in range(len(vecMissingStats)):
                oData = vecMissingStats[iS].ICompute(tLogits, tSampleIdx = tIdx, vecIdx = vecIdx, Trainer = Trainer, YA = YA)
                print("Checkpoint: {}, Stat: {}".format(vecCkpts[k], vecMissingStats[iS].strDisplayName))
                print(oData)
                vecMissingData[iS].append(oData)

        #Save the results
        for iS in range(len(vecMissingStats)):
            vecMissingStats[iS].Save(vecMissingData[iS], Trainer)

        return
    

    def PlotStats(self, **kwargs) -> None:
        #print(kwargs)
        vecPlots = [[] for _ in range(len(self.vecTrainers))]
        for iT in range(len(self.vecTrainers)):
            for iS in range(len(self.vecStats)):
                vecPlots[iT] += self.vecStats[iS].IPlot(self.vecData[iT][iS])

        vecX = self.vecTrainers[0].GetCheckpointIntervals()
        fMaxY = -1
        #Single trainer case ONLY TODO: extend
        for plot in vecPlots[0]:
            #if self.IsNormalized(plot.label): plot.Y = [y / max(plot.Y) for y in plot.Y]
            #if len(self.vecTrainers) > 1 and len(plot.label) > 0: plot.label = self.vecTrainers[iT].GetValue(self.strDisplayKey) + ": " + plot.label
            if max(plot.Y) > fMaxY: fMaxY = max(plot.Y)

            plt.plot(vecX, plot.Y, label = plot.label, color = plot.color, linewidth = plot.width, linestyle = plot.style)

        if "PlotResult" in kwargs and kwargs["PlotResult"]:
            self.vecTrainers[0].IDisplayResult(bPlotOnly = True)
        if "PlotTestLoss" in kwargs and kwargs["PlotTestLoss"]:
            dRes = self.vecTrainers[0].GetResult()
            idx = dRes["TestMetricNames"].index("TestLoss")

            vecTestLoss = [dRes["Runs"][0]["TestMetrics"][k][idx] for k in range(len(dRes["Runs"][0]["TestMetrics"]))]
            fRatio = fMaxY / max(vecTestLoss)

            plt.plot([e for e in range(self.vecTrainers[0].GetValue("NumEpochs"))],
                     [tl * fRatio for tl in vecTestLoss],
                     color = "red", linewidth = 3, label = "Test Loss")

        plt.legend(fontsize = 24)

        if "strTitle" in kwargs:
            plt.title(kwargs.get("strTitle"), fontsize = 26)
        else:
            plt.title("Logit Trajectory Analysis of " + str(self.vecTrainers[0].GetValue(self.strBackupTitleKey if self.strTitle is None else self.strTitle)), fontsize = 26)

        plt.xlabel("Training Epoch", fontsize=26)
        plt.ylabel("Metrics", fontsize = 26)

        plt.xticks(np.linspace(0, vecX[-1], 20) if len(self.vecTrainers) > 1 else vecX, fontsize = 20)
        plt.yticks(np.linspace(0, fMaxY, 20), fontsize = 20)
        plt.grid()


        plt.show()
        plt.close()