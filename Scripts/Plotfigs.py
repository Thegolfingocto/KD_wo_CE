'''
Code written by Nicholas J. Cooper.
Released under the MIT license, see the GitHub page for the full legal boilerplate.
tldr: you freely can do whatever you like with this code, but please cite the GitHub at: https://github.com/Thegolfingocto/KDwoCE.git
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
'''




import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from ConfigUtils import *
from Trainer import KDTrainer
from TrainerUtils import *

dDisplayNames = {
        "SingleLayerConv": "SLC",
        "ThreeLayerConv": "3LC",

        "CIFAR10": "CIFAR 10",
        "CIFAR100F": "CIFAR 100",
        "TinyImagenet": "Tiny ImageNet",

        "ViT_ETT": "ViT_ET",
        "ViT_B": "ViT_B",
        "MobileNetV2": "MNV2",
        "ResNet34": "RN34",
        "ResNet9": "RN9",
        "VGG19": "VGG19",
        "VGG11": "VGG11",

        "Baseline1LC+VKD": "Base FKD",
        "Baseline1LCFC+VKD": "Base FKD+FC",
        "Baseline3LC+VKD": "3LC",
        "Baseline3LCFC+VKD": "3LC+FC",
        "BaselineSP+VKD": "Sim. Pres.",
        "BaselineFC+VKD+SP": "SP+FC",
        "OursMulti": "Ours",
        "EndLayers": "Ours - Only S",
        "STL+SepLin": "Base FKD-LL",
        "VKD": "Van. KD",
        "VKD-T2": "Van. KD T2",
        "VKDNormalized": "Van. KD Std.",
        "OursMulti+CE+VKD": "Ours+LL",
        "OursMulti+CE": "Ours+CE",
        "Baseline1LC": "Base FKD-KL",

        "Ours-S": "S Only",
        "Ours-I": "I Only",
        "Ours-E": "E Only",
        "Ours-SqrtIE": "No S",
    }

def PlotDimensionExample() -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    

    x = np.linspace(0, 1.5, 15)
    for i in range(x.shape[0]):
        #print(x[i])
        ax.plot((x[i] - 1.5, x[i]), (x[i], x[i] - 1.5), 0, linewidth = 1, color = "slategray", linestyle = "dashed", alpha = 0.8)
        ax.plot((-x[i], 1.5 - x[i]), (x[i] - 1.5, x[i]), 0, linewidth = 1, color = "slategray", linestyle = "dashed", alpha = 0.8)

    theta = np.linspace(0, 2 * np.pi, 201)
    ax.plot(np.sin(theta),
            np.cos(theta), 0, linewidth = 5, color = "cyan")
    

    ax.plot((0,0),(0,0), (-2,2), '-k', label='z-axis')
    ax.plot((0,0), (-2,2),(0,0), '-k', label='z-axis')
    ax.plot((-2,2),(0,0),(0,0),  '-k', label='z-axis')

    ax.grid(False)

    # Hide axes ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    plt.show()
    plt.close()
    return

def PlotMainExperimentResults(vecExperiments: list[list[str]] = None, iRef: int = -1, bDataAug: bool = False) -> None:
    vecDatasets, vecModelPairs, vecProjectors, _, vecConfigs = GenerateMainExperimentParams()

    vecProjectors.append("DotProd")
    vecModelPairs.remove({"Teacher": "ViT_B", "Student": "ViT_ETT"})

    T = KDTrainer("../KDTrainer", GenerateBaseConfig(), bStartLog = False)

    fBarWidth = 1
    fBarSpacing = 0.1
    fBarGroupSpacing = 0.5
    iBorderWidth = 1
    iBarClip = 25
    vecMinXTicks = [11, 19, 22]
    vecLabelPeriods = [3, 3, 3]
    iBarClipNeg = 0
    iStdClip = 2
    fARICutoff = 0.5


    dExperimentPlotParams = {
        "VKD": {"color": "dimgray", "alpha": 0.8},
        "VKD+240E": {"color": "dimgray", "alpha": 0.6, "hatch": "/"},
        "VKD-T2": {"color": "dimgrey", "alpha": 0.6, "hatch": "x"},
        "VKDNormalized": {"color": "darkgrey", "alpha": 0.8},
        "VKDNormalized+240E": {"color": "darkgrey", "alpha": 0.6, "hatch": "/"},

        # "Baseline1LCFC+VKD": {"color": "deepskyblue", "alpha": 0.8},
        # "Baseline3LC+VKD": {"color": "royalblue", "alpha": 0.8},
        # "Baseline3LCFC+VKD": {"color": "deepskyblue", "alpha": 0.8},
        # "OursMultiSL": {"color": "orangered", "alpha": 0.8},
        # "OursMultiFC": {"color": "red", "alpha": 0.8},
        # "OursSingle": {"color": "orangered", "alpha": 0.8},

        "OursMulti": {"color": "orangered", "alpha": 0.8,},

        "Baseline1LC+VKD": {"color": "royalblue", "alpha": 0.8},
        #"Baseline1LC+VKD": {"color": "royalblue", "alpha": 0.6, "hatch": "/"},
        "Baseline1LC+VKD+240E": {"color": "royalblue", "alpha": 0.6, "hatch": "/"},
        
        "Baseline1LCFC+VKD": {"color": "dodgerblue", "alpha": 0.8},
        "Baseline1LCFC+VKD+240E": {"color": "dodgerblue", "alpha": 0.6, "hatch": "/"},

        "BaselineSP+VKD": {"color": "cornflowerblue", "alpha": 0.8},
        "BaselineSP+VKD+240E": {"color": "cornflowerblue", "alpha": 0.6, "hatch": "/"},

        "SimKD": {"color": "skyblue", "alpha": 0.8},
        "SimKD+240E": {"color": "skyblue", "alpha": 0.6, "hatch": "/"},
        "SemCKD": {"color": "cyan", "alpha": 0.8},
        "SemCKD+240E": {"color": "cyan", "alpha": 0.6, "hatch": "/"},

        "BaselineFC+VKD+SP": {"color": "slategrey", "alpha": 0.8},

        "Baseline1LC": {"color": "royalblue", "alpha": 0.4, "hatch": "/"},
        "STL+SepLin": {"color": "orangered", "alpha": 0.4, "hatch": "/"},
        "OursMulti+CE+VKD": {"color": "royalblue", "alpha": 0.8},
        "OursMulti+CE": {"color": "deepskyblue", "alpha": 0.8},
        
        "Ours-S": {"color": "indigo", "alpha": 0.8},
        "Ours-I": {"color": "royalblue", "alpha": 0.8},
        "Ours-E": {"color": "deepskyblue", "alpha": 0.8},
        "Ours-SqrtIE": {"color": "lightsteelblue", "alpha": 0.8}
    }

    def SetBarParams(pltB, strN) -> None:
        pltB.set_edgecolor("black")
        pltB.set_linewidth(iBorderWidth)
        
        if strN not in dExperimentPlotParams.keys(): return

        if "color" in dExperimentPlotParams[strN].keys(): pltB.set_color(dExperimentPlotParams[strN]["color"])
        if "alpha" in dExperimentPlotParams[strN].keys(): pltB.set_alpha(dExperimentPlotParams[strN]["alpha"])
        if "hatch" in dExperimentPlotParams[strN].keys(): pltB.set_hatch(dExperimentPlotParams[strN]["hatch"])
        pltB.set_edgecolor("black")

        return
    
    def GetDisplayName(strN) -> str:
        if "240E" in strN: return ""
        #if "+240E" in strN: strN = strN[:-5]
        if strN not in dDisplayNames.keys():
            return strN
        else:
            strN = dDisplayNames[strN]
        return strN
    
    def CheckResults(dR: dict) -> tuple[float, float]:
        vecR = []
        for i in range(dR["NumRuns"]):
            vecR.append(max([dR["Runs"][i]["TestMetrics"][j][1] for j in range(dR["NumEpochs"])]))
        iL = len(vecR)
        nR = np.array(vecR)
        #Sometimes, training with OneCycle can result in un-representatively poor outliers for baseline methods.
        #In these cases, throw out the outliers to ensure we give the baselines the best chance to compare favorably.
        vecIdx = []
        for i in range(dR["NumRuns"]):
            if nR[i] < np.max(nR) - 0.01: continue
            vecIdx.append(i)
        #print(len(vecIdx))
        if len(vecIdx) < 2 and iL >= 2:
            print("Warning! Many outliers detected!")
        return np.mean(nR[vecIdx]) * 100, np.std(nR[vecIdx]) * 100

    _, axes = plt.subplots(len(vecModelPairs), len(vecDatasets))

    for idxData in range(len(vecDatasets)):
        #do whatever we need to do at the dataset level
        strData = vecDatasets[idxData]
        axes[0, idxData].set_title(dDisplayNames[strData] if strData in dDisplayNames.keys() else strData, fontsize = 24) #Print dataset name on the top of each column
        axes[-1, idxData].set_xlabel("Top-1 Accuracy (%)", fontsize = 20) #print x-axis label at the bottom

        for idxModelPair in range(len(vecModelPairs)):
            dModelPair = vecModelPairs[idxModelPair]
            #do whatever we need to do at the model pair level
            fBarOffset = 0
            vecYTicks = []
            vecYTickLabels = []
            ax = axes[idxModelPair, idxData]
            ax.invert_yaxis()
            
            # if idxData == 0:
            #     vecYTicks.append(-1*fBarWidth)
            #     vecYTickLabels.append(dModelPair["Teacher"] + "->" + dModelPair["Student"])

            fTeacherResult = T.GetResult(GenMainExperimentBaselinePath(strData, dModelPair["Teacher"], bDataAug))["MaxTestAcc"] * 100
            fStudentResult, _ = CheckResults(T.GetResult(GenMainExperimentBaselinePath(strData, dModelPair["Student"], bDataAug)))
            ax.set_xticks([x for x in range(-iBarClip, iBarClip + 1)])
            fOffset = fStudentResult - 1


            def RemoveOffset(fR):
                fP = fR - fOffset
                if fP > iBarClip: fP = iBarClip
                if fP < iBarClipNeg: fP = iBarClipNeg
                return fP

            
            for vecExp in vecExperiments:
                fMax = 0
                fARI = 0
                vecFBARI = []
                for strName in vecExp:
                    for dCfg in vecConfigs:
                        if dCfg["Name"] != strName: continue

                        if dCfg["Params"].get("FeatureKD"):
                            if dCfg["Params"].get("ProjectionMethod") == "LearnedProjector":
                                strProj = dCfg["Params"].get("ProjectorArchitecture")
                            elif dCfg["Params"].get("ProjectionMethod") == "RelationFunction":
                                strProj = dCfg["Params"].get("RelationFunction")
                        else:
                            strProj = ""

                        try:
                            print(dCfg["Name"])
                            fR, fStd = CheckResults(T.GetResult(GenMainExperimentPath(strData, dModelPair, strProj, dCfg["Name"], bDataAug)))
                            fPlot = RemoveOffset(fR)
                            if fPlot == iBarClipNeg: fStd = 0
                            #elif fStd < 0.15: fStd = 0.15 #overly low std could be accidentally too good
                        except: continue

                        fErr = fStd
                        #print(fErr)
                        if fPlot > fMax: fMax = fPlot
                        pltBar = ax.barh([fBarOffset], [fPlot], fBarWidth, xerr = fErr if fErr <= iStdClip else iStdClip,
                                         ecolor = "black", capsize = 4)[0]
                        SetBarParams(pltBar, dCfg["Name"])

                        if iRef > -1:
                            if vecExp.index(strName) != iRef:
                                fari = fR - fStudentResult
                                if fari >= fARICutoff: vecFBARI.append(fari)
                            else:
                                fARI = fR - fStudentResult
                                pltRefBar = pltBar
                        
                        vecYTicks.append(fBarOffset)
                        vecYTickLabels.append(GetDisplayName(dCfg["Name"]))

                        fBarOffset += fBarWidth + fBarSpacing

                fBarOffset += fBarGroupSpacing

                #print ARI
                if iRef > -1:
                    print(strData, dModelPair)
                    dARI = np.mean(np.array([fARI / fBARI for fBARI in vecFBARI]))
                    print(dARI)
                    print("-----------------------")
                    strARI = "ARI: {:0.0f}%".format(dARI * 100) if (dARI < 15 and dARI > 0) else "Unstable"
                    strTeacherPerf = "Teacher: {:0.2f}%".format(fTeacherResult)
                    #ax.text(0.1, pltRefBar.get_y() + fBarWidth / 2 + 0.1, strARI, color = "black", ha = "left", va = "center")

                    ax.text(0.825 * max([fMax, vecMinXTicks[idxData]]), pltRefBar.get_y() + 7*(fBarWidth / 2) + 0.1, strARI + "\n" + strTeacherPerf,
                            size=16, rotation=0.,
                            ha="center", va="center",
                            bbox=dict(boxstyle="round",
                                ec = "black",
                                fc = "lightgrey",
                                )
                            )

            #plot the student vertical line
            ax.plot([1 for _ in range(10)], [((fBarOffset / 9) * i) - 0.5*fBarWidth for i in range(10)], color = "black", linewidth = 2)
            
            #teacher vertical line
            # fPlot = fTeacherResult - fOffset
            # if fPlot > fMax: fMax = fPlot
            # if fPlot > iBarClip: fPlot = iBarClip
            # if fPlot < iBarClipNeg: fPlot = iBarClipNeg
            # ax.plot([fPlot for _ in range(10)], [((fBarOffset / 9) * i) - 0.5*fBarWidth  for i in range(10)], 
            #         color = "black", linewidth = 2, linestyle = "dashed")
            
            ax.set_yticks(vecYTicks)
            if idxData == 0:
                #print(vecYTickLabels, vecYTicks)
                ax.set_yticklabels(vecYTickLabels, rotation = 15, fontsize = 16)
            else:
                ax.set_yticklabels([])

            if idxData == len(vecDatasets) - 1:
                ax.set_ylabel(dDisplayNames[dModelPair["Teacher"]] + "->" + dDisplayNames[dModelPair["Student"]], rotation = 270, fontsize = 18)
                ax.yaxis.set_label_position('right')
                ax.yaxis.labelpad = 20

            if fMax < vecMinXTicks[idxData]:
                ax.plot([vecMinXTicks[idxData]], [fBarOffset / 2], color = "gray", linewidth = 0.1)

            vecXTickLabels = []
            iLabelPeriod = vecLabelPeriods[idxData]
            for j in range(-iBarClip, iBarClip + 1):
                if j < 1:
                    strL = "-"
                elif j > 1:
                    strL = "+"
                strL += str(j - 1)# + "%"
                if j == 1:
                    strL = "{:.2f}%".format(fOffset + 1)
                elif j == iBarClip and fPlot == iBarClip:
                    #print the teacher's accuracy when it cannot be inferred
                    strL = "{:.2f}%".format(fTeacherResult)
                vecXTickLabels.append(strL if (j-1) % iLabelPeriod == 0 else "")



            ax.set_xticklabels(vecXTickLabels, fontsize = 16)
            ax.grid(True, axis = "x")

    plt.show()
    plt.close()


def CPlotAccuracyVsEpoch(strConfigDir: str = None, dFilter: dict = None, 
                         vecExtras: list[str] = None, vecNames: list[str] = None, iRef: int = -1, strTitle: str = None) -> None:
    T = KDTrainer("../KDTrainer", GenerateBaseConfig(),
                    vecIgnoredFields = ["SaveResult", "SaveModel",
                                        "Plot", "ResultSavePath",
                                        "ModelSavePath", "EvalInterval",
                                        "CheckpointInterval", "StoreFeatures"], bStartLog = False)
    vecR = []
    vecN = []
    if strConfigDir is not None:
        for strDir, _, vecF in os.walk(strConfigDir):
            for strF in vecF:
                if ".json" not in strF: continue
                print(strDir, strF)
                strPath = strDir + "/" + strF
                with open(strPath, "r") as f:
                    dCfg = json.load(f)
                if dFilter is not None and not IsDictSubset(dFilter, dCfg): continue
                vecN.append(Get4ParamConfigString(dCfg))
                dR = T.GetResult(dCfg)
                if dR: vecR.append(dR)

    if vecExtras is not None:
        for strPath in vecExtras:
            with open(strPath, "r") as f:
                dCfg = json.load(f)
            if dFilter is not None and not IsDictSubset(dFilter, dCfg): continue
            #vecN.append(Get4ParamConfigString(dCfg))
            dR = T.GetResult(dCfg)
            if dR: vecR.append(dR)

    return IPlotAccuracyVsEpoch(vecR, vecNames if vecNames is not None else vecN, iRef = iRef, strTitle = strTitle)

def IPlotAccuracyVsEpoch(vecR: list[dict], vecNames: list[str] = None, iRef: int = -1, 
                         vecColors: list[str] = ["black", "deepskyblue", "royalblue", "indigo", "lightsteelblue", "orangered"], strTitle: str = None) -> None:
    vecP = []
    fRef = -1
    for r in range(len(vecR)):
        dRes = vecR[r]
        #print(dRes)
        #print(dRes["NumRuns"])
        nY = np.zeros((dRes["NumRuns"], dRes["NumEpochs"]))
        for i in range(len(dRes["Runs"])):
            nY[i,:] = np.array([100 * dRes["Runs"][i]["TestMetrics"][j][1] for j in range(dRes["NumEpochs"])])
        nY = np.max(nY, axis = 0)
        #nX = 100 * (np.arange(0, nY.shape[0]) / (nY.shape[0] - 1))
        nX = np.arange(0, nY.shape[0]) + 1
        vecP.append((nX, nY))
        if iRef == r:
            fRef = np.max(nY)
            print(fRef)

    vecStyles = ["dashed", "dotted", "dashdot"]
    vecPlotted = []
    if fRef > -1:
        for i in range(len(vecP)):
            nDelta = vecP[i][1] - fRef
            iX = np.where(nDelta >= 0)[0]
            iX = iX[0] if iX.shape[0] > 0 else vecP[i][1].shape[0] - 1
            iX += 1
            print(iX)
            
            if i != iRef: plt.scatter([iX], [fRef], color = "black", marker = "x", s = 300)

            if i != iRef:
                plt.plot([iX for _ in range(11)], [j * fRef / 10 for j in range(11)], color = vecColors[i % len(vecColors)], linewidth = 3, linestyle = 
                         vecStyles[iX in vecPlotted])
            vecPlotted.append(iX)
            plt.plot([iX + (j * (nX[-1] - iX) / 10) for j in range(11)], [fRef for _ in range(11)], color = "black", linewidth = 2, linestyle = "solid")

    for i in range(len(vecP)):
        plt.plot(vecP[i][0], vecP[i][1], label = vecNames[i % len(vecNames)] if vecNames is not None else "???", color = vecColors[i % len(vecColors)], 
                 linewidth = 2.5,
                 linestyle = "solid")

    

    plt.xticks(np.arange(0, 51, 5), fontsize = 18)
    plt.tick_params(axis='y', which='both', labelleft='off', labelright='on')
    plt.yticks(np.arange(0, 100, 5), fontsize = 18)
    plt.grid()

    plt.ylabel("Student Accuracy (Top-1 %)", fontsize = 24)
    plt.xlabel("Epoch", fontsize = 24)
    if strTitle is not None: plt.title(strTitle, fontsize = 32)
    plt.legend(fontsize = 24, loc = "lower right")
    plt.show()
    plt.close()

def CPlotTeacherLayerSelection(strDataset: str, vecTeachers: list[str] = ["VGG19", "ResNet34", "ViT_B"]) -> None:
    _, _, _, dLayerSelections, _ = GenerateMainExperimentParams()

    vecFQColors = ["dimgrey", "dodgerblue", "mediumaquamarine"]
    vecPlots = []
    for i in range(len(vecTeachers)):
        strT = vecTeachers[i]
        with open(GenMainExperimentBaselinePath(strDataset, strT), "r") as f:
            dCfg = json.load(f)

        T = KDTrainer("../KDTrainer", dCfg, bStartLog = False)
        
        dFQ = TPlotClassificationEfficiency(T, bNormalize = False)

        X = [i for i in range(T.tModel.iGenLayers)]
        plt.plot(X, dFQ["Separation"], color = "midnightblue", linewidth = 7, linestyle = "solid", label = "Separation", zorder = 0)
        plt.plot(X, dFQ["Information2"], color = "cyan", linewidth = 7, linestyle = "solid", label = "Information", zorder = 0)
        plt.plot(X, dFQ["Efficiency"], color = "red", linewidth = 7, linestyle = "solid", label = "Efficiency", zorder = 0)
        plt.plot(X, dFQ["Quality2"], color = "dimgrey", linewidth = 7, linestyle = "solid", label = "Knowledge Quality", zorder = 0)

        vecBL = dLayerSelections["Baseline"][strT]
        if type(dLayerSelections["OursMulti"][strT]) == list:
            vecOurs = dLayerSelections["OursMulti"][strT]
        else:
            vecOurs = dLayerSelections["OursMulti"][strT][strDataset]
        print(vecBL, vecOurs)

        plt.scatter(vecBL, dFQ["Quality2"][vecBL], marker = "o", s = 1000, color = "orange", zorder = 1)
        plt.scatter(vecOurs, dFQ["Quality2"][vecOurs], marker = "x", s = 1200, color = "black", zorder = 1)

        #plt.legend(fontsize = 34)
        plt.title(strT.replace("_", " ") + " Knowledge Quality on CIFAR 100", fontsize = 48)
        plt.xlabel("Layer Index", fontsize=42)
        plt.ylabel("Metrics", fontsize=42)
        plt.xticks(np.arange(0, 2*(max(X) // 2) + 2, 2), fontsize = 42)
        #plt.yticks(np.linspace(0, fMaY, 10) if not bNormalize else [i/10 for i in range(11)], fontsize = 16)
        plt.yticks(np.arange(0, 2.25, 0.25), fontsize = 42)
        #plt.grid(color="black", linestyle="dashed", linewidth = 1)
        plt.grid(zorder = 0)
        
        plt.show()
        plt.close() 

        nPlot = dFQ["Quality2"]
        vecPlots.append(nPlot)

    for i in range(len(vecTeachers)):
        nPlot = vecPlots[i]
        strT = vecTeachers[i]
        L = [l / nPlot.shape[0] for l in range(nPlot.shape[0])]

        plt.plot(L, nPlot, label = strT, color = vecFQColors[i % len(vecFQColors)], linewidth = 3, zorder = 0)

        vecBL = dLayerSelections["Baseline"][strT]
        if type(dLayerSelections["OursMulti"][strT]) == list:
            vecOurs = dLayerSelections["OursMulti"][strT]
        else:
            vecOurs = dLayerSelections["OursMulti"][strT][strDataset]
        print(vecBL, vecOurs)
        #plt.scatter([l / nPlot.shape[0] for l in vecBL], nPlot[vecBL], marker = "o", s = 130, color = "black")
        #plt.scatter([l / nPlot.shape[0] for l in vecOurs], nPlot[vecOurs], marker = "x", s = 150, color = "black", zorder = 1)

    plt.xticks(np.arange(0, 1.1, 0.1), fontsize = 28)
    plt.yticks(np.arange(0, 1.6, 0.1), fontsize = 28)
    plt.xlabel("Relative Depth", fontsize = 32)
    plt.ylabel("Knowledge Quality", fontsize = 32)
    plt.legend(fontsize = 32)
    strD = dDisplayNames[strDataset] if strDataset in dDisplayNames.keys() else strDataset
    plt.title("Knowledge Quality of Teacher Models on " + strD, fontsize = 36)
    plt.grid()

    plt.show()
    plt.close()

def MainFig():
    # PlotMainExperimentResults(vecExperiments = [
    #     ["OursMulti", "VKD", "VKD-T2", "Baseline1LC+VKD", "BaselineSP+VKD", "SemCKD",], 
    #     ["OursMulti", "OursMulti+CE+VKD", "OursMulti+CE", "STL+SepLin", "Baseline1LC+VKD", "Baseline1LC"],
    #     ["OursMulti", "Ours-S", "Ours-I", "Ours-E", "Ours-SqrtIE"],
    #     ], iRef = 0)

    PlotMainExperimentResults(vecExperiments = [
        ["OursMulti", "VKD", "VKDNormalized", "Baseline1LC+VKD", "Baseline1LCFC+VKD", "BaselineSP+VKD", "SimKD", "SemCKD"], 
        ], iRef = 0, bDataAug = False)

    # PlotMainExperimentResults(vecExperiments = [
    #     ["OursMulti", "STL+SepLin"], ["OursMulti+CE", "Baseline1LC"], ["OursMulti+CE+VKD", "Baseline1LC+VKD"], 
    #     ])

    # PlotMainExperimentResults(vecExperiments = [
    #     ["OursMulti", "Ours-S", "Ours-I", "Ours-E", "Ours-SqrtIE"], 
    #     ], iRef = 0, bDataAug = False)

    # PlotMainExperimentResults(vecExperiments = [
    #     ["OursMulti"], ["VKD", "VKD+240E", "VKD-T2"], ["VKDNormalized", "VKDNormalized+240E"],
    #     ["Baseline1LC+VKD", "Baseline1LC+VKD+240E"], ["Baseline1LCFC+VKD", "Baseline1LCFC+VKD+240E"],
    #     ["BaselineSP+VKD", "BaselineSP+VKD+240E"], ["SemCKD", "SemCKD+240E"], ["SimKD", "SimKD+240E"],
    #     ], bDataAug = False)
    
    # PlotMainExperimentResults([
    #     ["OursMulti", "VKD+240E", "VKDNormalized+240E", "Baseline1LC+VKD+240E", "Baseline1LCFC+VKD+240E", "BaselineSP+VKD+240E", "SimKD+240E", "SemCKD+240E"]
    # ], iRef = 0, bDataAug = False)

def TrainingCurves():
    strStudent = "ViT_ETT"
    strTeacher = "ViT_B"
    strData = "CIFAR10"
    strDS = dDisplayNames[strData] if strData in dDisplayNames.keys() else strData
    CPlotAccuracyVsEpoch(None, vecExtras = [
        "../ExperimentConfigs/MainExperiments/" + strData + "/Baselines/" + strStudent + "_Baseline.json",
        "../ExperimentConfigs/MainExperiments/" + strData + "/" + strTeacher + "->" + strStudent + "/VKD.json",
        "../ExperimentConfigs/MainExperiments/" + strData + "/" + strTeacher + "->" + strStudent + "/SingleLayerConv/Baseline1LC+VKD.json",
        "../ExperimentConfigs/MainExperiments/" + strData + "/" + strTeacher + "->" + strStudent + "/DotProd/BaselineSP+VKD.json",
        "../ExperimentConfigs/MainExperiments/" + strData + "/" + strTeacher + "->" + strStudent + "/ThreeLayerConv/SemCKD.json",
        "../ExperimentConfigs/MainExperiments/" + strData + "/" + strTeacher + "->" + strStudent + "/SingleLayerConv/OursMulti.json",],
        vecNames = ["Baseline Student", "Vanilla KD", "Base FKD", "Similarity Preserving", "SemCKD", "Ours"], iRef = 0, 
        strTitle = strDS + ": " + strTeacher + " -> " + strStudent)


        #     "../ExperimentConfigs/MainExperiments/CIFAR100F/Baselines/ResNet9_Baseline.json",
        # "../ExperimentConfigs/MainExperiments/CIFAR100F/ResNet34->ResNet9/VKD.json",
        # "../ExperimentConfigs/MainExperiments/CIFAR100F/ResNet34->ResNet9/ThreeLayerConv/SemCKD.json",
        # "../ExperimentConfigs/MainExperiments/CIFAR100F/ResNet34->ResNet9/SingleLayerConv/OursMulti.json",

def FQ():
    CPlotTeacherLayerSelection("CIFAR100F", vecTeachers = ["ResNet34", "ViT_B"])

if __name__ == "__main__":
    #FQ()
    #TrainingCurves()
    MainFig()
    #PlotDimensionExample()