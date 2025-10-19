'''
Code written by Nicholas J. Cooper.
Released under the MIT license, see the GitHub page for the full legal boilerplate.
tldr: you freely can do whatever you like with this code, but please cite the GitHub at: https://github.com/Thegolfingocto/KDwoCE.git
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
'''


import torch

#from FishModels import *
#from ConeModels import *

from Arch.Analysis.MathUtils import *
from Arch.Analysis.Analysis import *

from Arch.Analysis.IAnalyzers import *
from Arch.Analysis.TDA import *
from Arch.Utils import *


class PCAStats(IStat):
    def __init__(self):
        super().__init__("PCAIntrinsicDimension", "PCA Intrinsic Dimension", strFormat = ".json")

    def Compute(self, tData: torch.tensor, tSampleIdx: torch.tensor = None, **kwargs) -> dict:
        assert tSampleIdx is not None, "PCAStats Requires Sample Indices!"

        tF = tData
        f = tF[tSampleIdx,...]
        return IComputePCAID(f)
    
    def Plot(self, vecData: list[object]) -> list[mplPlot]:
        vecPlots = []
        vecPlots.append(mplPlot([d["PCAID"] / 5000 for d in vecData], "PCA Intrinsic Dimension", "red", 3, "solid"))
        vecPlots.append(mplPlot([d["PCAID"] / d["HostDim"] for d in vecData], "PCA Intrinsic Dimension (Normalized by AD)", "orangered", 3, "solid"))

        #vecPlots.append(mplPlot([d["HostDim"] for d in vecData], "Ambient Dimension", "black", 3, "solid"))

        return vecPlots
    

class Conv2PCAStats(IStat):
    def __init__(self):
        super().__init__("Conv2PCAIntrinsicDimension", "Conv2 PCA Intrinsic Dimension", strFormat = ".json")

    def Compute(self, tData: torch.tensor, tSampleIdx: torch.tensor = None, **kwargs) -> dict:
        assert tSampleIdx is not None, "Conv2PCAStats Requires Sample Indices!"

        tF = tData
        f = tF[tSampleIdx,...]

        if len(f.shape) == 4:
            vecED = []
            for _ in range(5):
                tConv2 = torch.nn.Conv2d(f.shape[1], f.shape[1], kernel_size = 3, padding = 1, bias = False)
                #torch.nn.init.kaiming_normal_(tConv2.weight, mode="fan_out", nonlinearity="relu")

                fVar = np.sqrt(12 / (sum(list(tConv2.weight.shape))))
                tW = fVar * (2*torch.rand_like(tConv2.weight) - 1)
                #print(tW.shape)
                #tW /= torch.norm(tW.view(tW.shape[0], -1), dim = 1, keepdim = True).unsqueeze(-1).unsqueeze(-1)
                #print(tW.shape)
                tConv2.weight = torch.nn.Parameter(tW)

                tConv2 = tConv2.to(device)
                with torch.no_grad():
                    x = tConv2(f.to(device))
                
                dED = IComputePCAID(x)
                vecED.append(dED["PCAID"])
                iAD = dED["HostDim"]

            nED = np.array(vecED)
            return {"AvgPCAID": np.mean(nED),
                    "StdPCAID": np.std(nED),
                    "HostDim": iAD,
                    }

        else:
            dED = IComputePCAID(f)
            return {"AvgPCAID": dED["PCAID"],
                    "StdPCAID": 0,
                    "HostDim": dED["HostDim"],
                    }
    
    def Plot(self, vecData: list[object]) -> list[mplPlot]:
        vecPlots = []
        vecPlots.append(mplPlot([d["AvgPCAID"] / 5000 for d in vecData], "Conv2 PCA Intrinsic Dimension", "firebrick", 3, "solid"))
        vecPlots.append(mplPlot([(d["AvgPCAID"] + d["StdPCAID"]) / 5000 for d in vecData], "", "firebrick", 3, "dotted"))
        vecPlots.append(mplPlot([(d["AvgPCAID"] - d["StdPCAID"]) / 5000 for d in vecData], "", "firebrick", 3, "dotted"))

        #vecPlots.append(mplPlot([d["HostDim"] for d in vecData], "Ambient Dimension", "black", 3, "solid"))

        return vecPlots


class RandConv2Stats(IStat):
    def __init__(self):
        super().__init__("RandConv2Stats", "Rand Conv2 Stats", strFormat = ".json")

    def Compute(self, tData: torch.tensor, tSampleIdx: torch.tensor = None, YA: torch.tensor = None, **kwargs) -> dict:
        assert tSampleIdx is not None, "Conv2PCAStats Requires Sample Indices and Labels!"

        tF = tData
        f = tF[tSampleIdx,...]

        if len(f.shape) == 4:
            vecED = []
            vecSMI = []
            for _ in range(5):
                tConv2 = torch.nn.Conv2d(f.shape[1], f.shape[1], kernel_size = 3, padding = 1, bias = False)
                #torch.nn.init.kaiming_normal_(tConv2.weight, mode="fan_out", nonlinearity="relu")

                fVar = np.sqrt(6 / (sum(list(tConv2.weight.shape))))
                tW = fVar * (2*torch.rand_like(tConv2.weight) - 1)
                #print(tW.shape)
                #tW /= torch.norm(tW.view(tW.shape[0], -1), dim = 1, keepdim = True).unsqueeze(-1).unsqueeze(-1)
                #print(tW.shape)
                tConv2.weight = torch.nn.Parameter(tW)

                tConv2 = tConv2.to(device)
                with torch.no_grad():
                    x = tConv2(f.to(device))
                
                dED = IComputePCAID(x)
                vecED.append(dED["PCAID"])
                iAD = dED["HostDim"]

                dSMI = IComputeSlicedMI(x, YA[tSampleIdx,...])
                vecSMI.append(dSMI["SMI"])

            nED = np.array(vecED)
            nSMI = np.array(vecSMI)
            return {"AvgPCAID": np.mean(nED),
                    "StdPCAID": np.std(nED),

                    "AvgSMI": np.mean(nSMI),
                    "StdSMI": np.std(nSMI),

                    "HostDim": iAD,
                    }

        else:
            dED = IComputePCAID(f)
            dSMI = IComputeSlicedMI(f, YA[tSampleIdx,...])
            return {"AvgPCAID": dED["PCAID"],
                    "StdPCAID": 0,

                    "AvgSMI": dSMI["SMI"],
                    "StdSMI": 0,

                    "HostDim": dED["HostDim"],
                    }
    
    def Plot(self, vecData: list[object]) -> list[mplPlot]:
        vecPlots = []
        vecPlots.append(mplPlot([d["AvgPCAID"] / 5000 for d in vecData], "Conv2 PCA Intrinsic Dimension", "firebrick", 3, "solid"))
        vecPlots.append(mplPlot([(d["AvgPCAID"] + d["StdPCAID"]) / 5000 for d in vecData], "", "firebrick", 3, "dotted"))
        vecPlots.append(mplPlot([(d["AvgPCAID"] - d["StdPCAID"]) / 5000 for d in vecData], "", "firebrick", 3, "dotted"))

        vecPlots.append(mplPlot([d["AvgSMI"] for d in vecData], "Conv2 SMI", "dodgerblue", 3, "solid"))
        vecPlots.append(mplPlot([(d["AvgSMI"] + d["StdSMI"]) for d in vecData], "", "dodgerblue", 3, "dotted"))
        vecPlots.append(mplPlot([(d["AvgSMI"] - d["StdSMI"]) for d in vecData], "", "dodgerblue", 3, "dotted"))

        #vecPlots.append(mplPlot([d["HostDim"] for d in vecData], "Ambient Dimension", "black", 3, "solid"))

        return vecPlots

'''
class FishPCAStats(IStat):
    def __init__(self):
        super().__init__("FishPCAIntrinsicDimension", "Fish PCA Intrinsic Dimension", strFormat = ".json")

    def Compute(self, tData: torch.tensor, tSampleIdx: torch.tensor = None, **kwargs) -> dict:
        assert tSampleIdx is not None, "FishPCAStats Requires Sample Indices!"

        tF = tData
        f = tF[tSampleIdx,...].to(device)
        x = torch.zeros_like(f).to(device)

        iTBS = 1000

        if len(f.shape) == 4:
            vecED = []
            vecS = list(f.shape)[1:]
            for _ in range(5):
                tFish = MultiFishLayer(vecS, vecS, strInit = "uniform")

                tFish = tFish.to(device)
                with torch.no_grad():
                    for i in range(f.shape[0] // iTBS):
                        x[i*iTBS:(i+1)*iTBS,...] = tFish(f[i*iTBS:(i+1)*iTBS,...])

                dED = IComputePCAID(x)
                vecED.append(dED["PCAID"])
                iAD = dED["HostDim"]

            nED = np.array(vecED)
            return {"AvgPCAID": np.mean(nED),
                    "StdPCAID": np.std(nED),
                    "HostDim": iAD,
                    }

        else:
            dED = IComputePCAID(f)
            return {"AvgPCAID": dED["PCAID"],
                    "StdPCAID": 0,
                    "HostDim": dED["HostDim"],
                    }
    
    def Plot(self, vecData: list[object]) -> list[mplPlot]:
        vecPlots = []
        vecPlots.append(mplPlot([d["AvgPCAID"] / 5000 for d in vecData], "Multi-Fish PCA Intrinsic Dimension", "orange", 3, "solid"))
        vecPlots.append(mplPlot([(d["AvgPCAID"] + d["StdPCAID"]) / 5000 for d in vecData], "", "orange", 3, "dotted"))
        vecPlots.append(mplPlot([(d["AvgPCAID"] - d["StdPCAID"]) / 5000 for d in vecData], "", "orange", 3, "dotted"))

        return vecPlots
    

class RandFishStats(IStat):
    def __init__(self):
        super().__init__("RandFishStats", "Rand Fish Stats", strFormat = ".json")

    def Compute(self, tData: torch.tensor, tSampleIdx: torch.tensor = None, YA: torch.tensor = None, **kwargs) -> dict:
        assert tSampleIdx is not None, "FishPCAStats Requires Sample Indices and Labels!"

        tF = tData
        f = tF[tSampleIdx,...].to(device)
        x = torch.zeros_like(f).to(device)

        iTBS = 1000

        if len(f.shape) == 4:
            vecED = []
            vecSMI = []
            vecS = list(f.shape)[1:]
            for _ in range(5):
                tFish = MultiFishLayer(vecS, vecS, strInit = "uniform")

                tFish = tFish.to(device)
                with torch.no_grad():
                    for i in range(f.shape[0] // iTBS):
                        x[i*iTBS:(i+1)*iTBS,...] = tFish(f[i*iTBS:(i+1)*iTBS,...])

                dED = IComputePCAID(x)
                vecED.append(dED["PCAID"])
                iAD = dED["HostDim"]

                dSMI = IComputeSlicedMI(x, YA[tSampleIdx,...])
                vecSMI.append(dSMI["SMI"])

            nED = np.array(vecED)
            nSMI = np.array(vecSMI)
            return {"AvgPCAID": np.mean(nED),
                    "StdPCAID": np.std(nED),

                    "AvgSMI": np.mean(nSMI),
                    "StdSMI": np.std(nSMI),

                    "HostDim": iAD,
                    }

        else:
            dED = IComputePCAID(f)
            dSMI = IComputeSlicedMI(f, YA[tSampleIdx,...])
            return {"AvgPCAID": dED["PCAID"],
                    "StdPCAID": 0,

                    "AvgSMI": dSMI["SMI"],
                    "StdSMI": 0,

                    "HostDim": dED["HostDim"],
                    }
    
    def Plot(self, vecData: list[object]) -> list[mplPlot]:
        vecPlots = []
        vecPlots.append(mplPlot([d["AvgPCAID"] / 5000 for d in vecData], "Fish PCA Intrinsic Dimension", "orange", 3, "solid"))
        vecPlots.append(mplPlot([(d["AvgPCAID"] + d["StdPCAID"]) / 5000 for d in vecData], "", "orange", 3, "dotted"))
        vecPlots.append(mplPlot([(d["AvgPCAID"] - d["StdPCAID"]) / 5000 for d in vecData], "", "orange", 3, "dotted"))

        vecPlots.append(mplPlot([d["AvgSMI"] for d in vecData], "Fish SMI", "steelblue", 3, "solid"))
        vecPlots.append(mplPlot([(d["AvgSMI"] + d["StdSMI"]) for d in vecData], "", "steelblue", 3, "dotted"))
        vecPlots.append(mplPlot([(d["AvgSMI"] - d["StdSMI"]) for d in vecData], "", "steelblue", 3, "dotted"))

        #vecPlots.append(mplPlot([d["HostDim"] for d in vecData], "Ambient Dimension", "black", 3, "solid"))

        return vecPlots
    

class DualFishPCAStats(IStat):
    def __init__(self):
        super().__init__("DualFishPCAIntrinsicDimension", "Dual Fish PCA Intrinsic Dimension", strFormat = ".json")

    def Compute(self, tData: torch.tensor, tSampleIdx: torch.tensor = None, **kwargs) -> dict:
        assert tSampleIdx is not None, "FishPCAStats Requires Sample Indices!"

        tF = tData
        f = tF[tSampleIdx,...].to(device)
        x = torch.zeros_like(f).to(device)

        iTBS = 1000

        if len(f.shape) == 4:
            vecED = []
            vecS = list(f.shape)[1:]
            for _ in range(5):
                tFish = FishLayerType1Dual(vecS)

                tFish = tFish.to(device)
                with torch.no_grad():
                    for i in range(f.shape[0] // iTBS):
                        x[i*iTBS:(i+1)*iTBS,...] = tFish(f[i*iTBS:(i+1)*iTBS,...])

                dED = IComputePCAID(x)
                vecED.append(dED["PCAID"])
                iAD = dED["HostDim"]

            nED = np.array(vecED)
            return {"AvgPCAID": np.mean(nED),
                    "StdPCAID": np.std(nED),
                    "HostDim": iAD,
                    }

        else:
            dED = IComputePCAID(f)
            return {"AvgPCAID": dED["PCAID"],
                    "StdPCAID": 0,
                    "HostDim": dED["HostDim"],
                    }
    
    def Plot(self, vecData: list[object]) -> list[mplPlot]:
        vecPlots = []
        vecPlots.append(mplPlot([d["AvgPCAID"] / 5000 for d in vecData], "Dual-Fish PCA Intrinsic Dimension", "orange", 3, "solid"))
        vecPlots.append(mplPlot([(d["AvgPCAID"] + d["StdPCAID"]) / 5000 for d in vecData], "", "orange", 3, "dotted"))
        vecPlots.append(mplPlot([(d["AvgPCAID"] - d["StdPCAID"]) / 5000 for d in vecData], "", "orange", 3, "dotted"))

        return vecPlots


class RandDualFishStats(IStat):
    def __init__(self):
        super().__init__("RandDualFishStats", "Rand DualFish Stats", strFormat = ".json")

    def Compute(self, tData: torch.tensor, tSampleIdx: torch.tensor = None, YA: torch.tensor = None, **kwargs) -> dict:
        assert tSampleIdx is not None, "DualFishPCAStats Requires Sample Indices and Labels!"

        tF = tData
        f = tF[tSampleIdx,...].to(device)
        x = torch.zeros_like(f).to(device)

        iTBS = 1000

        if len(f.shape) == 4:
            vecED = []
            vecSMI = []
            vecS = list(f.shape)[1:]
            for _ in range(5):
                tFish = FishLayerType1Dual(vecS)

                tFish = tFish.to(device)
                with torch.no_grad():
                    for i in range(f.shape[0] // iTBS):
                        x[i*iTBS:(i+1)*iTBS,...] = tFish(f[i*iTBS:(i+1)*iTBS,...])

                dED = IComputePCAID(x)
                vecED.append(dED["PCAID"])
                iAD = dED["HostDim"]

                dSMI = IComputeSlicedMI(x, YA[tSampleIdx,...])
                vecSMI.append(dSMI["SMI"])

            nED = np.array(vecED)
            nSMI = np.array(vecSMI)
            return {"AvgPCAID": np.mean(nED),
                    "StdPCAID": np.std(nED),

                    "AvgSMI": np.mean(nSMI),
                    "StdSMI": np.std(nSMI),

                    "HostDim": iAD,
                    }

        else:
            dED = IComputePCAID(f)
            dSMI = IComputeSlicedMI(f, YA[tSampleIdx,...])
            return {"AvgPCAID": dED["PCAID"],
                    "StdPCAID": 0,

                    "AvgSMI": dSMI["SMI"],
                    "StdSMI": 0,

                    "HostDim": dED["HostDim"],
                    }
    
    def Plot(self, vecData: list[object]) -> list[mplPlot]:
        vecPlots = []
        vecPlots.append(mplPlot([d["AvgPCAID"] / 5000 for d in vecData], "DualFish PCA Intrinsic Dimension", "indianred", 3, "solid"))
        vecPlots.append(mplPlot([(d["AvgPCAID"] + d["StdPCAID"]) / 5000 for d in vecData], "", "indianred", 3, "dotted"))
        vecPlots.append(mplPlot([(d["AvgPCAID"] - d["StdPCAID"]) / 5000 for d in vecData], "", "indianred", 3, "dotted"))

        vecPlots.append(mplPlot([d["AvgSMI"] for d in vecData], "DualFish SMI", "midnightblue", 3, "solid"))
        vecPlots.append(mplPlot([(d["AvgSMI"] + d["StdSMI"]) for d in vecData], "", "midnightblue", 3, "dotted"))
        vecPlots.append(mplPlot([(d["AvgSMI"] - d["StdSMI"]) for d in vecData], "", "midnightblue", 3, "dotted"))

        #vecPlots.append(mplPlot([d["HostDim"] for d in vecData], "Ambient Dimension", "black", 3, "solid"))

        return vecPlots


class FishConv2PCAStats(IStat):
    def __init__(self):
        super().__init__("FishConv2PCAIntrinsicDimension", "FishConv2 PCA Intrinsic Dimension", strFormat = ".json")

    def Compute(self, tData: torch.tensor, tSampleIdx: torch.tensor = None, **kwargs) -> dict:
        assert tSampleIdx is not None, "FishConv2PCAStats Requires Sample Indices!"

        tF = tData
        f = tF[tSampleIdx,...].to(device)
        x = torch.zeros_like(f).to(device)

        iTBS = 500

        if len(f.shape) == 4:
            vecED = []
            vecS = list(f.shape)[1:]
            for _ in range(5):
                tFishConv2 = FishConv2d(3, 1, vecS[0], vecS[0], iFishType = 4)

                tFishConv2 = tFishConv2.to(device)
                with torch.no_grad():
                    for i in range(f.shape[0] // iTBS):
                        x[i*iTBS:(i+1)*iTBS,...] = tFishConv2(f[i*iTBS:(i+1)*iTBS,...])

                dED = IComputePCAID(x)
                vecED.append(dED["PCAID"])
                iAD = dED["HostDim"]

            nED = np.array(vecED)
            return {"AvgPCAID": np.mean(nED),
                    "StdPCAID": np.std(nED),
                    "HostDim": iAD,
                    }

        else:
            dED = IComputePCAID(f)
            return {"AvgPCAID": dED["PCAID"],
                    "StdPCAID": 0,
                    "HostDim": dED["HostDim"],
                    }
    
    def Plot(self, vecData: list[object]) -> list[mplPlot]:
        vecPlots = []
        vecPlots.append(mplPlot([d["AvgPCAID"] / 5000 for d in vecData], "FishConv2 PCA Intrinsic Dimension", "peru", 3, "solid"))
        vecPlots.append(mplPlot([(d["AvgPCAID"] + d["StdPCAID"]) / 5000 for d in vecData], "", "peru", 3, "dotted"))
        vecPlots.append(mplPlot([(d["AvgPCAID"] - d["StdPCAID"]) / 5000 for d in vecData], "", "peru", 3, "dotted"))

        return vecPlots
    

class ConePCAStats(IStat):
    def __init__(self):
        super().__init__("ConePCAIntrinsicDimension", "Cone PCA Intrinsic Dimension", strFormat = ".json")

        self.vecNames = ["C1", "C2", "C3", "MC"]
        self.vecColors = ["midnightblue", "dodgerblue", "skyblue", "cyan"]

    def Compute(self, tData: torch.tensor, tSampleIdx: torch.tensor = None, **kwargs) -> dict:
        assert tSampleIdx is not None, "ConePCAStats Requires Sample Indices!"

        tF = tData
        f = tF[tSampleIdx,...].to(device)
        x = torch.zeros_like(f).to(device)

        iTBS = 100

        if len(f.shape) == 4:
            vecEDs = [[], [], [], []] #C1, C2, C3, CM
            vecS = list(f.shape)[1:]
            for _ in range(5):
                tC1 = ConeLayerType1(vecS, vecS[-1]).to(device)
                tC2 = ConeLayerType2(vecS, vecS[-2]).to(device)
                tC3 = ConeLayerType3(vecS, vecS[-3]).to(device)

                tCM = MultiConeLayer(vecS, vecS[-1]).to(device)

                vecCones = [tC1, tC2, tC3, tCM]

                for c in range(4):
                    with torch.no_grad():
                        for i in range(f.shape[0] // iTBS):
                            x[i*iTBS:(i+1)*iTBS,...] = vecCones[c](f[i*iTBS:(i+1)*iTBS,...])

                    dED = IComputePCAID(x)
                    vecEDs[c].append(dED["PCAID"])

                iAD = dED["HostDim"]


            dRet = {"HostDim": iAD}
            for i in range(4):
                nED = np.array(vecEDs[i])
                dRet["AvgPCAID" + self.vecNames[i]] = np.mean(nED)
                dRet["StdPCAID" + self.vecNames[i]] = np.std(nED)

            return dRet

        else:
            dED = IComputePCAID(f)
            return {"AvgPCAID": dED["PCAID"],
                    "StdPCAID": 0,
                    "HostDim": dED["HostDim"],
                    }
    
    def Plot(self, vecData: list[object]) -> list[mplPlot]:
        vecPlots = []
        if "AvgPCAIDC1" in vecData[0].keys():
            for i in range(4):
                vecPlots.append(mplPlot([(d["AvgPCAID" + self.vecNames[i]] if "AvgPCAIDC1" in d.keys() else d["AvgPCAID"]) / 5000 for d in vecData], self.vecNames[i] + " PCA Intrinsic Dimension", self.vecColors[i], 3, "solid"))
                vecPlots.append(mplPlot([((d["AvgPCAID" + self.vecNames[i]] + d["StdPCAID" + self.vecNames[i]]) if "AvgPCAIDC1" in d.keys() else (d["AvgPCAID"] + d["StdPCAID"])) / 5000 for d in vecData], "", self.vecColors[i], 3, "dotted"))
                vecPlots.append(mplPlot([((d["AvgPCAID" + self.vecNames[i]] - d["StdPCAID" + self.vecNames[i]]) if "AvgPCAIDC1" in d.keys() else (d["AvgPCAID"] - d["StdPCAID"])) / 5000 for d in vecData], "", self.vecColors[i], 3, "dotted"))
        else:
            vecPlots.append(mplPlot([d["PCAID"] / 5000 for d in vecData], "PCA Intrinsic Dimension", "red", 3, "solid"))
            vecPlots.append(mplPlot([d["PCAID"] / d["HostDim"] for d in vecData], "PCA Intrinsic Dimension (Normalized by AD)", "orangered", 3, "solid"))

        return vecPlots
'''

class SMIStats(IStat):
    def __init__(self):
        super().__init__("SlicedMutualInformation", "Sliced Mutual Information", strFormat = ".json")

    def Compute(self, tData: torch.tensor, tSampleIdx: torch.tensor = None, YA: torch.tensor = None, **kwargs) -> dict:
        assert (tSampleIdx is not None) and (YA is not None), "SMIStats Requires Sample Indices and Class Labels!"

        tF = tData
        f = tF[tSampleIdx,...]
        ya = YA[tSampleIdx,...]
        return IComputeSlicedMI(f, ya)
    
    def Plot(self, vecData: list[object]) -> list[mplPlot]:
        return [mplPlot([d["SMI"] for d in vecData], "Sliced Mutual Information", "cyan", 3, "solid")]
    
class KSMIStats(IStat):
    def __init__(self):
        super().__init__("KSlicedMutualInformation", "K-Sliced Mutual Information", strFormat = ".json")

    def Compute(self, tData: torch.tensor, tSampleIdx: torch.tensor = None, YA: torch.tensor = None, **kwargs) -> dict:
        assert (tSampleIdx is not None) and (YA is not None), "KSMIStats Requires Sample Indices and Class Labels!"

        tF = tData
        f = tF[tSampleIdx,...]
        ya = YA[tSampleIdx,...]
        return IComputeKSlicedMI(f, ya)
    
    def Plot(self, vecData: list[object]) -> list[mplPlot]:
        return [mplPlot([d["KSMI"] for d in vecData], "K-Sliced Mutual Information", "dodgerblue", 3, "solid")]
    

class Conv2SMIStats(IStat):
    def __init__(self):
        super().__init__("Conv2SlicedMutualInformation", "Conv2-Sliced Mutual Information", strFormat = ".json")

    def Compute(self, tData: torch.tensor, tSampleIdx: torch.tensor = None, YA: torch.tensor = None, **kwargs) -> dict:
        assert (tSampleIdx is not None) and (YA is not None), "KSMIStats Requires Sample Indices and Class Labels!"

        tF = tData
        f = tF[tSampleIdx,...]
        ya = YA[tSampleIdx,...]
        return IComputeConv2SlicedMI(f, ya)
    
    def Plot(self, vecData: list[object]) -> list[mplPlot]:
        return [mplPlot([d["Conv2SMI"] for d in vecData], "Conv2-Sliced Mutual Information", "midnightblue", 3, "solid")]

    

class BinaryDPStats(IStat):
    def __init__(self):
        super().__init__("BinaryDotProd", "Binary Dot Product", strFormat = ".json")

    def Compute(self, tData: torch.tensor, tSampleIdx: torch.tensor = None, **kwargs) -> dict:
        assert tSampleIdx is not None, "BinaryDPStats Requires Sample Indices and Class Labels!"

        tF = tData
        f = tF[tSampleIdx,...]
        return IComputeBinaryDPStats(f)
    
    def Plot(self, vecData: list[object]) -> list[mplPlot]:
        vecPlots = []
        vecPlots.append(mplPlot([d["AvgDPS"] for d in vecData], "Average Binary DPS", "midnightblue", 3, "solid"))
        #vecPlots.append(mplPlot([d["AvgDPS"] + d["StdDPS"] for d in vecData], "Average Binary DPS", "midnightblue", 3, "dotted"))
        #vecPlots.append(mplPlot([d["AvgDPS"] - d["StdDPS"] for d in vecData], "Average Binary DPS", "midnightblue", 3, "dotted"))
        #vecPlots.append(mplPlot([d["MinDPS"] for d in vecData], "Average Binary DPS", "midnightblue", 3, "dashed"))
        vecPlots.append(mplPlot([d["MaxDPS"] for d in vecData], "Max Binary DPS", "midnightblue", 3, "dashed"))
    
        return vecPlots


class TernaryDPStats(IStat):
    def __init__(self):
        super().__init__("TernaryDotProd", "Ternary Dot Product", strFormat = ".json")

    def Compute(self, tData: torch.tensor, tSampleIdx: torch.tensor = None, **kwargs) -> dict:
        assert tSampleIdx is not None, "TernaryDPStats Requires Sample Indices and Class Labels!"

        tF = tData
        f = tF[tSampleIdx,...]
        return IComputeTernaryDPStats(f)
    
    def Plot(self, vecData: list[object]) -> list[mplPlot]:
        vecPlots = []
        vecPlots.append(mplPlot([d["AvgDPS"] for d in vecData], "Average Ternary DPS", "skyblue", 3, "solid"))
        #vecPlots.append(mplPlot([d["AvgDPS"] + d["StdDPS"] for d in vecData], "Average Ternary DPS", "skyblue", 3, "dotted"))
        #vecPlots.append(mplPlot([d["AvgDPS"] - d["StdDPS"] for d in vecData], "Average Ternary DPS", "skyblue", 3, "dotted"))
        #vecPlots.append(mplPlot([d["MinDPS"] for d in vecData], "Average Ternary DPS", "skyblue", 3, "dashed"))
        vecPlots.append(mplPlot([d["MaxDPS"] for d in vecData], "Max Ternary DPS", "skyblue", 3, "dashed"))
    
        return vecPlots
    

class DPSCCStats(IStat):
    def __init__(self):
        super().__init__("DPSCC", "Arity Filtration", strFormat = ".json")

    def Compute(self, tData: torch.tensor, tSampleIdx: torch.tensor = None, **kwargs) -> dict:
        assert tSampleIdx is not None, "TernaryDPStats Requires Sample Indices!"

        tF = tData
        f = tF[tSampleIdx,...]
        return IComputeDPSCCStats(f, fEps = 3.0, bRandom = False)
    
    def Plot(self, vecData: list[object]) -> list[mplPlot]:
        vecPlots = []
        #vecPlots.append(mplPlot([d["AvgDepth"] for d in vecData], "Average Arity Depth", "green", 3, "solid"))
        #vecPlots.append(mplPlot([d["MaxDepth"] for d in vecData], "Max Arity Depth", "green", 3, "dashed"))
        #vecPlots.append(mplPlot([d["Width"] for d in vecData], "Width of Arity Filt.", "green", 3, "dotted"))

        #vecPlots.append(mplPlot([d["AvgDPSEntropy"] / np.log(len(d["AvgDPS"])) for d in vecData], "Avg DPS Entropy over Filt.", "lime", 3, "solid"))
        #vecPlots.append(mplPlot([d["MaxDPSEntropy"] / np.log(len(d["MaxDPS"])) for d in vecData], "Max DPS Entropy over Filt.", "lime", 3, "dashed"))

        #vecPlots.append(mplPlot([d["AbsMax"] for d in vecData], "Global L-inf", "red", 3, "dashed"))
        #vecPlots.append(mplPlot([d["AbsAvg"] for d in vecData], "Avg. Element-wise Magnitude", "red", 3, "solid"))

        vecPlots.append(mplPlot([d["AvgDPS"][0] / d["AD"] if len(d["AvgDPS"]) >= 1 else 0 for d in vecData], "Avg Binary DPS", "midnightblue", 3, "solid"))
        #vecPlots.append(mplPlot([d["AvgDPS"][0] for d in vecData], "Avg Binary DPS", "cyan", 3, "solid"))

        #vecPlots.append(mplPlot([d["MaxDPS"][0] for d in vecData], "Max Binary DPS", "cyan", 3, "dashed"))

        vecPlots.append(mplPlot([d["AvgDPS"][2] / d["AD"] if len(d["AvgDPS"]) >= 3 else 0 for d in vecData], "Avg Quad-ary DPS", "royalblue", 3, "solid"))
        #vecPlots.append(mplPlot([d["AvgDPS"][2] for d in vecData], "Avg Quad-ary DPS", "midnightblue", 3, "solid"))

        #print("Standard dev. 5-ary:", [d["StdDPS"][3] for d in vecData])
        #vecPlots.append(mplPlot([d["AvgDPS"][3] / d["AD"] for d in vecData], "Avg 5-ary DPS", "midnightblue", 3, "dashed"))
        #vecPlots.append(mplPlot([d["MaxDPS"][2] for d in vecData], "Max Quad-ary DPS", "midnightblue", 3, "dashed"))

        vecPlots.append(mplPlot([d["AvgDPS"][4] / d["AD"] if len(d["AvgDPS"]) >= 5 else 0 for d in vecData], "Avg 6-ary DPS", "dodgerblue", 3, "solid"))
        #vecPlots.append(mplPlot([d["AvgDPS"][4] for d in vecData], "Avg 6-ary DPS", "royalblue", 3, "solid"))

        #vecPlots.append(mplPlot([d["AvgDPS"][5] / d["AD"] for d in vecData], "Avg 7-ary DPS", "royalblue", 3, "dashed"))
        #vecPlots.append(mplPlot([d["MaxDPS"][4] for d in vecData], "Max 6-ary DPS", "royalblue", 3, "dashed"))

        vecPlots.append(mplPlot([d["AvgDPS"][6] / d["AD"] if len(d["AvgDPS"]) >= 7 else 0 for d in vecData], "Avg 8-ary DPS", "cornflowerblue", 3, "solid"))
        #vecPlots.append(mplPlot([d["AvgDPS"][6] if len(d["AvgDPS"]) >= 7 else 0 for d in vecData], "Avg 8-ary DPS", "cornflowerblue", 3, "solid"))

        #vecPlots.append(mplPlot([d["AvgDPS"][7] for d in vecData], "Avg 9-ary DPS", "cornflowerblue", 3, "dashed"))
        #vecPlots.append(mplPlot([d["MaxDPS"][6] for d in vecData], "Max 8-ary DPS", "dodgerblue", 3, "dashed"))

        vecPlots.append(mplPlot([d["AvgDPS"][8] / d["AD"] if len(d["AvgDPS"]) >= 9 else 0 for d in vecData], "Avg 10-ary DPS", "deepskyblue", 3, "solid"))

        vecPlots.append(mplPlot([d["AvgDPS"][10] / d["AD"] if len(d["AvgDPS"]) >= 11 else 0 for d in vecData], "Avg 12-ary DPS", "cyan", 3, "solid"))
        #vecPlots.append(mplPlot([d["AvgDPS"][10] if len(d["AvgDPS"]) >= 11 else 0 for d in vecData], "Avg 12-ary DPS", "dodgerblue", 3, "solid"))

        #vecPlots.append(mplPlot([d["AvgDPS"][11] for d in vecData], "Avg 13-ary DPS", "dodgerblue", 3, "dashed"))
        #vecPlots.append(mplPlot([d["AvgDPS"][14] / d["AD"] for d in vecData], "Avg 16-ary DPS", "deepskyblue", 3, "solid"))
        #vecPlots.append(mplPlot([d["AvgDPS"][18] for d in vecData], "Avg 20-ary DPS", "lightsteelblue", 3, "solid"))

        # plt.plot(range(len(vecData[0]["NumCells"])), vecData[0]["NumCells"], color = "midnightblue", linewidth = 3, label = "Number of Cells - Layer 0")
        # plt.plot(range(len(vecData[2]["NumCells"])), vecData[2]["NumCells"], color = "royalblue", linewidth = 3, label = "Number of Cells - Layer 2")
        # plt.plot(range(len(vecData[4]["NumCells"])), vecData[4]["NumCells"], color = "dodgerblue", linewidth = 3, label = "Number of Cells - Layer 4")
        # plt.plot(range(len(vecData[6]["NumCells"])), vecData[6]["NumCells"], color = "deepskyblue", linewidth = 3, label = "Number of Cells - Layer 6")
        # plt.plot(range(len(vecData[8]["NumCells"])), vecData[8]["NumCells"], color = "lightsteelblue", linewidth = 3, label = "Number of Cells - Layer 8")
        # plt.legend()
        # plt.show()
        # plt.close()

    
        return vecPlots


class TDAStats(IStat):
    def __init__(self):
        super().__init__("TDA", "Topological Complexity", strFormat = ".json")

    def Compute(self, tData: torch.tensor, tSampleIdx: torch.tensor = None, **kwargs) -> dict:
        assert tSampleIdx is not None, "TDAStats Requires Sample Indices!"

        tF = tData
        f = tF[tSampleIdx,...]
        PD = ComputePD(f)
        vecES, PE, nL, Ea = ComputePDES(PD)
        return {"EntropySummary": vecES, "PersistentEntropy": PE, "NumBars": nL, "LifetimeSum": Ea}
    
    def Plot(self, vecData: list[object]) -> list[mplPlot]:
        vecPlots = []
        #vecPlots.append(mplPlot([d["PersistentEntropy"] for d in vecData], "Persistent Entropy", "limegreen", 3, "solid"))
        vecPlots.append(mplPlot([d["LifetimeSum"] for d in vecData], "Lifetime Sum", "green", 3, "solid"))
        #vecPlots.append(mplPlot([d["NumBars"] for d in vecData], "Number of Lifetimes", "midnightblue", 3, "solid"))

        return vecPlots
    

class DotProdTDAStats(IStat):
    def __init__(self):
        super().__init__("DotProdTDA", "Topological Complexity (DPS)", strFormat = ".json")

    def Compute(self, tData: torch.tensor, tSampleIdx: torch.tensor = None, **kwargs) -> dict:
        assert tSampleIdx is not None, "DotProdTDAStats Requires Sample Indices!"

        tF = tData
        f = tF[tSampleIdx,...]
        PD = ComputeDPSPD(f)
        vecES, PE, nL, Ea = ComputePDES(PD)
        return {"EntropySummary": vecES, "PersistentEntropy": PE, "NumBars": nL, "LifetimeSum": Ea}
    
    def Plot(self, vecData: list[object]) -> list[mplPlot]:
        vecPlots = []
        vecPlots.append(mplPlot([d["PersistentEntropy"] for d in vecData], "Persistent Entropy (DPS)", "limegreen", 3, "solid"))
        vecPlots.append(mplPlot([d["LifetimeSum"] for d in vecData], "Lifetime Sum (DPS)", "green", 3, "dashed"))
        #vecPlots.append(mplPlot([d["NumBars"] for d in vecData], "Number of Lifetimes", "midnightblue", 3, "solid"))

        return vecPlots


class L2Stats(IStat):
    def __init__(self):
        super().__init__("DistanceStats", "L2 Metrics", strFormat = ".json")

    def Compute(self, tData: torch.tensor, tSampleIdx: torch.tensor = None, vecIdx: list[torch.tensor] = None, **kwargs) -> dict:
        assert tSampleIdx is not None or vecIdx is not None, "DistStats Requires Sample Indices AND Class Indices!"

        tF = tData
        f = tF[tSampleIdx,...]
        return IComputeDistMtxStats(f, vecIdx)
    
    def Plot(self, vecData: list[object]) -> list[mplPlot]:
        vecPlots = []
        #vecPlots.append(([d["DistanceEntropy"] for d in vecData], "limegreen", 3, "solid", "Distance Entropy"))
        #vecPlots.append(([min(d["ClassCompression"]) for d in vecData], "limegreen", 3, "dotted", "Minimum Class Compression"))
        #vecPlots.append(([np.mean(np.array(d["ClassCompression"])) for d in vecData], "limegreen", 3, "solid", "Average Class Compression"))
        #vecPlots.append(([max(d["ClassCompression"]) for d in vecData], "limegreen", 3, "dashed", "Maximum Class Compression"))

        #vecPlots.append(([min(d["ClassSeparation"]) for d in vecData], "midnightblue", 3, "dotted", "Minimum Class Separation"))
        #vecPlots.append(([np.mean(np.array(d["ClassSeparation"])) for d in vecData], "midnightblue", 3, "solid", "Average Class Separation"))
        #vecPlots.append(([max(d["ClassSeparation"]) for d in vecData], "midnightblue", 3, "dashed", "Maximum Class Separation"))


        #vecPlots.append(([min(d["MaxIntraClassDist"]) for d in vecData], "cyan", 3, "dotted", "Minimum Max Intra Class Distance"))
        #vecPlots.append(([np.mean(np.array(d["MaxIntraClassDist"])) for d in vecData], "cyan", 3, "solid", "Average Max Intra Class Distance"))
        #vecPlots.append(([max(d["MaxIntraClassDist"]) for d in vecData], "cyan", 3, "dashed", "Maximum Max Intra Class Distance"))

        #vecPlots.append(([min(d["AvgIntraClassDist"]) for d in vecData], "deepskyblue", 3, "dotted", "Minimum Avg Intra Class Distance"))
        #vecPlots.append(([np.mean(np.array(d["AvgIntraClassDist"])) for d in vecData], "deepskyblue", 3, "solid", "Average Avg Intra Class Distance"))
        #vecPlots.append(([max(d["AvgIntraClassDist"]) for d in vecData], "deepskyblue", 3, "dashed", "Maximum Avg Intra Class Distance"))

        #vecPlots.append(([np.mean(np.array(d["MinIntraClassDist"])) for d in vecData], "red", 3, "solid", "Average Min Intra Class Distance"))

        #vecPlots.append(([min(d["MinInterClassDist"]) for d in vecData], "maroon", 3, "dotted", "Minimum Min Inter Class Distance"))
        #vecPlots.append(([np.mean(np.array(d["MinInterClassDist"])) for d in vecData], "maroon", 3, "solid", "Average Min Inter Class Distance"))
        #vecPlots.append(([max(d["MinInterClassDist"]) for d in vecData], "maroon", 3, "dashed", "Maximum Min Inter Class Distance"))

        #vecPlots.append(([min(d["AvgInterClassDist"]) for d in vecData], "red", 3, "dotted", "Minimum Avg Inter Class Distance"))
        #vecPlots.append(([np.mean(np.array(d["AvgInterClassDist"])) for d in vecData], "red", 3, "solid", "Average Avg Inter Class Distance"))
        #vecPlots.append(([max(d["AvgInterClassDist"]) for d in vecData], "red", 3, "dashed", "Maximum Avg Inter Class Distance"))

        vecPlots.append(mplPlot([np.mean(np.array(d["MinClassNorm"])) for d in vecData], "Average Min Class Norm" , "black", 3, "dotted"))
        vecPlots.append(mplPlot([np.mean(np.array(d["AvgClassNorm"])) for d in vecData], "Average Avg Class Norm", "black", 3, "solid"))
        vecPlots.append(mplPlot([np.mean(np.array(d["MaxClassNorm"])) for d in vecData], "Average Max Class Norm", "black", 3, "dotted"))
        #vecPlots.append(([np.mean(np.array(d["MaxClassNorm"])) - np.mean(np.array(d["MinClassNorm"])) for d in vecResults[6]], "black", 3, "dashed", "Average Norm Difference"))

        return vecPlots


class HODStats(IStat):
    def __init__(self):
        super().__init__("NumOrthants", "HOD Stats", strFormat = ".json")

    def Compute(self, tData: torch.tensor, tSampleIdx: torch.tensor = None, **kwargs) -> dict:
        assert tSampleIdx is not None, "HODStats Requires Sample Indices!"
        tF = tData
        f = tF.view(tF.shape[0], -1)
        tCenter = torch.mean(f, dim = 0)
        tOrths = torch.sum(torch.where(f > 0, 1, 0), dim = 1).float()

        tF = tData[tSampleIdx,...]
        dHODMtxStats = IComputeHODMtxStats(tF, bComputeRLEDist = False)

        dHODMtxStats["AvgDistFromPO"] = torch.mean(tOrths).to("cpu").item()
        dHODMtxStats["StdDistFromPO"] = torch.std(tOrths).to("cpu").item()
        dHODMtxStats["AD"] = f.shape[1]
        dHODMtxStats["CenterL1Norm"] = torch.sum(torch.abs(tCenter)).to("cpu").item()

        f = tF.view(tF.shape[0], -1)
        tAbs = torch.abs(tF)
        tAbs /= torch.sum(tAbs, dim = 1, keepdim = True)
        tE = -1 * torch.sum(torch.log(tAbs + 1e-31) * tAbs, dim = 1)
        dHODMtxStats["AvgL1Entropy"] = torch.mean(tE).to("cpu").item()
        dHODMtxStats["StdL1Entropy"] = torch.std(tE).to("cpu").item()

        return dHODMtxStats
    
    def Plot(self, vecData: list[object]) -> list[mplPlot]:
        vecPlots = []

        #vecPlots.append(mplPlot([dO["AvgDistFromPO"] / dO["AD"] for dO in vecData], "Average HOD From Positive Orthant", "cyan", 3, "solid"))
        #vecPlots.append(mplPlot([(dO["AvgDistFromPO"] + dO["StdDistFromPO"]) / dO["AD"] for dO in vecData], "", "cyan", 3, "dotted"))
        #vecPlots.append(mplPlot([(dO["AvgDistFromPO"] - dO["StdDistFromPO"]) / dO["AD"] for dO in vecData], "", "cyan", 3, "dotted"))

        vecPlots.append(mplPlot([dO["AvgHOD"] / dO["AD"] for dO in vecData], "Average Pair-wise HOD", "firebrick", 3, "solid"))
        vecPlots.append(mplPlot([(dO["AvgHOD"] + dO["StdHOD"]) / dO["AD"] for dO in vecData], "", "firebrick", 3, "dotted"))
        #vecPlots.append(mplPlot([(dO["AvgHOD"] - dO["StdHOD"]) / dO["AD"] for dO in vecData], "", "firebrick", 3, "dotted"))
        
        #vecPlots.append(mplPlot([dO["AvgRLE"] / dO["AD"] for dO in vecData], "Average Pair-wise RLE", "slategrey", 3, "solid"))

        #vecPlots.append(mplPlot([dO["AD"] for dO in vecData], "Ambient Dimension", "black", 3, "solid"))
        vecPlots.append(mplPlot([dO["CenterL1Norm"] / dO["AD"] for dO in vecData], "L1 Norm of Global Center", "lime", 3, "solid"))

        return vecPlots
    
class PCAHODStats(IStat):
    def __init__(self):
        super().__init__("NumOrthantsPCA", "PCA'd HOD Stats", strFormat = ".json")

    def Compute(self, tData: torch.tensor, tSampleIdx: torch.tensor = None, **kwargs) -> dict:
        assert tSampleIdx is not None, "PCAHODStats Requires Sample Indices!"
        tF = tData
        tF = tData[tSampleIdx,...]

        f = tF.view(tF.shape[0], -1).to(device)
        tCenter = torch.mean(f, dim = 0, keepdim = True)

        f -= tCenter
        pca = torch_pca.PCA(n_components = 3000, svd_solver = "full")
        pca.fit(f)
        f = pca.transform(f)
        f += pca.transform(tCenter)

        del pca

        dHODMtxStats = IComputeHODMtxStats(f, bComputeRLEDist = True)
        dHODMtxStats["AD"] = f.shape[1]

        return dHODMtxStats
    
    def Plot(self, vecData: list[object]) -> list[mplPlot]:
        vecPlots = []

        vecPlots.append(mplPlot([dO["AvgHOD"] / dO["AD"] for dO in vecData], "Average Pair-wise PCA HOD", "dodgerblue", 3, "solid"))
        #vecPlots.append(mplPlot([(dO["AvgHOD"] + dO["StdHOD"]) / dO["AD"] for dO in vecData], "", "dodgerblue", 3, "dotted"))
        #vecPlots.append(mplPlot([(dO["AvgHOD"] - dO["StdHOD"]) / dO["AD"] for dO in vecData], "", "dodgerblue", 3, "dotted"))

        vecPlots.append(mplPlot([dO["AvgRLE"] / dO["AD"] for dO in vecData], "Average Pair-wise PCA RLE", "lime", 3, "solid"))

        return vecPlots
    

class LogitHODStats(IStat):
    def __init__(self):
        super().__init__("LogitHOD", "HOD of Logits", strFormat = ".json")

    def Compute(self, tData: torch.tensor, **kwargs) -> dict:
        assert "tSampleIdx" in kwargs.keys() and "Trainer" in kwargs.keys() and "Y" in kwargs.keys(), "LogitHODStats Requires Sample Indices, Trainer, and Y"

        tSampleIdx = kwargs["tSampleIdx"]
        Trainer = kwargs["Trainer"]
        Y = kwargs["Y"]

        iSS = 500

        tF = tData
        f = tF[tSampleIdx,...]
        y = Y[tSampleIdx,...]
        ya = torch.argmax(y, dim = 1).to(device)
        f = f.view(f.shape[0], -1).to(device)

        vecParams = Trainer.tModel.GetClassifierParams()
        iProjectDim = vecParams[0].shape[1]

        acc = 0
        tHODMtx = torch.zeros((f.shape[0], f.shape[0]), device = device)
        for _ in tqdm.tqdm(range(iSS)):
            P = torch.randn((f.shape[1], iProjectDim), device = device) / np.sqrt(f.shape[1])
            L = (f @ P) @ torch.transpose(vecParams[0], 0, 1)
            if len(vecParams) > 1: L += vecParams[1]
            tHODMtx += ComputeHODMtx(L, bVerbose = False, bComputeRLEDist = False)
            acc += torch.sum(torch.where(torch.argmax(L, dim = 1) == ya, 1, 0)).to("cpu").item()
        tHODMtx /= iSS
        acc /= (iSS * f.shape[0])

        tIdxTriu = torch.triu_indices(f.shape[0], f.shape[0], offset = 1)

        tFlatDistMtx = tHODMtx[tIdxTriu[0], tIdxTriu[1]]  

        return {
            "AvgHOD": torch.mean(tFlatDistMtx).to("cpu").item(),
            "StdHOD": torch.std(tFlatDistMtx).to("cpu").item(),
            "AD": f.shape[1],
            "ProjectDim": vecParams[0].shape[1],
            "C": vecParams[0].shape[0],
            "Accuracy": acc,
            "SS": iSS,
            }
    
    def Plot(self, vecData: list[object]) -> list[mplPlot]:
        vecPlots = []

        vecPlots.append(mplPlot([dO["AvgHOD"] / dO["C"] for dO in vecData], "Average Pair-wise Logit HOD", "skyblue", 3, "solid"))
        vecPlots.append(mplPlot([(dO["AvgHOD"] + dO["StdHOD"]) / dO["C"] for dO in vecData], "", "skyblue", 3, "dotted"))

        #vecPlots.append(mplPlot([dO["Accuracy"] for dO in vecData], "Post-Projection Accuracy", "orange", 3, "solid"))

        return vecPlots
    

class LogitHODUnifStats(IStat):
    def __init__(self):
        super().__init__("LogitHODUnif", "HOD of Logits (Unif)", strFormat = ".json")

    def Compute(self, tData: torch.tensor, **kwargs) -> dict:
        assert "tSampleIdx" in kwargs.keys() and "Trainer" in kwargs.keys() and "Y" in kwargs.keys(), "LogitHODStats Requires Sample Indices, Trainer, and Y"

        tSampleIdx = kwargs["tSampleIdx"]
        Trainer = kwargs["Trainer"]
        Y = kwargs["Y"]

        iSS = 500

        tF = tData
        f = tF[tSampleIdx,...]
        y = Y[tSampleIdx,...]
        ya = torch.argmax(y, dim = 1).to(device)
        f = f.view(f.shape[0], -1).to(device)

        vecParams = Trainer.tModel.GetClassifierParams()
        iProjectDim = vecParams[0].shape[1]

        acc = 0
        tHODMtx = torch.zeros((f.shape[0], f.shape[0]), device = device)
        for _ in tqdm.tqdm(range(iSS)):
            P = torch.rand((f.shape[1], iProjectDim), device = device) - 0.5
            L = (f @ P) @ torch.transpose(vecParams[0], 0, 1)
            if len(vecParams) > 1: L += vecParams[1]
            tHODMtx += ComputeHODMtx(L, bVerbose = False, bComputeRLEDist = False)
            acc += torch.sum(torch.where(torch.argmax(L, dim = 1) == ya, 1, 0)).to("cpu").item()
        tHODMtx /= iSS
        acc /= (iSS * f.shape[0])

        tIdxTriu = torch.triu_indices(f.shape[0], f.shape[0], offset = 1)

        tFlatDistMtx = tHODMtx[tIdxTriu[0], tIdxTriu[1]]  

        return {
            "AvgHOD": torch.mean(tFlatDistMtx).to("cpu").item(),
            "StdHOD": torch.std(tFlatDistMtx).to("cpu").item(),
            "AD": f.shape[1],
            "ProjectDim": vecParams[0].shape[1],
            "C": vecParams[0].shape[0],
            "Accuracy": acc,
            "SS": iSS,
            }
    
    def Plot(self, vecData: list[object]) -> list[mplPlot]:
        vecPlots = []

        vecPlots.append(mplPlot([dO["AvgHOD"] / dO["C"] for dO in vecData], "Average Pair-wise Logit HOD (Unif)", "darkcyan", 3, "solid"))
        vecPlots.append(mplPlot([(dO["AvgHOD"] + dO["StdHOD"]) / dO["C"] for dO in vecData], "", "darkcyan", 3, "dotted"))

        #vecPlots.append(mplPlot([dO["Accuracy"] for dO in vecData], "Post-Projection Accuracy", "tan", 3, "solid"))

        return vecPlots