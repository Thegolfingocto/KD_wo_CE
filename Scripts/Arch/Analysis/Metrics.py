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
