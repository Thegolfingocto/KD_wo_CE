'''
Code written by Nicholas J. Cooper.
Released under the MIT license, see the GitHub page for the full legal boilerplate.
tldr: you freely can do whatever you like with this code, but please cite the GitHub at: https://github.com/Thegolfingocto/KDwoCE.git
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
'''


import torch
import copy
import os
from typing import Callable
import math

from sklearn.decomposition import PCA
#Important NOTE: PCA.transform() implicity centers the result via XW' - mean(X, dim=0)W' !!!!

from Arch.IConfig import *
from Arch.Utils import *

from Arch.Analysis.MathUtils import ComputeDistanceMatrix, ComputeDPMatrix

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FKDAttention(torch.nn.Module):
    def __init__(self, iNS: int, iNT: int, fConstructModel: Callable[[int, list[int]], torch.nn.Module], iDimIn: int, vecHiddenDims: list[int]) -> None:
        ''''
        Simple wrapper around multiple MLPs for KD purposes
        '''
        super(FKDAttention, self).__init__()
        self.vecStudentMLPs = torch.nn.ModuleList([fConstructModel(iDimIn, vecHiddenDims, bNormalizeOutput = True).to(device) for _ in range(iNS)])
        self.vecTeacherMLPs = torch.nn.ModuleList([fConstructModel(iDimIn, vecHiddenDims, bNormalizeOutput = True).to(device) for _ in range(iNT)])
        self.iOutDim = vecHiddenDims[-1]
        self.iNS = iNS
        self.iNT = iNT
        
    def forward(self, xS, xT): 
        tSRet = torch.zeros((xS.shape[0], self.iOutDim, self.iNS), device = xS.device)
        tTRet = torch.zeros((xT.shape[0], self.iOutDim, self.iNT), device = xT.device)
        
        for i in range(self.iNS):
            tSRet[:, :, i] = self.vecStudentMLPs[i](xS[:, :, i])
            
        for i in range(self.iNT):
            tTRet[:, :, i] = self.vecTeacherMLPs[i](xT[:, :, i])
        
        return tSRet, tTRet


class FKDProjector1D(torch.nn.Module):
    def __init__(self, fConsturctModel: Callable[[int, int], torch.nn.Module], mapStoTLayers: list[list[int]],
                 vecSShape: list[torch.Size], vecTShape: list[torch.Size], bPoolMode: bool = True, bLearned: bool = True) -> None:
        super(FKDProjector1D, self).__init__()
        self.mapStoTLayers = mapStoTLayers
        self.mapStoTModels = copy.deepcopy(mapStoTLayers) #we modify this copy
        self.mapStoTUpConvs = None if bPoolMode else copy.deepcopy(mapStoTLayers) #somewhere to store the transposed convolutions
        self.bPoolMode = bPoolMode

        self.vecSShape = vecSShape
        self.vecTShape = vecTShape

        print("Creating projectors")

        if self.bPoolMode:
            for i in range(len(self.mapStoTLayers)):
                for j in range(len(self.mapStoTLayers[i])):
                    #print(vecSC[i], vecTC[self.mapStoTLayers[i][j]])
                    self.mapStoTModels[i][j] = fConsturctModel(vecSShape[i][1], vecTShape[self.mapStoTLayers[i][j]][1], bLearned).to(device)
                self.mapStoTModels[i] = torch.nn.ModuleList(self.mapStoTModels[i])

        else:
            for i in range(len(self.mapStoTLayers)):
                for j in range(len(self.mapStoTLayers[i])):
                    iSC = vecSShape[i][1]
                    iTC = vecTShape[self.mapStoTLayers[i][j]][1]
                    if len(vecSShape[i]) == 3:
                        iSDim = vecSShape[i][2]
                    elif len(vecSShape[i]) == 4:
                        iSDim = vecSShape[i][2] * vecSShape[i][3]
                    else: iSDim = 1
                    if len(vecTShape[self.mapStoTLayers[i][j]]) == 3:
                        iTDim = vecTShape[self.mapStoTLayers[i][j]][2]
                    elif len(vecTShape[self.mapStoTLayers[i][j]]) == 4:
                        iTDim = vecTShape[self.mapStoTLayers[i][j]][2] * vecTShape[self.mapStoTLayers[i][j]][3]
                    else: iTDim = 1

                    #This assumes things play nicely (e.g. are divisible!)
                    if iSDim <= iTDim:
                        iStride = math.ceil((iTDim - 3) / (max([iSDim, 2]) - 1))
                        self.mapStoTUpConvs[i][j] = torch.nn.ConvTranspose1d(iSC, iSC, 3, stride = iStride, bias = False).to(device)
                        #torch.nn.init.kaiming_normal_(self.mapStoTUpConvs[i][j].weight, mode="fan_out")
                    elif iSDim > iTDim:
                        iStride = math.ceil((iSDim - 3) / (max([iTDim, 2]) - 1))
                        self.mapStoTUpConvs[i][j] = torch.nn.ConvTranspose1d(iTC, iTC, 3, stride = iStride, bias = False).to(device)
                        #torch.nn.init.kaiming_normal_(self.mapStoTUpConvs[i][j].weight, mode="fan_out")

                    iInC = iSC if iSC < iTC else iTC
                    iOutC = iSC if iSC > iTC else iTC

                    self.mapStoTModels[i][j] = fConsturctModel(iInC, iOutC).to(device)
                self.mapStoTModels[i] = torch.nn.ModuleList(self.mapStoTModels[i])
                #if self.mapStoTUpConvs is not None: self.mapStoTUpConvs[i] = torch.nn.ModuleList(self.mapStoTUpConvs[i])


        self.mapStoTModels = torch.nn.ModuleList(self.mapStoTModels)
        #if self.mapStoTUpConvs is not None: self.mapStoTUpConvs = torch.nn.ModuleList(self.mapStoTUpConvs)
                
        return
    
    def forward(self, vecSF: list[torch.Tensor], vecTF: list[torch.Tensor]):
        vecSRet = []
        vecTRet = []
        #loop over layer map student-wise
        for iSL in range(len(self.mapStoTLayers)):
            vecSR = []
            vecTR = []
            for iMidx in range(len(self.mapStoTLayers[iSL])):
                iTL = self.mapStoTLayers[iSL][iMidx]
                if len(vecSF[iSL].shape) == 2:
                    #print(vecSF[iSL].shape)
                    vecSF[iSL] = vecSF[iSL].unsqueeze(-1)
                elif len(vecSF[iSL].shape) == 4:
                    vecSF[iSL] = vecSF[iSL].view(vecSF[iSL].shape[0], vecSF[iSL].shape[1], -1)
                elif len(vecSF[iSL].shape) != 3:
                    print("Error! Invalid student tensor of shape {}".format(vecSF[iSL].shape))
                    
                if len(vecTF[iTL].shape) == 2:
                    #print(vecTF[iTL].shape)
                    vecTF[iTL] = vecTF[iTL].unsqueeze(-1)
                elif len(vecTF[iTL].shape) == 4:
                    vecTF[iTL] = vecTF[iTL].view(vecTF[iTL].shape[0], vecTF[iTL].shape[1], -1)
                elif len(vecTF[iTL].shape) != 3:
                    print("Error! Invalid teacher tensor of shape {}".format(vecTF[iTL].shape))
                
                #print(vecSF[iSL].shape, vecTF[iTL].shape) #useful debug
                
                iSS, iTS = vecSF[iSL].shape[2], vecTF[iTL].shape[2]
                iSC, iTC = vecSF[iSL].shape[1], vecTF[iTL].shape[1]
                
                #print(iSH, iSW, iTH, iTW)
                
                #dimensionally align the features
                if self.bPoolMode:
                    if iSS <= iTS:
                        vecTR.append(torch.nn.functional.adaptive_avg_pool1d(vecTF[iTL], (iSS)))
                        vecSR.append(self.mapStoTModels[iSL][iMidx](vecSF[iSL]))
                    else:
                        vecTR.append(vecTF[iTL]) #no modification necessary for teacher features in this case
                        vecSR.append(self.mapStoTModels[iSL][iMidx](
                            torch.nn.functional.adaptive_avg_pool1d(vecSF[iSL], (iTS))))
                else:
                    if iSS <= iTS:
                        with torch.no_grad():
                            vecSF[iSL] = self.mapStoTUpConvs[iSL][iMidx](vecSF[iSL])
                        vecSF[iSL] = torch.nn.functional.adaptive_avg_pool1d(vecSF[iSL], (iTS))
                    elif iTS < iSS:
                        with torch.no_grad():
                            vecTF[iTL] = self.mapStoTUpConvs[iSL][iMidx](vecTF[iTL])
                        vecTF[iTL] = torch.nn.functional.adaptive_avg_pool1d(vecTF[iTL], (iSS))


                    if iSC <= iTC:
                        vecTR.append(vecTF[iTL])
                        vecSR.append(self.mapStoTModels[iSL][iMidx](vecSF[iSL]))
                    else:
                        vecSR.append(vecSF[iSL])
                        vecTR.append(self.mapStoTModels[iSL][iMidx](vecTF[iTL]))

            vecSRet.append(vecSR)
            vecTRet.append(vecTR)
            
        return vecSRet, vecTRet
    
    def inference(self, tSF: torch.tensor, iSL: int = -1) -> torch.tensor:
        if len(self.mapStoTLayers[iSL]) > 1:
            print("inference() currently only supports one-to-one layer mapping!")
            return
        
        #grab the single (by hypothesis) teacher layer index
        iTL = self.mapStoTLayers[iSL][0]

        #downsample and project
        if (self.vecSShape[iSL][2] <= self.vecTShape[iTL][2]):
            return self.mapStoTModels[iSL][0](tSF)
        else:
            return self.mapStoTModels[iSL][0](torch.nn.functional.adaptive_avg_pool1d(tSF, (self.vecTShape[iTL][2])))
        

class FKDProjector2D(torch.nn.Module):
    def __init__(self, fConsturctModel: Callable[[int, int], torch.nn.Module], mapStoTLayers: list[list[int]],
                 vecSShape: list[torch.Size], vecTShape: list[torch.Size], bPoolMode: bool = True, bLearned: bool = True) -> None:
        super(FKDProjector2D, self).__init__()
        self.mapStoTLayers = mapStoTLayers
        self.mapStoTModels = copy.deepcopy(mapStoTLayers) #we modify this copy
        self.mapStoTUpConvs = None if bPoolMode else copy.deepcopy(mapStoTLayers) #somewhere to store the transposed convolutions
        self.bPoolMode = bPoolMode

        #self.vecDownsampleStudent = [(vecSShape[i][2] >= vecTShape[i][2]) and (vecSShape[i][3] >= vecTShape[i][3]) for i in range(len(vecSShape))]
        self.vecSShape = vecSShape
        self.vecTShape = vecTShape

        print("Creating projectors")
        if self.bPoolMode:
            for i in range(len(self.mapStoTLayers)):
                for j in range(len(self.mapStoTLayers[i])):
                    #print(vecSC[i], vecTC[self.mapStoTLayers[i][j]])
                    self.mapStoTModels[i][j] = fConsturctModel(vecSShape[i][1], vecTShape[self.mapStoTLayers[i][j]][1], bLearned).to(device)
                self.mapStoTModels[i] = torch.nn.ModuleList(self.mapStoTModels[i])

        else:
            for i in range(len(self.mapStoTLayers)):
                for j in range(len(self.mapStoTLayers[i])):
                    iSC = vecSShape[i][1]
                    iTC = vecTShape[self.mapStoTLayers[i][j]][1]
                    if len(vecSShape[i]) == 4:
                        iSDim = vecSShape[i][2]
                    else: iSDim = 1
                    if len(vecTShape[self.mapStoTLayers[i][j]]) == 4:
                        iTDim = vecTShape[self.mapStoTLayers[i][j]][2]
                    else: iTDim = 1

                    #This assumes things play nicely (e.g. are divisible!)
                    if iSDim <= iTDim:
                        iStride = math.ceil(iTDim / iSDim)
                        self.mapStoTUpConvs[i][j] = torch.nn.ConvTranspose2d(iSC, iSC, 3, stride = iStride, bias = False).to(device)
                        #torch.nn.init.kaiming_normal_(self.mapStoTUpConvs[i][j].weight, mode="fan_out")
                    elif iSDim > iTDim:
                        iStride = math.ceil(iSDim / iTDim)
                        self.mapStoTUpConvs[i][j] = torch.nn.ConvTranspose2d(iTC, iTC, 3, stride = iStride, bias = False).to(device)
                        #torch.nn.init.kaiming_normal_(self.mapStoTUpConvs[i][j].weight, mode="fan_out")

                    iInC = iSC if iSC < iTC else iTC
                    iOutC = iSC if iSC > iTC else iTC

                    self.mapStoTModels[i][j] = fConsturctModel(iInC, iOutC).to(device)
                self.mapStoTModels[i] = torch.nn.ModuleList(self.mapStoTModels[i])
                #if self.mapStoTUpConvs is not None: self.mapStoTUpConvs[i] = torch.nn.ModuleList(self.mapStoTUpConvs[i])

        self.mapStoTModels = torch.nn.ModuleList(self.mapStoTModels)
                
        return
    
    def forward(self, vecSF: list[torch.Tensor], vecTF: list[torch.Tensor]):
        vecSRet = []
        vecTRet = []
        #loop over layer map student-wise
        for iSL in range(len(self.mapStoTLayers)):
            vecSR = []
            vecTR = []
            for iMidx in range(len(self.mapStoTLayers[iSL])):
                iTL = self.mapStoTLayers[iSL][iMidx]
                if len(vecSF[iSL].shape) == 2:
                    #print(vecSF[iSL].shape)
                    vecSF[iSL] = vecSF[iSL].unsqueeze(-1)
                    vecSF[iSL] = vecSF[iSL].unsqueeze(-1)
                elif len(vecSF[iSL].shape) != 4:
                    print("Error! Invalid student tensor of shape {}".format(vecSF[iSL].shape))
                    
                if len(vecTF[iTL].shape) == 2:
                    #print(vecTF[iTL].shape)
                    vecTF[iTL] = vecTF[iTL].unsqueeze(-1)
                    vecTF[iTL] = vecTF[iTL].unsqueeze(-1)
                elif len(vecTF[iTL].shape) != 4:
                    print("Error! Invalid teacher tensor of shape {}".format(vecTF[iTL].shape))
                
                #print(vecSF[iSL].shape, vecTF[iTL].shape) #useful debug
                
                iSH, iSW, iTH, iTW = vecSF[iSL].shape[2], vecSF[iSL].shape[3], vecTF[iTL].shape[2], vecTF[iTL].shape[3]
                iSC, iTC = vecSF[iSL].shape[1], vecTF[iTL].shape[1]
                
                #print(iSH, iSW, iTH, iTW)
                
                if self.bPoolMode:
                    #dimensionally align the features
                    if iSH <= iTH and iSW <= iTW:
                        vecTR.append(torch.nn.functional.adaptive_avg_pool2d(vecTF[iTL], (iSH, iSW)))
                        vecSR.append(self.mapStoTModels[iSL][iMidx](vecSF[iSL]))
                    elif iSH > iTH and iSW > iTW:
                        vecTR.append(vecTF[iTL]) #no modification necessary for teacher features in this case
                        vecSR.append(self.mapStoTModels[iSL][iMidx](
                            torch.nn.functional.adaptive_avg_pool2d(vecSF[iSL], (iTH, iTW))))
                    else:
                        print("Unsupported projection! ({}x{})-->({}x{})".format(iTH, iTW, iSH, iSW))

                else:
                    if iSH <= iTH and iSW <= iTW:
                        with torch.no_grad():
                            vecSF[iSL] = self.mapStoTUpConvs[iSL][iMidx](vecSF[iSL])
                        vecSF[iSL] = torch.nn.functional.adaptive_avg_pool2d(vecSF[iSL], (iTH, iTW))
                    elif iSH > iTH and iSW > iTW:
                        with torch.no_grad():
                            vecTF[iTL] = self.mapStoTUpConvs[iSL][iMidx](vecTF[iTL])
                        vecTF[iTL] = torch.nn.functional.adaptive_avg_pool2d(vecTF[iTL], (iSH, iSW))
                    else:
                        print("Unsupported projection! ({}x{})-->({}x{})".format(iTH, iTW, iSH, iSW))


                    if iSC <= iTC:
                        vecTR.append(vecTF[iTL])
                        vecSR.append(self.mapStoTModels[iSL][iMidx](vecSF[iSL]))
                    else:
                        vecSR.append(vecSF[iSL])
                        vecTR.append(self.mapStoTModels[iSL][iMidx](vecTF[iTL]))

            vecSRet.append(vecSR)
            vecTRet.append(vecTR)
            
        return vecSRet, vecTRet
    
    def inference(self, tSF: torch.tensor, iSL: int = -1) -> torch.tensor:
        if len(self.mapStoTLayers[iSL]) > 1:
            print("inference() currently only supports one-to-one layer mapping!")
            return
        
        #grab the single (by hypothesis) teacher layer index
        iTL = self.mapStoTLayers[iSL][0]

        #downsample and project
        if (self.vecSShape[iSL][2] <= self.vecTShape[iTL][2]) and (self.vecSShape[iSL][3] <= self.vecTShape[iTL][3]):
            return self.mapStoTModels[iSL][0](tSF)
        else:
            return self.mapStoTModels[iSL][0](torch.nn.functional.adaptive_avg_pool2d(tSF, (self.vecTShape[iTL][2], self.vecTShape[iTL][3])))

class FKDLoss:
    def __init__(self, vecStudentLayers: list[int], vecTeacherLayers: list[int], 
                 vecStudentFeatures: list[torch.Tensor], vecTeacherFeatures: list[torch.Tensor], strMappingMode: str,
                 strProjectionMode: str, iBatchSize: int, vecStudentPartitions: list[int] = [], vecTeacherPartitions: list[int] = [],
                 mapStoTLayers: list[list[int]] = [[]], mapStoTWeights: list[list[float]] = [[]], bStoreFeatures: bool = True,
                 vecPowers: list[float] = []) -> None:
        '''
        Constructor expects lists of tensors of the appropriate shape. If using PCA or another DimR method, 
        pass all relevant teacher feature maps for projection.
        strMappingMode := {"One2One" / "O", "PreDefined" / "P", "LearnedAttn" / "L"}
        One2One: maps teacher->student layers in the order passed to the constructor
        PreDefined: maps teacher->student layers based on the map provided in mapStoTLayers, these maps can be linear combinations.
        FullyConnected: all student layers learn from all teacher layers (loss averaged).
        LearnedAttn: implements idea from: https://ojs.aaai.org/index.php/AAAI/article/view/16865. Effectively the same as FullyConnected with the
        addition of learned attention maps per student layer across teacher features. If provided, segments teacher and student based on splits 
        provided in vecXPartitions. [iP1, iP2,...] ==> [0:iP1], [iP1:iP2], ...

        strProjectionMode := {"PCA", "LearnedProjector", "RelationFunction"}
        PCA: fits PCA matrices for each mapped pair of layers using the supplied teacher features. 
        LearnedProjector: by using the SetProjector() method, user can specify a learnable projector Module. Makes copies under the hood 
        for each mapped pair of layers.
        RelationFunction: User can specify a Callable[Tensor]->Tensor relation function by calling SetRelation(). Function must return 
        tensors of shape = function of input tensor.shape[0] or fixed shape.

        bStoreFeatures controls memory management scheme. When true (default), this class maintains a copy of all teacher features
        in RAM. This results in fast train time as no inference passes need to be made thru the teacher. If features are too large
        to fit in RAM, set this to false. Then, user is responsible for passing inference'd features from teacher for each call to
        ComputeLoss().
        '''
        if len(vecStudentLayers) != len(vecStudentFeatures):
            print("Error! Mismatched number of student layers!")
            return
        if len(vecTeacherLayers) != len(vecTeacherFeatures):
            print("Error! Mismatched number of teacher layers!")
            return
        self.bStoreFeatures = bStoreFeatures
        self.bGlobalOnly = False
        self.vecPowers = vecPowers
        
        self.vecStudentLidx: list[int] = copy.deepcopy(vecStudentLayers)
        self.vecTeacherLidx: list[int] = copy.deepcopy(vecTeacherLayers)
        
        self.vecStudentFeatureSizes: list[int] = [tF.contiguous().view(tF.shape[0], -1).shape[1] for tF in vecStudentFeatures]
        self.vecStudentFeatureShapes: list[torch.Size] = [tF.shape for tF in vecStudentFeatures]
        self.vecStudentChannels: list[int] = [tF.shape[1] for tF in vecStudentFeatures]
        self.vecTeacherFeatureSizes: list[int] = [tF.contiguous().view(tF.shape[0], -1).shape[1] for tF in vecTeacherFeatures]
        self.vecTeacherFeatureShapes: list[torch.Size] = [tF.shape for tF in vecTeacherFeatures]
        self.vecTeacherChannels: list[int] = [tF.shape[1] for tF in vecTeacherFeatures]
        
        #print(self.vecStudentChannels)
        #print(self.vecTeacherChannels)
        
        self.iSL: int = len(self.vecStudentFeatureSizes)
        self.iTL: int = len(self.vecTeacherFeatureSizes)
        self.iBatchSize: int = iBatchSize
        
        self.vecStudentPartitions: list[int] = None
        self.vecTeacherPartitions: list[int] = None
        self.mapStudentPartitionToLayers: list[list[int]] = None
        self.mapTeacherPartitionToLayers: list[list[int]] = None
        
        self.strProjMode = strProjectionMode
        
        #Allocate VRAM up front to avoid re-allocs every loss calculation
        #self.vecLoss = [torch.zeros((iBatchSize, self.vecStudentFeatureSizes[i]), device = device) for i in range(len(self.vecStudentFeatureSizes))]
        self.tSAttnIn = None
        self.tTAttnIn = None
        
        if self.bStoreFeatures or self.strProjMode == "PCA":
            '''
            Need teacher's features temporarily for PCA fitting
            '''
            self.vecTeacherFeatures: list[torch.Tensor] = vecTeacherFeatures #TODO: potentially revisit this memory-efficiency nightmare
        
        #layer maps
        self.mapStoTLayers: list[list[int]] = None
        self.mapStoTWeights: list[list[float]] = None
        self.mapTtoSLayers: list[list[int]] = None
        
        #modules
        self.tProj: torch.nn.Module = None
        self.tAttn: torch.nn.Module = None
        self.cRelation: Callable[[torch.Tensor], torch.Tensor] = None
        
        #pca stuff
        self.strPCADir: str = None
        self.vecGlobalPCAs: list[torch.Tensor] = None
        self.vecGlobalCenters: list[torch.Tensor] = None
        self.vecClassPCAs: list[torch.Tensor] = None
        self.vecClassCenters: list[torch.Tensor] = None
        self.vecTranslatedClassCenters: list[torch.Tensor] = None
        self.vecRequiredDims: list[int] = None
        
        if strMappingMode == "One2One" or strMappingMode == "O":
            #usage checks
            if self.iTL != self.iSL:
                print("Error! Setting up for One2One mode, but different numbers of teacher/student features provided! ({} vs. {})".format(
                    self.iTL, self.iSL))
                return
            #Setup the map
            self.mapStoTLayers = [[i] for i in range(self.iTL)]
            self.mapStoTWeights = [[1] for i in range(self.iTL)]
            
        elif strMappingMode == "PreDefined" or strMappingMode == "P":
            #usage checks
            if len(mapStoTLayers) > self.iSL or len(mapStoTWeights) > self.iSL:
                print("Error! If using PreDefined mode, pass map indices relative to student feature vectors!")
                return
            #Setup the map
            self.mapStoTLayers = copy.deepcopy(mapStoTLayers)
            #translate from absolute to relative indices
            for i in range(self.iSL):
                for j in range(len(self.mapStoTLayers[i])):
                    self.mapStoTLayers[i][j] = self.vecTeacherLidx.index(self.mapStoTLayers[i][j])
            if len(mapStoTWeights) < self.iSL:
                self.mapStoTWeights = [[1 for _ in range(len(mapStoTLayers[i]))] for i in range(len(mapStoTLayers))]
            else:
                self.mapStoTWeights = copy.deepcopy(mapStoTWeights)
        
        elif strMappingMode == "FullyConnected" or strMappingMode == "FC":
            self.mapStoTLayers = [[i for i in range(self.iTL)] for _ in range(self.iSL)]
            self.mapStoTWeights = [[1.0 / self.iTL for _ in range(self.iTL)] for _ in range(self.iSL)] 

        elif strMappingMode == "LearnedAttn" or strMappingMode == "L":
            #usage checks
            if len(vecStudentPartitions) > 0 or len(vecTeacherPartitions) > 0:
                if len(vecStudentPartitions) != len(vecTeacherPartitions):
                    print("Error! If using partitions, #tp must equal #sp!")
                    return
                # if max(vecStudentPartitions) > self.iSL or max(vecTeacherPartitions) > self.iTL:
                #     print("Error! Partition indices must be relative to the passed feature vectors!")
                #     return
                
            #Setup the map
            if len(vecStudentPartitions) > 0:
                #make copys cause python is weird
                self.vecStudentPartitions = copy.deepcopy(vecStudentPartitions)
                self.vecTeacherPartitions = copy.deepcopy(vecTeacherPartitions)
                
                #translate negative indices
                self.vecStudentPartitions = [iP + self.vecStudentLidx[-1] + 1 if iP < 0 else iP for iP in self.vecStudentPartitions]
                self.vecTeacherPartitions = [iP + self.vecTeacherLidx[-1] + 1 if iP < 0 else iP for iP in self.vecTeacherPartitions]
                
                #add end points
                if self.vecStudentPartitions[-1] <= self.vecStudentLidx[-1]:
                    self.vecStudentPartitions.append(self.vecStudentLidx[-1] + 1)
                if self.vecTeacherPartitions[-1] <= self.vecTeacherLidx[-1]:
                    self.vecTeacherPartitions.append(self.vecTeacherLidx[-1] + 1)
                
                #translate to relative indexing
                self.mapStudentPartitionToLayers = [[] for _ in range(len(self.vecStudentPartitions))]
                self.mapTeacherPartitionToLayers = [[] for _ in range(len(self.vecTeacherPartitions))]
                sidx = 0
                for i in range(len(self.vecStudentLidx)):
                    if self.vecStudentLidx[i] >= self.vecStudentPartitions[sidx] and sidx + 1< len(self.mapStudentPartitionToLayers):
                        sidx += 1
                    self.mapStudentPartitionToLayers[sidx].append(i)
                tidx = 0
                for i in range(len(self.vecTeacherLidx)):
                    if self.vecTeacherLidx[i] >= self.vecTeacherPartitions[tidx]:
                        tidx += 1
                    self.mapTeacherPartitionToLayers[tidx].append(i)
                #print(self.mapStudentPartitionToLayers)
                #print(self.mapTeacherPartitionToLayers)
                self.vecStudentPartitions = [0] + [self.mapStudentPartitionToLayers[i][-1] + 1 for i in range(len(self.mapStudentPartitionToLayers))]
                self.vecTeacherPartitions = [0] + [self.mapTeacherPartitionToLayers[i][-1] + 1 for i in range(len(self.mapTeacherPartitionToLayers))]
                
                self.mapStoTLayers = []
                self.mapStoTWeights = []
                for i in range(self.iSL):
                    pidx = self.GetStudentPartitionIdx(i)
                    self.mapStoTLayers.append([j for j in self.mapTeacherPartitionToLayers[pidx]])
                    self.mapStoTWeights.append([-1 for _ in range(len(self.mapStoTLayers[-1]))])
                
            else:
                #self.tSAttnIn = torch.zeros((self.iSL, iBatchSize, iBatchSize), device = device)
                #self.tTAttnIn = torch.zeros((self.iTL, iBatchSize, iBatchSize), device = device)
                self.mapStoTLayers = [[i for i in range(self.iTL)] for _ in range(self.iSL)]
                self.mapStoTWeights = [[-1 for _ in range(self.iTL)] for _ in range(self.iSL)] #these get learned later
            
        else:
            print("Error! Invalid mapping mode {}".format(strMappingMode))
            return
        
        self.mapTtoSLayers = [[] for _ in range(self.iTL)]
        for i in range(len(self.mapStoTLayers)):
            for l in self.mapStoTLayers[i]:
                if i not in self.mapTtoSLayers[l]:
                    self.mapTtoSLayers[l].append(i)
        
        return
    
    def GetStudentPartitionIdx(self, idx: int) -> int:
        for i in range(len(self.mapStudentPartitionToLayers)):
            if idx in self.mapStudentPartitionToLayers[i]:
                return i
        print("Warning! Failed to find partition for index {}".format(idx))
        return -1
    
    def GetTeacherPartitionIdx(self, idx: int) -> int:
        for i in range(len(self.mapTeacherPartitionToLayers)):
            if idx in self.mapTeacherPartitionToLayers[i]:
                return i
        print("Warning! Failed to find partition for index {}".format(idx))
        return -1
    
    def SetProjector(self, tProj: Callable[[int, int], torch.nn.Module], iDim: int = 2, bPoolMode: bool = True, bLearned: bool = True) -> None:
        if iDim == 2:
            self.tProj = FKDProjector2D(tProj, self.mapStoTLayers, self.vecStudentFeatureShapes, self.vecTeacherFeatureShapes, bPoolMode = bPoolMode, bLearned = bLearned)
        elif iDim == 1:
            self.tProj = FKDProjector1D(tProj, self.mapStoTLayers, self.vecStudentFeatureShapes, self.vecTeacherFeatureShapes, bPoolMode = bPoolMode, bLearned = bLearned)
        else:
            print("Invalid dimension {}!".format(iDim))
        return
    
    def GetProjector(self) -> torch.nn.Module:
        return self.tProj
    
    def SetAttnModule(self, tAttn: Callable[[int, list[int]], torch.nn.Module], vecArch: list[int]) -> None:
        self.tAttn = FKDAttention(self.iSL, self.iTL, tAttn, self.iBatchSize, vecArch)
        return
    
    def GetAttnModule(self) -> torch.nn.Module:
        return self.tAttn
    
    def SetRelation(self, cRel: Callable[[torch.Tensor], torch.Tensor]) -> None:
        self.cRelation = cRel
        return
    
    def GetLayerWeightMap(self) -> torch.Tensor:
        '''
        Getter for plotting the learned attention maps
        '''
        tMap = torch.zeros((self.iSL, self.iTL))
        for iSL in range(len(self.mapStoTWeights)):
            for iWidx in range(len(self.mapStoTWeights[iSL])):
                iTL = self.mapStoTLayers[iSL][iWidx]
                tMap[iSL, iTL] = torch.mean(self.mapStoTWeights[iSL][iWidx])
        #print(tMap)
        return tMap
    
    def ComputeClassCenters(self, vecF: list[torch.Tensor]):
        tCC = torch.zeros((len(vecF), vecF[0].shape[1]), device = device)
        for i in range(len(vecF)):
            tCC[i, :] = torch.sum(vecF[i], dim=0)
            tCC[i, :] /= vecF[i].shape[0]
        return tCC
    
    def FitAndSavePCA(self, tF: torch.Tensor, iDim: int, strDir: str, Y: torch.Tensor):
        tF = tF.view(tF.shape[0], -1).to(device)

        if iDim > tF.shape[1]:
            print("Error! Trying to increase dimensionality with PCA! Double check your Layer Map")
            return
        if not os.path.isdir(strDir):
            os.mkdir(strDir)
        
        print("Fitting Global PCA")
        pcaG = PCA(n_components = iDim)
        pcaG.fit(tF.to("cpu"))
        
        strSavePath = strDir + "GlobalPCA.pkl"
        if os.path.exists(strSavePath):
            print("Warning! File {} already exists!".format(strSavePath))
            if not GetInput("Overwrite? (Y/X)"):
                return
        with open(strSavePath, "wb") as f:
            torch.save(torch.tensor(pcaG.components_).float(), f)
            
        print("Computing Global Center")
        strSavePath = strDir + "GlobalCenter.pkl"
        with open(strSavePath, "wb") as f:
            torch.save(torch.mean(tF, dim = 0, keepdim = True), f)
        
        vecF = SplitTrainFeaturesByClass(tF, Y)
        print("Computing Class Centers")
        tCenters = self.ComputeClassCenters(vecF)
        strSavePath = strDir + "ClassCenters.pkl"
        with open(strSavePath, "wb") as f:
            torch.save(tCenters, f)

        if self.bGlobalOnly: return

        tW = torch.zeros((len(vecF), iDim, tF.shape[1]))
        for i in range(len(vecF)):
            print("Fitting PCA for class {}".format(i))
            pca = PCA(n_components = iDim)
            pca.fit((vecF[i] - tCenters[i,:]).to("cpu"))
            tW[i,:,:] = torch.tensor(pca.components_).float()
        strSavePath = strDir + "ClassPCAs.pkl"
        with open(strSavePath, "wb") as f:
            torch.save(tW, f)
                
        return
    
    def LoadPCA(self, strBaseDir: str, Y: torch.Tensor, bUseGlobalCenter: bool = True, vecTeacherFeatures: list[torch.Tensor] = []) -> None:
        '''
        loads fit pca's for all required teacher->student layer pairs. If missing, generates them.
        Operates in the base directory provided. If labels are passed (Y), #c + 1 PCAs are fit, one per class and a global
        '''
        self.strPCADir = strBaseDir + "PCA/"
        if not os.path.isdir(self.strPCADir):
            os.mkdir(self.strPCADir)
        
        if len(vecTeacherFeatures) > 0:
            #safety check
            if len(vecTeacherFeatures) != len(self.vecTeacherFeatures):
                print("Error! Incorrect number of new teacher features passed to LoadPCA()!")
                return
            #overwrite the stored features if new ones are provided here
            self.vecTeacherFeatures = vecTeacherFeatures
        
        #figure out how many distinct PCAs we need for the layer config
        self.vecRequiredDims = [0 for _ in range(len(self.vecTeacherFeatures))]
        for i in range(len(self.mapTtoSLayers)):
            for j in self.mapTtoSLayers[i]:
                sz = self.vecStudentFeatureSizes[j]
                if sz > self.vecRequiredDims[i]:
                    self.vecRequiredDims[i] = sz
                    
        #Load in the PCAs
        if self.vecGlobalPCAs is not None or self.vecClassPCAs is not None:
            print("Warning! PCAs already loaded")
            if not GetInput("Overwrite? (Y/X)"):
                return
        self.vecGlobalPCAs = []
        self.vecGlobalCenters = []
        self.vecClassPCAs = []
        self.vecClassCenters = []
        self.vecTranslatedClassCenters = []
        for i in range(len(self.vecRequiredDims)):
            strDir = self.GenPCAString(i, self.vecRequiredDims[i])
            #generate the required matrices if they don't already exist
            if not os.path.exists(strDir) or not os.path.exists(strDir + "GlobalPCA.pkl"):
                self.FitAndSavePCA(self.vecTeacherFeatures[i], self.vecRequiredDims[i], strDir, Y)
            #read in the data
            with open(strDir + "GlobalPCA.pkl", "rb") as f:
                self.vecGlobalPCAs.append(torch.transpose(torch.load(f).to(device), 0, 1))
            with open(strDir + "GlobalCenter.pkl", "rb") as f:
                self.vecGlobalCenters.append(torch.load(f).to(device))
            with open(strDir + "ClassCenters.pkl", "rb") as f:
                self.vecClassCenters.append(torch.load(f).to(device))
                if bUseGlobalCenter:
                    self.vecTranslatedClassCenters.append(
                        torch.matmul((self.vecClassCenters[-1] - self.vecGlobalCenters[-1]), self.vecGlobalPCAs[-1]))
                else:
                    self.vecTranslatedClassCenters.append(
                        torch.matmul(self.vecClassCenters[-1], self.vecGlobalPCAs[-1]))
            
            if not self.bGlobalOnly:
                with open(strDir + "ClassPCAs.pkl", "rb") as f:
                    self.vecClassPCAs.append(torch.transpose(torch.load(f).to(device), 1, 2))
                
        if not self.bStoreFeatures:
            #Free up VRAM if not operating in stored feature mode
            for i in range(len(self.vecTeacherFeatures), 0, -1):
                del self.vecTeacherFeatures[i - 1]
                
            self.vecTeacherFeatures = None
        
        return
    
    def GenPCAString(self, tidx: int, sdim: int) -> str:
        return self.strPCADir + "L" + str(self.vecTeacherLidx[tidx]) + "_" + str(self.vecTeacherFeatureSizes[tidx]) + "_" + str(sdim) + "/"
    
    def ComputeLoss(self, vecSF: list[torch.Tensor], tIdx: torch.Tensor, Y: torch.Tensor, dOpt: dict, vecTF: list[torch.Tensor] = None) -> list[torch.Tensor]:
        '''
        Top level loss calculation function. If not operating in stored feature mode, caller must supply teacher features for the batch.
        Requires Y to be in argmax() format, e.g. not one-hot encoded
        TODO: Add stored vs. not TF mode
        '''
        #Error checks
        if len(vecSF) != len(self.vecStudentLidx):
            print("Error! Incorrect number of student features passed to ComputeLoss()")
            return None
        if vecTF is not None and len(vecTF) != len(self.vecTeacherLidx):
            print("Error! Incorrect number of teacher features passed to ComputeLoss()")
            return None
        if vecTF is None and not self.bStoreFeatures:
            print("Error! Must pass teacher features to ComputeLoss() when not operating in stored feature mode!")
            return None
        
        if vecTF is None and self.bStoreFeatures:
            #Load the relevent features from storage
            vecTF = [self.vecTeacherFeatures[i][tIdx,...].to(device) for i in range(len(self.vecTeacherFeatures))]

        if len(self.vecPowers) == len(vecTF):
            for i in range(len(vecTF)):
                fN = torch.norm_except_dim(vecTF[i], dim = 0)#[:, 0, 0, 0]
                vecTF[i] = torch.pow(vecTF[i], self.vecPowers[i])
                fN2 = torch.norm_except_dim(vecTF[i], dim = 0)#[:, 0, 0, 0]
                vecTF[i] *= fN / fN2
        
        #update the layer map weights when operating in learnedattn mode
        if self.tAttn is not None:
            self.ComputeAttnWeights(vecSF, vecTF, dOpt)
        
        if self.strProjMode == "PCA":
            return self.ComputePCALoss(vecSF, vecTF, Y, dOpt)
        elif self.strProjMode == "LearnedProjector":
            return self.ComputeProjLoss(vecSF, vecTF, dOpt)
        elif self.strProjMode == "RelationFunction":
            return self.ComputeRelationLoss(vecSF, vecTF, dOpt)
        
        #This should never execute
        print("Error! Something weird happened!")
        return None
    
    def ComputePCALoss(self, vecSF: list[torch.Tensor], vecTF: list[torch.Tensor], Y: torch.Tensor, dOpt: dict) -> list[torch.Tensor]:
        '''
        loss calculation for PCA proj. mode. pass options via dOpt, these can change dynamically post initialization.
        Requires Y to be in argmax() format, e.g. not one-hot encoded
        '''
        
        
        #torch.nn.HuberLoss(), torch.nn.CosineEmbeddingLoss

        #torch.nn.HuberLoss(), torch.nn.CosineEmbeddingLoss
        #TODO: add support for non-CC center calulation modes
        
        if self.tAttn is not None:
            self.ComputeAttnWeights(vecSF, vecTF, dOpt)
        
        vecLoss = [torch.zeros((self.iBatchSize, self.vecStudentFeatureSizes[i]), device = device) for i in range(len(self.vecStudentFeatureSizes))]
        
        #main layer loops
        for iTL in range(len(self.mapTtoSLayers)):
            #loop over teacher layers in order to only project each feature once
            tF = vecTF[iTL].view(self.iBatchSize, -1) 
            
            if self.bGlobalOnly:
                C = self.vecGlobalCenters[iTL]
                W = self.vecGlobalPCAs[iTL]
                TC = self.vecTranslatedClassCenters[iTL][Y, :]

                #Project the features
                T = ((torch.matmul((tF - C), W) - TC) * dOpt["LandmarkPostScaleFactor"]) + TC
            else:
                #select the appropriate class centers and projections
                C = self.vecClassCenters[iTL][Y, :]
                W = self.vecClassPCAs[iTL][Y, :, :]
                TC = self.vecTranslatedClassCenters[iTL][Y, :]
                
                #Project the features
                T = (torch.bmm((tF - C).unsqueeze(1), W)[:,0,:] * dOpt["LandmarkPostScaleFactor"]) + TC
            
            #push all required subsets to their respective student layers    
            for iWidx in range(len(self.mapTtoSLayers[iTL])):
                #get the relevent index and size
                iSL = self.mapTtoSLayers[iTL][iWidx]
                iD = self.vecStudentFeatureSizes[iSL]
                #calculate loss
                if dOpt["UseHuberLoss"]:
                    vecLoss[iSL] += (self.mapStoTWeights[iSL][iWidx] * 
                                                torch.nn.functional.huber_loss(vecSF[iSL].view(self.iBatchSize, -1), 
                                                T[:, :iD], reduction = "none"))
                elif dOpt["UseCosineLoss"]:
                    vecLoss[iSL] += (self.mapStoTWeights[iSL][iWidx] * 
                                                torch.nn.functional.cosine_embedding_loss(vecSF[iSL].view(self.iBatchSize, -1),
                                                T[:, :iD], torch.ones((self.iBatchSize), device = device), reduction = "none"))
                else:
                    vecLoss[iSL] += (self.mapStoTWeights[iSL][iWidx] * 
                                                torch.nn.functional.mse_loss(vecSF[iSL].view(self.iBatchSize, -1), 
                                                T[:, :iD], reduction = "none"))
        
        return vecLoss
    
    def ComputeProjLoss(self, vecSF: list[torch.Tensor], vecTF: list[torch.Tensor], dOpt: dict) -> list[torch.Tensor]:
        '''
        loss calculation for learned projector mode. Based on FKDProjector class.
        '''

        vecST, vecTT = self.tProj(vecSF, vecTF)

        
        vecLoss = [torch.zeros((self.iBatchSize, 1), device = device) for _ in range(len(vecST))]
        for iSL in range(len(self.mapStoTLayers)):
            for iWidx in range(len(self.mapStoTLayers[iSL])):
                #calculate loss
                if dOpt["UseHuberLoss"]:
                    vecLoss[iSL] += torch.mean((self.mapStoTWeights[iSL][iWidx] * 
                                                torch.nn.functional.huber_loss(vecST[iSL][iWidx].view(self.iBatchSize, -1), 
                                                                               vecTT[iSL][iWidx].view(self.iBatchSize, -1),
                                                                               reduction = "none")), dim = 1, keepdim = True)
                elif dOpt["UseCosineLoss"]:
                    vecLoss[iSL] += torch.mean((self.mapStoTWeights[iSL][iWidx] * 
                                                torch.nn.functional.cosine_embedding_loss(vecST[iSL][iWidx].view(self.iBatchSize, -1),
                                                vecTT[iSL][iWidx].view(self.iBatchSize, -1), torch.ones((self.iBatchSize), device = device),
                                                reduction = "none")), dim = 1, keepdim = True)
                else:
                    # print(self.mapStoTWeights[iSL][iWidx].shape, vecST[iSL][iWidx].view(self.iBatchSize, -1).shape,
                    #       vecTT[iSL][iWidx].view(self.iBatchSize, -1).shape) #useful debug
                    vecLoss[iSL] += torch.mean((self.mapStoTWeights[iSL][iWidx] * 
                                                torch.nn.functional.mse_loss(vecST[iSL][iWidx].reshape(self.iBatchSize, -1), 
                                                                             vecTT[iSL][iWidx].reshape(self.iBatchSize, -1),
                                                                             reduction = "none")), dim = 1, keepdim = True)
                
        return vecLoss
    
    def ComputeRelationLoss(self, vecSF: list[torch.Tensor], vecTF: list[torch.Tensor], dOpt: dict) -> list[torch.Tensor]:
        vecSR = [self.cRelation(f) for f in vecSF]
        vecTR = [self.cRelation(f) for f in vecTF]
        vecLoss = [torch.zeros((1), device = device) for _ in range(len(vecSR))]
        for iSL in range(len(self.mapStoTLayers)):
            for iWidx in range(len(self.mapStoTLayers[iSL])):
                #calculate loss
                if dOpt["UseHuberLoss"]:
                    vecLoss[iSL] += self.mapStoTWeights[iSL][iWidx] * (torch.nn.functional.huber_loss(vecSR[iSL][iWidx], 
                                                                        vecTR[iSL][iWidx], reduction = "mean"))
                else:
                    vecLoss[iSL] += self.mapStoTWeights[iSL][iWidx] * (torch.nn.functional.mse_loss(vecSR[iSL][iWidx], 
                                                                        vecTR[iSL][iWidx], reduction = "mean"))
                    
        return vecLoss
    
    def ComputeAttnWeights(self, vecSF: list[torch.Tensor], vecTF: list[torch.Tensor], dOpt: dict) -> None:
        tSAttnIn = torch.zeros((self.iBatchSize, self.iBatchSize, self.iSL), device = device)
        tTAttnIn = torch.zeros((self.iBatchSize, self.iBatchSize, self.iTL), device = device)
        #Compute the inputs to the attn module
        if dOpt["AttnInput"] == "DPS":
            for i in range(len(vecSF)):
                tS = vecSF[i]
                tSAttnIn[:,:,i] = ComputeDPMatrix(tS)
                
            for i in range(len(vecTF)):
                tT = vecTF[i]
                tTAttnIn[:,:,i] = ComputeDPMatrix(tT)
        elif dOpt["AttnInput"] == "ED":
            for i in range(len(vecSF)):
                tS = vecSF[i]
                tSAttnIn[:,:,i] = ComputeDistanceMatrix(tS)
                
            for i in range(len(vecTF)):
                tT = vecTF[i]
                tTAttnIn[:,:,i] = ComputeDistanceMatrix(tT)
        else:
            print("Error! Unsupported AttnInput {}".format(dOpt["AttnInput"]))
            return
        
        tQ, tK = self.tAttn(tSAttnIn, tTAttnIn)
        tAttnValues = torch.bmm(torch.transpose(tQ, 1, 2), tK)
        if self.vecStudentPartitions is not None and len(self.vecStudentPartitions) > 0:
            tAttnWeights = torch.zeros_like(tAttnValues)
            #partition the softmax operations
            for i in range(1, len(self.vecStudentPartitions)):
                tAttnWeights[:, self.vecStudentPartitions[i - 1]:self.vecStudentPartitions[i],
                self.vecTeacherPartitions[i - 1]:self.vecTeacherPartitions[i]] = torch.nn.functional.softmax(
                    tAttnValues[:, self.vecStudentPartitions[i - 1]:self.vecStudentPartitions[i],
                    self.vecTeacherPartitions[i - 1]:self.vecTeacherPartitions[i]], 
                    dim = 2)
        else:
            tAttnWeights = torch.nn.functional.softmax(tAttnValues, dim = 2)
        
        #Set the weights to the result of the attention calculation
        for iSL in range(len(self.mapStoTLayers)):
            for iWidx in range(len(self.mapStoTLayers[iSL])):
                iTL = self.mapStoTLayers[iSL][iWidx]
                self.mapStoTWeights[iSL][iWidx] = tAttnWeights[:, iSL, iTL].unsqueeze(1)
        return
    
if __name__ == "__main__":
    x = torch.randn((1, 1, 1, 1))
    vecS = [x for _ in range(5)]
    vecSL = [1, 2, 3, 4, 5]
    vecT = [x for _ in range(5)]
    vecTL = [2, 4, 8, 12, 16]
    
    fkd = FKDLoss(vecSL, vecTL, vecS, vecT, "L", "PCA", 128, vecStudentPartitions=[4], vecTeacherPartitions=[12])
    
    print(fkd.mapStoTLayers)
    print(fkd.mapTtoSLayers)
    print(fkd.vecStudentPartitions)
    print(fkd.vecTeacherPartitions)