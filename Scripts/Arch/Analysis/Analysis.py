'''
Code written by Nicholas J. Cooper.
Released under the MIT license, see the GitHub page for the full legal boilerplate.
tldr: you freely can do whatever you like with this code, but please cite the GitHub at: https://github.com/Thegolfingocto/KDwoCE.git
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
'''



import torch
import numpy as np

from typing import Callable

try:
    from Arch.Analysis.MathUtils import *
    from Arch.Models.ModelUtils import *
    from Arch.Utils.Utils import *
except:
    from MathUtils import *
    from Models.ModelUtils import *
    from Utils.Utils import *

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def IComputeCovarianceStats(F: torch.Tensor, Y: torch.Tensor) -> dict:
    nc = Y.shape[1] if len(Y.shape) == 2 else torch.max(Y) + 1

    tF = F.view(F.shape[0], -1)
    tNorms = torch.norm(tF, dim = 1).to("cpu")

    tF /= torch.norm(tF, dim = 1, keepdim = True) #normalize for computing the DPS

    dim = tF.shape[1]

    vecIdx = SplitLabelsByClass(Y)
    tCenters = torch.zeros((nc, dim))
    for i in range(nc):
        tCenters[i,:] = torch.mean(tF[vecIdx[i],...], dim = 0)
    
    tC = torch.mean(tCenters, dim = 0)

    dMtxCenters = ComputeDistanceMatrix(tCenters)
    tCentEntropy = torch.abs(tCenters) + 1e-5
    tCentEntropy /= torch.sum(tCentEntropy, dim = 1, keepdim = True)
    tCentEntropy = -1 * torch.sum(tCentEntropy * torch.log(tCentEntropy), dim = 1)
    dMtxCenters = dMtxCenters.flatten()[1:].view(nc - 1, nc + 1)[:,:-1].reshape(nc, nc - 1)
    D = (0.5 * torch.min(dMtxCenters)**2).to("cpu").item()

    B = 0
    for i in range(nc):
        B += torch.dot(tCenters[i,:] - tC, tCenters[i,:] - tC).to("cpu").item()
    B /= nc

    W = 0
    N = 0
    for i in tqdm.tqdm(range(nc)):
        for j in range(vecIdx[i].shape[0]):
            W += torch.dot(tF[vecIdx[i][j], ...] - tCenters[i, :], 
                          tF[vecIdx[i][j], ...] - tCenters[i, :])
            N += 1
    W /= N

    C = (W / B).to("cpu").item()

    tDPSMtx = torch.abs(ComputeDPMatrix(tF))
    vecMinDPS = []
    vecAvgDPS = []
    vecMaxDPS = []

    vecMinDPSInter = []
    vecAvgDPSInter = []
    vecMaxDPSInter = []

    vecWeightedAvgIntraDist = []

    vecMinNorm = []
    vecAvgNorm = []
    vecMaxNorm = []

    for i in tqdm.tqdm(range(nc)):
        idxCI = vecIdx[i]

        tSubDPSMtx = tDPSMtx[idxCI, :][:, idxCI]
        idxDiag = torch.triu_indices(idxCI.shape[0], idxCI.shape[0], offset = 1)
        tSubDPSMtx = tSubDPSMtx[idxDiag].to("cpu")
        vecMinDPS.append(float(torch.min(tSubDPSMtx)))
        vecAvgDPS.append(float(torch.mean(tSubDPSMtx)))
        vecMaxDPS.append(float(torch.max(tSubDPSMtx)))

        vecMinNorm.append(float(torch.min(tNorms[idxCI])))
        vecAvgNorm.append(float(torch.mean(tNorms[idxCI])))
        vecMaxNorm.append(float(torch.max(tNorms[idxCI])))

        for j in range(nc):
            if i == j: continue
            idxCJ = vecIdx[j]
            tSubDPSMtx = tDPSMtx[idxCI, :][:, idxCJ]
            vecMinDPSInter.append(float(torch.min(tSubDPSMtx)))
            vecAvgDPSInter.append(float(torch.mean(tSubDPSMtx)))
            vecMaxDPSInter.append(float(torch.max(tSubDPSMtx)))

    dRet = {
        "Compression": C,
        "Discrimination": D,
        "IntraClassCovariance": W.to("cpu").item(),
        "ClassCenterCovariance": B.to("cpu").item(),
        "MinIntraClassDPS": vecMinDPS,
        "AvgIntraClassDPS": vecAvgDPS,
        "MaxIntraClassDPS": vecMaxDPS,

        "MinInterClassDPS": vecMinDPSInter,
        "AvgInterClassDPS": vecAvgDPSInter,
        "MaxInterClassDPS": vecMaxDPSInter,

        "MinClassNorm": vecMinNorm,
        "AvgClassNorm": vecAvgNorm,
        "MaxClassNorm": vecMaxNorm,
        "AverageClassCenterEntropy": torch.mean(tCentEntropy).to("cpu").item(),
        "Log(HostDim)": torch.log(torch.tensor([dim])).to("cpu").item()
    }

    tF *= tNorms.unsqueeze(-1)#.to(device) #rescale the vectors to not screw up downstream analysis
    
    return dRet


def IComputeDistMtxStats(tF: torch.tensor, vecClsIdx: list[torch.tensor]) -> dict:
    n = tF.shape[0]
    nc = len(vecClsIdx)
    tF = tF.view(tF.shape[0], -1)
    tNorms = torch.norm(tF, dim = 1).to("cpu")
    tCenters = torch.zeros((nc, tF.shape[1]))
    for i in range(nc):
        tCenters[i,:] = torch.mean(tF[vecClsIdx[i],:], dim = 0)

    tCentersDistMtx = ComputeDistanceMatrix(tCenters).to("cpu")
    
    tDistMtx = ComputeDistanceMatrix(tF, bVerbose = True).to("cpu")
    

    #print(tCentersDistMtx)

    vecCompression = []
    vecSeparation = []

    vecMinIntra = []
    vecAvgIntra = []
    vecMaxIntra = []
    
    vecMinInter = []
    vecAvgInter = []
    vecMaxInter = []

    vecMinNorm = []
    vecAvgNorm = []
    vecMaxNorm = []

    for i in range(nc):
        idxCI = vecClsIdx[i]
        fMaxIntra = float(torch.max(tDistMtx[idxCI, :][:, idxCI]))
        vecAvgIntra.append(torch.mean(tDistMtx[idxCI, :][:, idxCI]).to("cpu").item())
        vecMinIntra.append(float(torch.min(tDistMtx[idxCI, :][:, idxCI])))

        vecMinNorm.append(float(torch.min(tNorms[idxCI])))
        vecAvgNorm.append(float(torch.mean(tNorms[idxCI])))
        vecMaxNorm.append(float(torch.max(tNorms[idxCI])))

        fMinInter = 2.0**63
        iMinInterIdx = -1
        fAvgInter = 0
        fMaxInter = 0
        for j in range(nc):
            if j == i: continue
            idxCJ = vecClsIdx[j]
            fI = float(torch.min(tDistMtx[idxCI, :][:, idxCJ]))
            fAvgInter += float(torch.mean(tDistMtx[idxCI, :][:, idxCJ]))
            if fI < fMinInter:
                fMinInter = fI
                iMinInterIdx = j
            fM = float(torch.max(tDistMtx[idxCI, :][:, idxCJ]))
            if fM > fMaxInter:
                fMaxInter = fM

        vecMaxIntra.append(fMaxIntra)
        vecMinInter.append(fMinInter)
        vecAvgInter.append(fAvgInter / (nc - 1))
        vecMaxInter.append(fMaxInter)
        vecCompression.append(fMinInter / float(tCentersDistMtx[i, iMinInterIdx]))
        vecSeparation.append(fMinInter / fMaxIntra)

    tIdxTriu = torch.triu_indices(n, n, offset = 1)
    tFlatDistMtx = tDistMtx[tIdxTriu[0], tIdxTriu[1]]
    nDistMtx = tFlatDistMtx.to("cpu").numpy()
    histDist = np.histogram(nDistMtx, bins = 512)

    nDistMtx /= np.sum(nDistMtx)
    fDistEntropy = -1 * np.sum(nDistMtx * np.log(nDistMtx))

    dRet = {
            "DistanceHistCounts": histDist[0].tolist(),
            "DistanceHistValues": histDist[1].tolist(),
            "DistanceEntropy": float(fDistEntropy),
            "ClassCompression": vecCompression,
            "ClassSeparation": vecSeparation,

            "MinIntraClassDist": vecMinIntra,
            "AvgIntraClassDist": vecAvgIntra,
            "MaxIntraClassDist": vecMaxIntra,

            "MinInterClassDist": vecMinInter,
            "AvgInterClassDist": vecAvgInter,
            "MaxInterClassDist": vecMaxInter,

            "MinClassNorm": vecMinNorm,
            "AvgClassNorm": vecAvgNorm,
            "MaxClassNorm": vecMaxNorm,
        }
    
    return dRet

def IComputeDistMtxIDs(tF: torch.Tensor) -> dict:
    F = tF
    F = F.view(F.shape[0], -1)
    
    tNorms = torch.norm(F, dim=1, keepdim=True)
    tDmtx = ComputeDistanceMatrix(F / tNorms, True)
    fNNID = ComputeNNID(tDmtx)
    tDmtx = tDmtx.flatten()[1:].view(tNorms.shape[0] - 1, tNorms.shape[0] + 1)[:,:-1].reshape(tNorms.shape[0], tNorms.shape[0] - 1)
    fMV = torch.mean(tDmtx).item()
    fMVID = fMV**2 / (2 * torch.var(tDmtx).item())
    
    return {
        "TwoNNID": fNNID,
        "AvgVariance": fMV,
        "VarID": fMVID,
        "HostDim": F.shape[1],
    }

def ComputeClassPCAID(F: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    vecIdx = SplitLabelsByClass(Y)
    tRet = torch.zeros((len(vecIdx), max([idx.shape[0] for idx in vecIdx]) + 1))
    for i in tqdm.tqdm(range(len(vecIdx))):
        #print("Computing PCA for class: ", i)
        idx = vecIdx[i]
        f = F[idx,...]
        f = f.view(f.shape[0], -1)
        f -= torch.mean(f, dim = 0)
        if f.shape[1] > f.shape[0]:
            f = torch.transpose(f, 0, 1)
        S = torch.linalg.svdvals(f)
        S = S**2
        totalVar = torch.sum(S)
        S = S / totalVar
        #print(tRet.shape, S.shape)
        tRet[i, :S.shape[0]] = S
        tRet[i, -1] = totalVar

    return tRet

def IComputePCAID(tF: torch.Tensor, fThreshold: float = 0.9) -> dict:
    F = tF
    F = F.reshape(F.shape[0], -1).to(device)
    iAD = F.shape[1]
    if F.shape[1] > F.shape[0]:
        F = torch.transpose(F, 0, 1) #free speed up!
    S = torch.linalg.svdvals(F)
    S = S**2
    totalVar = torch.sum(S)
    S = (S / totalVar).to("cpu")
    
    iD = 0
    while torch.sum(S[:iD]) < fThreshold:
        iD += 1
    
    return {
        "EVRatio": S.numpy().tolist(),
        "PCAID": iD,
        "TotalVariance": float(totalVar),
        "HostDim": iAD
    }
