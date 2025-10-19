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
    from Arch.Analysis.EntropyEstimators import *
    from Arch.Models.ModelUtils import *
    from Arch.Utils.Utils import *
except:
    from MathUtils import *
    from EntropyEstimators import *
    from Models.ModelUtils import *
    from Utils.Utils import *

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def IComputeSlicedMI(F: torch.Tensor, YA: torch.Tensor, m: int = 1000, bVerbose: bool = False) -> float:
    F = F.reshape(F.shape[0], -1).to(device)
    dim = F.shape[1]
    SMI = 0

    if bVerbose:
        for _ in tqdm.tqdm(range(m)):
            tProj = torch.rand(dim, 1) - 0.5
            tProj /= torch.norm(tProj)
            X = torch.mm(F, tProj.to(device))
            SMI += mi(X.to("cpu").numpy(), YA.numpy()) * (1 / m)
    else:
        for _ in range(m):
            tProj = torch.rand(dim, 1) - 0.5
            tProj /= torch.norm(tProj)
            X = torch.mm(F, tProj.to(device))
            SMI += mi(X.to("cpu").numpy(), YA.numpy()) * (1 / m)

    return {"SMI": SMI}

def IComputeKSlicedMI(F: torch.Tensor, YA: torch.Tensor, m: int = 1000) -> float:
    F = F.view(F.shape[0], -1).to(device)
    dim = F.shape[1]
    KSMI = 0
    k = torch.max(YA) + 1 #number of classes
    k = 10 #hardcode testing
    for _ in tqdm.tqdm(range(m)):
        tProj = torch.rand(dim, k, device = device) - 0.5
        tProj /= torch.norm(tProj, dim = 0, keepdim = True)
        X = F @ tProj
        KSMI += mi(X.to("cpu").numpy(), YA.numpy()) * (1 / m)

    return {"KSMI": KSMI}

def IComputeConv2SlicedMI(F: torch.Tensor, YA: torch.Tensor, m: int = 1000) -> float:
    F = F.to(device)
    assert len(F.shape) in [2, 4], "Conv2SMI expects CHW tensors"
    KSMI = 0
    k = torch.max(YA) + 1 #number of classes
    k = 10 #hardcode testing
    for _ in tqdm.tqdm(range(m)):
        if len(F.shape) == 4:
            tConv2 = torch.nn.Conv2d(F.shape[1], k, kernel_size = F.shape[2], padding = 0, bias = False)
            #torch.nn.init.kaiming_normal_(tConv2.weight, mode="fan_out", nonlinearity="relu")

            tW = torch.rand_like(tConv2.weight) - 0.5
            #print(tW.shape)
            tW /= torch.norm(tW.view(tW.shape[0], -1), dim = 1, keepdim = True).unsqueeze(-1).unsqueeze(-1)
            #print(tW.shape)
            tConv2.weight = torch.nn.Parameter(tW)

            tConv2 = tConv2.to(device)
            with torch.no_grad():
                X = tConv2(F)
        else:
            Fv = F.view(F.shape[0], -1)
            tProj = torch.rand(Fv.shape[1], k, device = device) - 0.5
            tProj /= torch.norm(tProj, dim = 0, keepdim = True)
            X = F @ tProj
        
        KSMI += mi(X.to("cpu").numpy(), YA.numpy()) * (1 / m)

    return {"Conv2SMI": KSMI}


def IComputeCovarianceStats(F: torch.Tensor, Y: torch.Tensor) -> dict:
    nc = Y.shape[1]

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
        "ClassCenterCovariance": B,
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

def IComputeHODMtxStats(tF: torch.tensor, bComputeRLEDist: bool = True) -> dict:
    tIdxTriu = torch.triu_indices(tF.shape[0], tF.shape[0], offset = 1)

    if bComputeRLEDist:
        tHODMtx, tRLEMtx = ComputeHODMtx(tF, bVerbose = True, bComputeRLEDist = bComputeRLEDist)
        tRLEMtx = tRLEMtx.to("cpu").float()
        tFlatRLEMtx = tRLEMtx[tIdxTriu[0], tIdxTriu[1]]
    else: 
        tHODMtx = ComputeHODMtx(tF, bVerbose = True, bComputeRLEDist = bComputeRLEDist)
    tHODMtx = tHODMtx.to("cpu").float()
    tFlatDistMtx = tHODMtx[tIdxTriu[0], tIdxTriu[1]]  

    dRet = {
        "AvgHOD": torch.mean(tFlatDistMtx).to("cpu").item(),
        "StdHOD": torch.std(tFlatDistMtx).to("cpu").item(),
        }
    
    if bComputeRLEDist:
        dRet["AvgRLE"] = torch.mean(tFlatRLEMtx).to("cpu").item()
        dRet["StdRLE"] = torch.std(tFlatRLEMtx).to("cpu").item()

    return dRet

def IComputeDistMtxStats(tF: torch.tensor, vecClsIdx: list[torch.tensor]) -> dict:
    tDistMtx = ComputeDistanceMatrix(tF, bVerbose = True).to("cpu")
    n = tDistMtx.shape[0]
    nc = len(vecClsIdx)
    tF = tF.view(tF.shape[0], -1)
    tNorms = torch.norm(tF, dim = 1).to("cpu")
    tCenters = torch.zeros((nc, tF.shape[1]))
    for i in range(nc):
        tCenters[i,:] = torch.mean(tF[vecClsIdx[i],:], dim = 0)

    tCentersDistMtx = ComputeDistanceMatrix(tCenters).to("cpu")

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

def IComputeDistMtxIDs(F: torch.Tensor) -> dict:
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

def IComputePCAID(F: torch.Tensor, fThreshold: float = 0.9) -> dict:
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


def IComputeBinaryDPStats(F: torch.tensor) -> dict:
    tF = F.view(F.shape[0], -1)
    tF /= torch.norm(tF, dim = 1, keepdim = True) #normalize for computing the DPS

    tDPM = torch.abs(ComputeDPMatrix(tF)).to("cpu")

    iN = tF.shape[0]
    idxTriu = torch.triu_indices(iN, iN, 1)
    #tDPS = torch.triu(torch.abs(tDPM), diagonal = 1).view(-1)
    tDPS = tDPM[idxTriu[0], idxTriu[1]]
    #tDPS = tDPM.flatten()[1:].view(iN - 1, iN + 1)[:,:-1]        #.reshape(iN, iN - 1)
    # tDPS = torch.zeros((0))
    # for i in range(iN):
    #     for j in range(i + 1, iN):
    #         tDPS = torch.cat((tDPS, tDPM[i:i+1, j]))

    print(tDPS.shape)

    dRet = {
        "AvgDPS": torch.mean(tDPS).item(),
        "StdDPS": torch.std(tDPS).item(),

        "MinDPS": torch.min(tDPS).item(),
        "MaxDPS": torch.max(tDPS).item(),
    }

    return dRet

def IComputeTernaryDPStats(F: torch.tensor) -> dict:
    tF = F.view(F.shape[0], -1)
    tF /= torch.norm(tF, dim = 1, keepdim = True, p = 3) #normalize for computing the DPS

    tDPM = torch.abs(ComputeTernaryDPMatrix(tF))

    iN = tF.shape[0]
    tDPS = tDPM[tDPM != 0] #this relies on the watermarks provided by ComputeTernaryDPMatrix()
    # tDPS = torch.zeros((0))
    # for i in range(iN):
    #     for j in range(i + 1, iN):
    #         for k in range(j + 1, iN):
    #             tDPS = torch.cat((tDPS, tDPM[i:i+1, j, k]))

    print(tDPS.shape)

    dRet = {
        "AvgDPS": torch.mean(tDPS).item(),
        "StdDPS": torch.std(tDPS).item(),

        "MinDPS": torch.min(tDPS).item(),
        "MaxDPS": torch.max(tDPS).item(),
    }

    return dRet


def IComputeDPSCCStats(F: torch.tensor, fEps: float = 1e-2, bRandom: bool = False) -> dict:
    tF = F.view(F.shape[0], -1)

    vecCCs, vecDPMs, vecAlphas, fMax, fAvg = ComputeDPSCC(tF, fEps = fEps, bRandom = bRandom)

    tDepth = torch.zeros((tF.shape[0]))
    vecN = []
    for tCC in vecCCs:
        iCnt = 0
        for i in range(tF.shape[0]):
            if not tCC[i, -1]: continue #skip disabled cells
            alpha = tCC[i, -2]
            tDepth[tCC[i, :alpha]] = torch.maximum(tDepth[tCC[i, :alpha]], alpha)
            iCnt += 1
        vecN.append(iCnt)

    iMaxAlpha = int(max([torch.max(tA).item() for tA in vecAlphas])) + 1
    #print(iMaxAlpha)
    vecDPS = [torch.zeros((0)) for _ in range(iMaxAlpha)]
    for i in range(len(vecAlphas)):
        #print("i:", i)
        tDPM = vecDPMs[i]
        #print(torch.mean(tDPM))
        tAlpha = vecAlphas[i]

        for a in range(int(torch.min(tAlpha)), int(torch.max(tAlpha) + 1)):
            #print(a)
            idx = torch.where(tAlpha == a)[0]
            #print(idx)
            #input()
            vecDPS[a] = torch.cat((vecDPS[a], tDPM[idx]))

    vecDPS = vecDPS[2:] #zero-ary and unary don't make sense here

    vecAvgDPS = [torch.mean(tDPS).item() if tDPS.shape[0] != 0 else 0 for tDPS in vecDPS]
    vecStdDPS = [torch.std(tDPS).item() if tDPS.shape[0] != 0 else 0 for tDPS in vecDPS]
    vecMaxDPS = [torch.max(tDPS).item() if tDPS.shape[0] != 0 else 0 for tDPS in vecDPS]


    dRet = {
        "AvgDepth": torch.mean(tDepth).item(),
        "StdDepth": torch.std(tDepth).item(),
        "MaxDepth": iMaxAlpha,

        "Width": torch.where(tDepth > 0)[0].shape[0] / tDepth.shape[0],
        "NumCells": vecN,
        "AD": tF.shape[1],

        "AvgDPS": vecAvgDPS,
        "StdDPS": vecStdDPS,
        "MaxDPS": vecMaxDPS,

        "AvgDPSbyLvl": [torch.mean(dpm).item() for dpm in vecDPMs],
        "StdDPSbyLvl": [torch.std(dpm).item() for dpm in vecDPMs],
        "MaxDPSbyLvl": [torch.max(dpm).item() for dpm in vecDPMs],

        "AbsMax": fMax,
        "AbsAvg": fAvg,
    }

    nAvgDPS = np.array(dRet["AvgDPS"])
    nAvgDPS /= np.sum(nAvgDPS)
    dRet["AvgDPSEntropy"] = -1 * np.sum(nAvgDPS * np.log(nAvgDPS))

    nMaxDPS = np.array(dRet["MaxDPS"])
    nMaxDPS /= np.sum(nMaxDPS)
    dRet["MaxDPSEntropy"] = -1 * np.sum(nMaxDPS * np.log(nMaxDPS))

    return dRet


def main() -> None:
    # tF = torch.zeros((6, 100))
    # tF[0, :] = 1
    # tF[1, :] = 1.41
    # tF[2, :] = 0.72
    # tF[3:,:] = -1 * tF[:3,:]

    # vecIdx = [[0, 1, 2], [3, 4, 5]]

    # print(IComputeDistMtxStats(tF, vecIdx)["ClassCompression"])

    X = torch.randn((100, 2048))
    print(IComputeBinaryDPStats(X))
    print(IComputeTernaryDPStats(X))
    return

if __name__ == "__main__":
    main()