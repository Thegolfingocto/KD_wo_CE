import torch
import numpy as np
import matplotlib.pyplot as plt

from Arch.Analysis.MathUtils import *

def GenPDGPU(vecRows, dThresh = np.inf, bDist = False):
    import ripserplusplus as rpp
    if (bDist): strArgs = "--format distance "
    else: strArgs = "--format point-cloud "
    strArgs += "--dim 1 "
    strArgs += "--threshold " + str(dThresh)
    return rpp.run(strArgs, vecRows)

def PlotPDLifetimes(pds: list[np.ndarray], bShow: bool = True):
    for n in range(len(pds)):
        pd = pds[n]
        data = pd[0]
        data1 = pd[1]
        lifetimes = np.zeros((data.shape[0]))
        lifetimes1 = np.zeros((data1.shape[0]))
        for i in range(data.shape[0]):
            lifetimes[i] = data[i][1]  - data[i][0]
        for i in range(data1.shape[0]):
            lifetimes1[i] = data1[i][1]  - data1[i][0]
        
        hist = np.histogram(lifetimes, bins=100)
        hist1 = np.histogram(lifetimes1, bins=100)
        plt.plot(hist[1][:-1], hist[0], label="Class {n} B0")
        plt.plot(hist1[1][:-1], hist1[0], label="Class {n} B1")
    
    if bShow:
        plt.legend()
        plt.show()
        plt.close()
        
def PlotPD(pd: np.ndarray, bShow: bool = True):
    data = pd[0]
    data1 = pd[1]
    m = data[-1][1]
    x = np.arange(0, m, 0.01)
    plt.plot(x, x, linestyle="dashed", color="black")
    
    for i in range(data.shape[0]):
        plt.scatter(data[i][0], data[i][1], color="blue", s=60)
    
    for i in range(data1.shape[0]):
        plt.scatter(data1[i][0], data1[i][1], color="orange", s=60)
        
    if bShow:
        plt.show()
        plt.close()
        
def ComputePDES(nPD: np.ndarray, iRes: int = 100) -> tuple[list[float], float]:
    nL = nPD[:,1] - nPD[:,0]
    L = np.sum(nL)
    nL /= L
    
    nT = np.linspace(np.min(nPD[:,0]), np.max(nPD[:,1]), iRes)
    vecRet = [0 for _ in range(iRes)]
    
    for i in range(nT.shape[0]):
        t = nT[i]
        idx = np.where((nPD[:,0] <= t) & (nPD[:,1] > t))
        l = nL[idx]
        vecRet[i] = (-1 * np.sum(l * np.log(l))).item()
        
    return vecRet, (-1 * np.sum(nL * np.log(nL))).item(), nPD.shape[0], L.item()


def ComputePD(tF: torch.Tensor, tDst: torch.Tensor = None):
    if tDst is None: tDst = ComputeDistanceMatrix(tF)
    PD = GenPDGPU(tDst.to("cpu").numpy(), bDist=True)[0]
    nPD = np.array([[p[0], p[1]] for p in PD])
    return nPD


def ComputeDPSPD(tF: torch.Tensor, tDst: torch.Tensor = None):
    if tDst is None: tDst = 1 - torch.abs(ComputeNormalizedDPMatrix(tF))
    torch.diagonal(tDst, 0).zero_()
    PD = GenPDGPU(tDst.to("cpu").numpy(), bDist=True)[0]
    nPD = np.array([[p[0], p[1]] for p in PD])
    return nPD