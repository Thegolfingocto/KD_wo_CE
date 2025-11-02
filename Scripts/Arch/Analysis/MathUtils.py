'''
Code written by Nicholas J. Cooper.
Released under the MIT license, see the GitHub page for the full legal boilerplate.
tldr: you freely can do whatever you like with this code, but please cite the GitHub at: https://github.com/Thegolfingocto/KDwoCE.git
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
'''

import numpy as np
import torch
import matplotlib.pyplot as plt
import tqdm
import math
import subprocess
import os
import copy
import time
from Arch.Models.Misc import LinearHead
from Arch.Models.ModelUtils import CountParams

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Cuda speed up is not from github

# Path setup
if os.environ["USER"] == "nickubuntuworkstation" or os.environ["USER"] == "nicklaptop":
    strBasePath = "/home/" + os.environ["USER"] + "/Repos/HSKD/"
elif os.environ["USER"] == "nickubuntu":
    strBasePath = "/home/" + os.environ["USER"] + "/ReposWSL/HSKD/"
else:
    print("NOT SET UP YET!")

def PlotLine3D(p1, p2, ax, color='black', style='solid'):
    t = np.arange(0, 1.01, 0.01)
    line = np.zeros((100, 3))
    for i in range(100):
        line[i,:] = (p1 * t[i]) + (p2 * (1 - t[i]))
    ax.plot(line[:,0], line[:,1], line[:,2], color=color, linestyle=style)
    return
    
def PlotSurface3D(p1, p2, p3, ax):
    t1 = np.arange(0, 1.01, 0.01)
    surf = np.zeros((0, 3))
    for i in range(100):
        t2 = np.arange(0, 1.01 - t1[i], 0.01)
        for j in range(int((1.01 - t1[i]) / 0.01)):
            surf = np.concatenate((surf, (p1 * t1[i]) + (p2 * t2[j]) + (p3 * (1 - t1[i] - t2[j]))), axis=0)
    print(surf)
    ax.plot_surface(surf[:,0], surf[:,1], surf[:,2:3])
    return
    
def PlotLine2D(p1, p2, color='black'):
    t = np.arange(0, 1.01, 0.01)
    line = np.zeros((100, 2))
    for i in range(100):
        line[i,:] = (p1 * t[i]) + (p2 * (1 - t[i]))
    plt.plot(line[:,0], line[:,1], color=color)

def ComputePointsV3(n, d, ss=0.01, iter=500):
    args = ['../build/ComputeSphericalCode', str(n), str(d), str(ss), str(iter), 'temp.npy']
    subprocess.run(args)
    with open('temp.npy', 'rb') as f:
        data = np.load(f)
    os.remove('temp.npy')
    return torch.tensor(data)

class LinearRegressor(torch.nn.Module):
    def __init__(self, iDimIn: int, iDimOut: int) -> None:
        super(LinearRegressor, self).__init__()
        self.tLin = torch.nn.Linear(iDimIn, iDimOut, bias = False)
        
    def forward(self, x):
        return self.tLin(x)
    
def LinearRegression(X, Y, dThreshold: float = 1e-2) -> torch.Tensor:
    tLR = LinearRegressor(X.shape[1], Y.shape[1]).to(X.device)
    tOpt = torch.optim.SGD(tLR.parameters(), lr = 0.01)
    #tCrit = torch.nn.MSELoss()
    tCrit = torch.nn.HuberLoss()
    iN = 0
    with torch.no_grad():
        loss = tCrit(tLR(X), Y).to("cpu").item()
    while loss > dThreshold and iN < 100000:
        tOpt.zero_grad()
        l = tCrit(tLR(X), Y)
        l.backward()
        tOpt.step()
        loss = l.to("cpu").item()
        iN += 1
    
    if iN == 100000:
        print("Stop reached without attaining {} loss".format(dThreshold))
    
    return tLR.tLin.weight.detach().to("cpu").item()

def ComputeNNID(tDmtx: torch.Tensor) -> float:
    if len(tDmtx.shape) != 2 or tDmtx.shape[0] != tDmtx.shape[1]:
        print("Error! Expected square matrix")
        return None
    
    tU = torch.zeros((tDmtx.shape[0]))
    for i in range(tDmtx.shape[0]):
        tR = torch.sort(tDmtx[i,:])[0]
        if tR[2] != tR[1]:
            tU[i] = tR[2] / tR[1]

    tU = tU[torch.where(tU != 0)[0]]
    
    tU = torch.log(torch.sort(tU)[0])
    tF = torch.arange(0, tU.shape[0]).float() / tU.shape[0]
    
    tF = -1 * torch.log(1 - tF)
    
    # plt.plot(tU.to("cpu").numpy(), tF.to("cpu").numpy())
    # plt.show()
    # plt.close()

    tID = LinearRegression(tU.unsqueeze(-1), tF.unsqueeze(-1))
    
    #print("TNNID:", tID)
    
    return tID

def ComputeDisjointDistanceMatrix(pts1: torch.tensor, pts2: torch.tensor):
    pts1 = pts1.to("cuda")
    pts2 = pts2.to("cuda")
    ret = torch.zeros((pts1.shape[0], pts2.shape[0]), device="cuda")
    
    for i in tqdm.tqdm(range(pts1.shape[0])):
        dists = pts2 - pts1[i:i+1,:]
        dists = dists**2
        dists = torch.sum(dists, dim=1)
        ret[i,:] = dists
        
    return ret.to("cpu")

def ComputeDisjointDPMatrix(pts1: torch.tensor, pts2: torch.tensor):
    pts1 = pts1.to("cuda")
    pts2 = pts2.to("cuda")
    ret = torch.zeros((pts1.shape[0], pts2.shape[0]), device="cuda")
    
    for i in tqdm.tqdm(range(pts1.shape[0])):
        dists = torch.multiply(pts2, pts1[i:i+1,:])
        dists = torch.sum(dists, dim=1)
        ret[i,:] = dists
        
    return ret.to("cpu")

def ComputeHODMtx(tPts: torch.tensor, bVerbose: bool = False, bComputeRLEDist: bool = True):
    tPts = tPts.to("cuda")
    tPts = tPts.view(tPts.shape[0], -1)
    ret = torch.zeros((tPts.shape[0], tPts.shape[0]), device="cuda")
    ret2 = torch.zeros_like(ret) if bComputeRLEDist else None
    tPts = torch.where(tPts > 0, 1, 0)
    torch.cuda.empty_cache()
    tPts = tPts.float()
    torch.cuda.empty_cache()

    if bVerbose:
        for i in tqdm.tqdm(range(tPts.shape[0])):
            tDist = torch.abs(tPts - tPts[i:i+1,:])
            ret[i,:] = torch.sum(tDist, dim = 1)
            torch.cuda.empty_cache()

            if bComputeRLEDist:
                tDist = tDist.unsqueeze(1)
                tK = torch.tensor([-1.0, 1.0], device = device).unsqueeze(0).unsqueeze(0)
                tDist = torch.abs(torch.nn.functional.conv1d(tDist, tK))
                ret2[i, :] = torch.sum(tDist[:, 0, :], dim = 1)
        
    else:
        for i in range(tPts.shape[0]):
            tDist = torch.abs(tPts - tPts[i:i+1,:])
            ret[i,:] = torch.sum(tDist, dim = 1)

            if bComputeRLEDist:
                tDist = tDist.unsqueeze(1)
                tK = torch.tensor([-1.0, 1.0], device = device).unsqueeze(0).unsqueeze(0)
                tDist = torch.abs(torch.nn.functional.conv1d(tDist, tK))
                ret2[i, :] = torch.sum(tDist[:, 0, :], dim = 1)
    
    if not bComputeRLEDist: return ret

    return ret, ret2

def ComputeDistanceMatrix(pts: torch.tensor, bVerbose: bool = False):
    pts = pts.to("cuda")
    pts = pts.view(pts.shape[0], -1)
    ret = torch.zeros((pts.shape[0], pts.shape[0]), device="cuda")
    
    if bVerbose:
        for i in tqdm.tqdm(range(pts.shape[0])):
            ret[i,:] = torch.sum((pts - pts[i:i+1,:])**2, dim = 1)**0.5
        
    else:
        for i in range(pts.shape[0]):
            ret[i,:] = torch.sum((pts - pts[i:i+1,:])**2, dim = 1)**0.5
        
    return ret#.to("cpu")

def ComputeSquaredDistanceMatrix(pts: torch.tensor, bVerbose: bool = False):
    pts = pts.to("cuda")
    pts = pts.view(pts.shape[0], -1)
    ret = torch.zeros((pts.shape[0], pts.shape[0]), device="cuda")
    
    if bVerbose:
        for i in tqdm.tqdm(range(pts.shape[0])):
            ret[i,:] = torch.sum((pts - pts[i:i+1,:])**2, dim = 1)
        
    else:
        for i in range(pts.shape[0]):
            ret[i,:] = torch.sum((pts - pts[i:i+1,:])**2, dim = 1)
        
    return ret#.to("cpu")

def ComputeDPMatrix(pts: torch.tensor):
    pts = pts.to("cuda")
    
    try:
        pts = pts.view(pts.shape[0], -1)
    except:
        pts = pts.reshape(pts.shape[0], -1)
    
    ret = torch.matmul(pts, torch.transpose(pts, 0, 1))
        
    return ret#.to("cpu")

def ComputeNormalizedDPMatrix(pts: torch.tensor, e: float = 1e-4):
    pts = pts.to("cuda")
    
    try:
        pts = pts.view(pts.shape[0], -1)
    except:
        pts = pts.reshape(pts.shape[0], -1)

    pts = pts / (torch.norm(pts, dim = 1, keepdim = True) + e)

    pts[pts != pts] = 0 #get rid of nans
    
    ret = torch.matmul(pts, torch.transpose(pts, 0, 1))
        
    return ret

def ComputeRowNormalizedDPMatrix(pts: torch.tensor, e: float = 1e-4) -> torch.tensor:
    dmtx = ComputeDPMatrix(pts)
    return dmtx / (torch.norm(dmtx, dim = 1, keepdim = True) + e)


def OrderPointsByDotProduct(pts: torch.tensor, lidx: int = -1, N: int = -1, dst: torch.tensor = None):
    ref = pts[lidx:lidx+1,:] if lidx > 0 else pts[0:1,:]
    if dst is None:
        dists = pts - ref
        dists = dists**2
        dists = torch.sum(dists, dim=1)
        r = torch.argsort(dists)
    else:
        r = torch.argsort(dst[lidx,:])
    if N < 0 : N = pts.shape[0]
    return pts[r[:N], :], r[:N]

def OrderPointsByMutualCloseness(pts: torch.tensor, lidx: int = -1, k: int = -1, N: int = -1, dst: torch.tensor = None, Verbose: bool = False):
    if dst is None: dst = ComputeDistanceMatrix(pts)
    idx = [lidx] if lidx > 0 else [0]
    eidx = N if N > 0 else pts.shape[0]
    ridx = [i for i in range(pts.shape[0])]
    del ridx[idx[0]]
    
    if Verbose:
        for _ in tqdm.tqdm(range(1, eidx)):
            if k > 0:
                tdst = torch.sum(dst[idx[-k:], :][:, ridx], dim=0)
            else:
                tdst = torch.sum(dst[idx, :][:, ridx], dim=0)
            #tdst[tdst == 0] = 1000 #make sure we select affinely independant points #unclear if we want this or not
            nidx = torch.argmin(tdst)
            idx.append(ridx[nidx])
            del ridx[nidx]
    else:
        for _ in range(1, eidx):
            if k > 0:
                tdst = torch.sum(dst[idx[-k:], :][:, ridx], dim=0)
            else:
                tdst = torch.sum(dst[idx, :][:, ridx], dim=0)
            #tdst[tdst == 0] = 1000 #make sure we select affinely independant points #unclear if we want this or not
            nidx = torch.argmin(tdst)
            idx.append(ridx[nidx])
            del ridx[nidx]
    return pts[idx, :], idx

def OrderPointsByMutualFarness(pts: torch.tensor, lidx: int = -1, k: int = -1, N: int = -1, dst: torch.tensor = None, Verbose: bool = False):
    if dst is None: dst = ComputeDistanceMatrix(pts)
    idx = [lidx] if lidx > 0 else [0]
    eidx = N if N > 0 else pts.shape[0]
    ridx = [i for i in range(pts.shape[0])]
    del ridx[idx[0]]
    
    if Verbose:
        for _ in tqdm.tqdm(range(1, eidx)):
            if k > 0:
                tdst = torch.sum(dst[idx[-k:], :][:, ridx], dim=0)
            else:
                tdst = torch.sum(dst[idx, :][:, ridx], dim=0)
            nidx = torch.argmax(tdst)
            idx.append(ridx[nidx])
            del ridx[nidx]
    else:
        for _ in range(1, eidx):
            if k > 0:
                tdst = torch.sum(dst[idx[-k:], :][:, ridx], dim=0)
            else:
                tdst = torch.sum(dst[idx, :][:, ridx], dim=0)
            nidx = torch.argmax(tdst)
            idx.append(ridx[nidx])
            del ridx[nidx]
    return pts[idx, :], idx

def CheckPoints(strPath):
    if ".pkl" in strPath:
        with open(strPath, "rb") as f:
            data = torch.load(f)
    elif ".npy" in strPath:
        with open(strPath, "rb") as f:
            data = torch.tensor(np.load(f))
    cnt = 0
    m = 1000
    for i in range(data.shape[0]):
        #print(data[i:i+1,:])
        if torch.norm(data[i:i+1,:]) != 1:
            print("WARNING!")
        dists = data - data[i:i+1,:]
        dists = dists**2
        dists = torch.sum(dists, axis=1)
        r = torch.argsort(dists)
        #print(dists[r])
        tm = dists[r[1]]
        #print(i, tm)
        if tm == 0: cnt += 1
        if tm < m: m = tm
    print(tm, cnt)
    
def PlotHist(data):
    dst = ComputeDistanceMatrix(data)
    dst_flat = torch.sort(dst[0,:])[0]
    dMin = dst_flat[1] / data.shape[0]
    dMax = dst_flat[-1] / data.shape[0]
    for i in range(1, data.shape[0]):
        temp = torch.sort(dst[i,:])[0]
        dst_flat += temp
        if temp[1] > 2:
            print(i, temp[1])
        dMin += temp[1] / data.shape[0]
        dMax += temp[-1] / data.shape[0]
    dst_flat = dst_flat[dst_flat != 0]
    dst_flat /= data.shape[0]
    print(dMin, dMax)
    hist = np.histogram(dst_flat.numpy(), [i for i in np.arange(0, 4.01, 0.01)])
    #print(hist[1])
    plt.plot([i for i in np.arange(0, 4.0, 0.01)], hist[0])
    return hist[0]
    
def DistanceHistogram(strPath, strPath2 = "None"):
    if ".pkl" in strPath:
        with open(strPath, "rb") as f:
            data = torch.load(f)
    elif ".npy" in strPath:
        with open(strPath, "rb") as f:
            data = torch.tensor(np.load(f))
    
    h1 = PlotHist(data)
    
    if strPath2 != "None":
        if ".pkl" in strPath2:
            with open(strPath2, "rb") as f:
                data2 = torch.load(f)
        elif ".npy" in strPath2:
            with open(strPath2, "rb") as f:
                data2 = torch.tensor(np.load(f))
    
        #plt.figure(2)
        h2 = PlotHist(data2)
        
        print("Error:", np.sum((h2 - h1)**2) / (np.max(h2) * np.max(h1)))
    
    plt.show()
    plt.close()

def NormHistogram(strPath: str, fig: int = -1):
    if ".pkl" in strPath:
        with open(strPath, "rb") as f:
            data = torch.load(f).to('cpu')
    elif ".npy" in strPath:
        with open(strPath, "rb") as f:
            data = torch.tensor(np.load(f)).to('cpu')
    NormHistogram(data, fig)
    return

def NormHistogram(data: torch.tensor, fig: int = -1, line: bool = False, ax = None):
    data = data.to('cpu')
    norms = torch.norm(data, dim=1)
    if fig > 0: plt.figure(fig)
    if line:
        hist = np.histogram(norms.numpy(), bins=100)
        if ax is not None:
            ax.plot(hist[1][:-1], hist[0])
        else:
            plt.plot(hist[1][:-1], hist[0])
    else:
        if ax is not None:
            ax.hist(norms, bins=100)
        else:
            plt.hist(norms, bins=100)
    if fig == -1 and ax is not None:
        plt.show()
        plt.close()
    return

def DPHistogram(data: torch.tensor, fig: int = -1):
    data = data.to('cpu')
    dp = ComputeDPMatrix(data, True)
    if fig > 0: plt.figure(fig)
    plt.hist(dp, bins=100)
    if fig == -1:
        plt.show()
        plt.close()
    return


def IMeasureSeperability(X: torch.tensor, Y: torch.tensor, iBatchSize: int = 2000) -> tuple[float, torch.tensor, torch.nn.Module]:
    X = X.view(X.shape[0], -1).to(device)
    model = LinearHead(X.shape[1], num_classes = Y.shape[1])
    model.to(device)
    Y = torch.argmax(Y.to(device), dim = 1)
    loss = torch.nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=0.001)
    
    best_acc = 0
    
    model.train()
    print("Training SVM on", X.shape[1], "dimensional features")
    for e in tqdm.tqdm(range(50000000 // X.shape[0])): #1k epochs for CIFAR10(0), 10k epochs for CIFAR100S
        #training loop
        rng = np.random.default_rng()
        vecIdx = rng.permutation(X.shape[0])
        for i in range(X.shape[0] // iBatchSize):
            idx = vecIdx[i*iBatchSize:(i+1)*iBatchSize]
            P = model(X[idx,...].to(device))
            l = loss(P, Y[idx,...].to(device))
            opt.zero_grad()
            l.backward()
            opt.step()
        #accuracy check
        acc = 0
        for i in range(X.shape[0] // iBatchSize):
            with torch.no_grad():
                P = model(X[i*iBatchSize:(i+1)*iBatchSize,...].to(device)).detach()
            acc += torch.sum(torch.where(torch.argmax(P, dim=1) == Y[i*iBatchSize:(i+1)*iBatchSize,...].to(device), 1, 0))
        if acc > best_acc:
            best_acc = acc
        

    print("SVM Accuracy: {:.4f}%".format(100 * best_acc / X.shape[0]))
    return (best_acc / X.shape[0]).to("cpu").item(), model
