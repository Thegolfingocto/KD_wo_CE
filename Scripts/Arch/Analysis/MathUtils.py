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
from numba import cuda
import math
import cvxpy as cp
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

#Taken from: https://github.com/minillinim/ellipsoid/blob/master/ellipsoid.py
class EllipsoidTool:
    """Some stuff for playing with ellipsoids"""
    def __init__(self): pass
    
    def getMinVolEllipse(self, P=None, tolerance=0.01):
        """ Find the minimum volume ellipsoid which holds all the points
        
        Based on work by Nima Moshtagh
        http://www.mathworks.com/matlabcentral/fileexchange/9542
        and also by looking at:
        http://cctbx.sourceforge.net/current/python/scitbx.math.minimum_covering_ellipsoid.html
        Which is based on the first reference anyway!
        
        Here, P is a numpy array of N dimensional points like this:
        P = [[x,y,z,...], <-- one point per line
             [x,y,z,...],
             [x,y,z,...]]
        
        Returns:
        (center, radii, rotation)
        
        """
        (N, d) = np.shape(P)
        d = float(d)
    
        # Q will be our working array
        Q = np.vstack([np.copy(P.T), np.ones(N)]) 
        QT = Q.T
        
        # initializations
        err = 1.0 + tolerance
        u = (1.0 / N) * np.ones(N)

        # Khachiyan Algorithm
        while err > tolerance:
            V = np.dot(Q, np.dot(np.diag(u), QT))
            M = np.diag(np.dot(QT , np.dot(np.linalg.inv(V), Q)))    # M the diagonal vector of an NxN matrix
            j = np.argmax(M)
            maximum = M[j]
            step_size = (maximum - d - 1.0) / ((d + 1.0) * (maximum - 1.0))
            new_u = (1.0 - step_size) * u
            new_u[j] += step_size
            err = np.linalg.norm(new_u - u)
            u = new_u

        # center of the ellipse 
        center = np.dot(P.T, u)
    
        # the A matrix for the ellipse
        A = np.linalg.inv(
                       np.dot(P.T, np.dot(np.diag(u), P)) - 
                       np.array([[a * b for b in center] for a in center])
                       ) / d
                       
        # Get the values we'd like to return
        U, s, rotation = np.linalg.svd(A)
        radii = 1.0/np.sqrt(s)
        
        return (center, radii, rotation)

@cuda.jit
def kDistanceMatrix(mtxIn, mtxOut):
    i, j, k = cuda.grid(3)
    if (i < mtxOut.shape[1] and j < mtxOut.shape[1] and k < mtxOut.shape[0]):
        mtxOut[k,i,j] = (mtxIn[i,k] - mtxIn[j,k])
        mtxOut[k,j,i] = -1 * mtxOut[k,i,j]
    return

def GenDistanceMatrixCuda(mtxIn):
    mtxOut = np.zeros((mtxIn.shape[1], mtxIn.shape[0], mtxIn.shape[0]))
    threadsperblock = (16, 16, 2)
    blockspergrid_x = math.ceil(mtxIn.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(mtxIn.shape[0] / threadsperblock[1])
    blockspergrid_z = math.ceil(mtxIn.shape[1] / threadsperblock[2])
    blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)
    kDistanceMatrix[blockspergrid, threadsperblock](mtxIn, mtxOut)
    return mtxOut

def ProjectToHull(x: np.ndarray[np.float64, np.float64], pts: np.ndarray[np.float64, np.float64]):
    w = cp.Variable((pts.shape[0], 1))
    constraints = [
        w <= 1,
        w >= 0,
        cp.sum(w) == 1
    ]
    proj = cp.sum(cp.multiply(w, pts), axis=0)
    obj = cp.sum_squares(x[0,:] - proj)
    
    prob = cp.Problem(cp.Minimize(obj), constraints=constraints)
    prob.solve()
    
    c = w.value
    y = np.sum(np.multiply(c, pts), axis=0)
    return y, c

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



@cuda.jit
def kSymTernaryDPM(A, B, C, R) -> None:
    i, j, k = cuda.grid(3)

    if j <= i or k <= j: return
    if i >= A.shape[0] or j >= B.shape[0] or k >= C.shape[0]: return

    for idx in range(A.shape[1]):
        R[i, j, k] += A[i, idx] * B[j, idx] * C[k, idx]
    
    if R[i, j, k] == 0: R[i, j, k] += 1e-42 #watermark for easier filtering later
    
    # R[i, k, j] = R[i, j, k]
    # R[j, i, k] = R[i, j, k]
    # R[j, k, i] = R[i, j, k]
    # R[k, j, i] = R[i, j, k]
    # R[k, i, j] = R[i, j, k]

    return

def ComputeTernaryDPMatrix(F: torch.tensor) -> torch.tensor:
    iN = F.shape[0]
    nF = cuda.to_device(F.to("cpu").numpy())
    tRet = cuda.to_device(np.zeros((iN, iN, iN)))

    threadsperblock = (8, 8, 8)
    blockspergrid_x = math.ceil(iN / threadsperblock[0])
    blockspergrid_y = math.ceil(iN / threadsperblock[1])
    blockspergrid_z = math.ceil(iN / threadsperblock[2])
    blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)
    kSymTernaryDPM[blockspergrid, threadsperblock](nF, nF, nF, tRet)

    return torch.tensor(tRet.copy_to_host())


#These versions of 3-4-ary DPS can be faster for small sample sizes (to avoid OOM)
def ComputeTernaryDPMatirxTorch(F: torch.tensor) -> torch.tensor:
    iN = F.shape[0]
    F = F.to("cuda")

    A = F.unsqueeze(1).unsqueeze(1)
    A = A.expand((-1, iN, iN, -1))

    B = F.unsqueeze(1).unsqueeze(0)
    B = B.expand((iN, -1, iN, -1))

    C = F.unsqueeze(0).unsqueeze(1)
    C = C.expand((iN, iN, -1, -1))

    return torch.sum(A * B * C, dim = -1).to("cpu")


def ComputeQuadaryDPMatirxTorch(F: torch.tensor) -> torch.tensor:
    iN = F.shape[0]
    F = F.to("cuda")

    A = F.unsqueeze(1).unsqueeze(1).unsqueeze(1)
    A = A.expand((-1, iN, iN, iN, -1))

    B = F.unsqueeze(1).unsqueeze(0).unsqueeze(0)
    B = B.expand((iN, iN, -1, iN, -1))

    C = F.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    C = C.expand((iN, iN, iN, -1, -1))

    D = F.unsqueeze(1).unsqueeze(1).unsqueeze(0)
    D = D.expand((iN, -1, iN, iN, -1))

    return torch.sum(A * B * C * D, dim = -1).to("cpu")


@cuda.jit
def kCellDotProd(nPoints: np.ndarray, nCells: np.ndarray, nRet: np.ndarray, iN: int, iD: int) -> None:
    i, j, k = cuda.grid(3)

    if j <= i: return
    if i >= iN or j >= iN or k >= iD: return
    if not (nCells[i, -1] and nCells[j, -1]): return

    fV = 1.0
    for idx in range(nCells[i, -2]):
        fV *= nPoints[nCells[i, idx], k]

    #iAlpha = nCells[i, -2]
    for idx in range(nCells[j, -2]):
        #if (fV != 0) and fV // nPoints[nCells[j, idx], k] == fV / nPoints[nCells[j, idx], k]: continue #skip if we already multiplied by this one
        fV *= nPoints[nCells[j, idx], k]
        #iAlpha += 1

    #nRet[i, j] += fV
    cuda.atomic.add(nRet, (i, j, 0), fV)
    #cuda.atomic.max(nRet, (i, j, 1), iAlpha)
    cuda.atomic.max(nRet, (i, j, 1), nCells[i, -2] + nCells[j, -2])

    return

def ComputeDPSCC(F: torch.tensor, fEps: float = 1e-2, bRandom: bool = False) -> torch.tensor:
    iN = F.shape[0]
    tF = F.to("cuda")
    tF = tF.view(iN, -1)

    tAF = torch.abs(tF)
    fMax = torch.max(tAF).to("cpu").item()
    fAvg = torch.mean(tAF).to("cpu").item()

    tF = torch.where(tF > 0, 1.0, 0.0) #orthant norm.

    #tF = tF / torch.norm(tF, dim = 1, keepdim = True) #l2
    #tF = tF / torch.sum(torch.abs(tF), dim = 1, keepdim = True) #l1
    #tF = tF / torch.max(torch.abs(tF), dim = 1, keepdim = True)[0] #linf, seems promising?!?
    
    #tF = tF / torch.max(torch.abs(tF)) #global max
    #tF = tF / torch.mean(torch.max(tAF, dim = 1)[0])
    #tF = tF / torch.max(torch.norm(tF, dim = 1))

    nF = cuda.as_cuda_array(tF)
    iD = tF.shape[1]

    idxTriu = torch.triu_indices(iN, iN, 1)

    tBDP = torch.abs(ComputeDPMatrix(tF)).to("cpu")
    tDPS = torch.triu(tBDP, diagonal = 1).view(-1)
    tDPM = tBDP[idxTriu[0], idxTriu[1]]
    print("Binary:", torch.mean(tDPM), torch.max(tDPM))

    _, idx = torch.topk(tDPS, k = iN)
    idxF = torch.where(tDPS > fEps)[0]

    #print(idx.shape, idxF.shape)

    #set intersection
    idx, cnt = torch.cat([idx, idxF]).unique(return_counts = True)
    idx = idx[torch.where(cnt > 1)]

    if bRandom:
        tSampleIdx = torch.randperm(idxF.shape[0])[:iN]
        idx = idxF[tSampleIdx] #randomly choose things that are above the threshold

        #tSampleIdx = torch.randperm(idxTriu[0].shape[0])[:iN]
        #idx = torch.cat((idxTriu[0][tSampleIdx].unsqueeze(1), idxTriu[1][tSampleIdx].unsqueeze(1)), dim = 1).numpy()

    idx = torch.cat(((idx // iN).unsqueeze(1), (idx % iN).unsqueeze(1)), dim=1).to("cpu").numpy()
    print("iCnt:", idx.shape[0])

    nCC = np.zeros((iN, iN + 2), dtype = np.int16)
    nCC[:idx.shape[0], :2] = idx #elements of the cells
    nCC[:idx.shape[0], -2] = 2 #number of elements
    nCC[:idx.shape[0], -1] = 1 #active flag
    nCC = cuda.to_device(nCC)

    nRet = cuda.to_device(np.zeros((iN, iN, 2)))

    threadsperblock = (4, 4, 32)
    blockspergrid_x = math.ceil(iN / threadsperblock[0])
    blockspergrid_y = math.ceil(iN / threadsperblock[1])
    blockspergrid_z = math.ceil(iD / threadsperblock[2])
    blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)

    vecCCs = []
    vecDPMs = [tDPM]
    vecAlphas = [2 * torch.ones_like(vecDPMs[-1])] #storage for per-arity results
    iDepth = 0
    while 1:
        #Compute the "matrix multiplication" over the cells
        kCellDotProd[blockspergrid, threadsperblock](nF, nCC, nRet, iN, iD)
        iDepth += 1
        nCC = nCC.copy_to_host()
        nRet = nRet.copy_to_host()
        #Save the last cell complex
        vecCCs.append(copy.deepcopy(nCC))
        
        #grab the top-N filtering out anything that fell below the threshold
        tBDP = torch.abs(torch.tensor(nRet[:, :, 0]))
        tAlpha = torch.tensor(nRet[:, :, 1])
        tDPS = torch.triu(tBDP, diagonal = 1).view(-1)
        tDPM = tBDP[idxTriu[0], idxTriu[1]]
        tAlpha = tAlpha[idxTriu[0], idxTriu[1]]

        #print(tDPM.shape, tAlpha.shape, torch.min(tAlpha), torch.max(tAlpha))

        vecDPMs.append(copy.deepcopy(tDPM))
        vecAlphas.append(copy.deepcopy(tAlpha))

        print("Depth: {}, Avg. Arity: {}".format(iDepth, torch.mean(tAlpha[tAlpha != 0])), torch.mean(tDPM), torch.max(tDPM))
        _, idx = torch.topk(tDPS, k = iN)
        idxF = torch.where(tDPS > fEps)[0]

        #print(idx.shape, idxF.shape)

        idx, cnt = torch.cat([idx, idxF]).unique(return_counts = True)
        idx = idx[torch.where(cnt > 1)]

        #print(idx.shape)
        if idx.shape[0] == 0: break #stop when nothing is above threshold

        if bRandom:
            tSampleIdx = torch.randperm(idxF.shape[0])[:iN]
            idx = idxF[tSampleIdx]

            #tSampleIdx = torch.randperm(idxTriu[0].shape[0])[:iN]
            #idx = torch.cat((idxTriu[0][tSampleIdx].unsqueeze(1), idxTriu[1][tSampleIdx].unsqueeze(1)), dim = 1).numpy()

        idx = torch.cat(((idx // iN).unsqueeze(1), (idx % iN).unsqueeze(1)), dim=1).numpy()

        nCCNext = np.zeros((iN, iN + 2), dtype = np.int16)
        iCnt = 0 #make sure we haven't ended up with a "clique"
        for i in range(idx.shape[0]):
            idxI = nCC[idx[i, 0], :nCC[idx[i, 0], -2]]
            idxJ = nCC[idx[i, 1], :nCC[idx[i, 1], -2]]

            # print(idxI, idxJ)
            # if idxI.shape[0] == 0 or idxJ.shape[0] == 0:
            #     input()

            idxCell = np.union1d(idxI, idxJ)
            #check for duplicates
            bFound = False
            for j in range(i):
                idxPrev = nCCNext[j, :nCCNext[j, -2]]
                if idxPrev.shape[0] == idxCell.shape[0] and np.sum(np.abs(idxPrev - idxCell)) == 0:
                    bFound = True
                    break

            if bFound: continue

            nCCNext[i, :idxCell.shape[0]] = idxCell
            nCCNext[i, -2] = idxCell.shape[0]
            nCCNext[i, -1] = 1
            iCnt += 1

        print("iCnt:", iCnt)

        if iCnt < 2:
            print("Degenerate CC Reached, Quitting...")
            break
        
        #load the new cell complex onto the GPU, and reset
        #print(nCC[:, :10])
        #print(nCCNext[:, :10])
        nCC = nCCNext
        nCC = cuda.to_device(nCC)
        nRet = cuda.to_device(np.zeros((iN, iN, 2)))

        #input()

    return [torch.tensor(cc, dtype = torch.int32) for cc in vecCCs], vecDPMs, vecAlphas, fMax, fAvg


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


def TrainToThreshold(tModel: torch.nn.Module, X: torch.tensor, YA: torch.tensor, TX: torch.tensor, TYA: torch.tensor,
                     iBS: int = 64, fT: float = 0.1, iMaxE: int = 100) -> dict:
    tModel.to(device)
    tModel.train()
    
    loss = torch.nn.CrossEntropyLoss().to(device)
    opt = torch.optim.AdamW(tModel.parameters(), lr = 0.0001) #TODO: remember me!
    
    timeStart = time.time()
    for iE in tqdm.tqdm(range(iMaxE)):
        rng = np.random.default_rng()
        vecIdx = rng.permutation(X.shape[0])
        
        fLoss = 0
        
        for i in range(X.shape[0] // iBS):
            idx = vecIdx[i*iBS:(i+1)*iBS]
            P = tModel(X[idx,...].to(device))
            l = loss(P, YA[idx,...].to(device))
            fLoss += l.to("cpu").item()
            opt.zero_grad()
            l.backward()
            opt.step()
            
        fLoss /= i
        
        #print(fLoss)
        if fLoss <= fT: break

    fAvgTrainTime = (time.time() - timeStart) / iE
    
    #accuracy checks
    iTBS = iBS #TODO: figure out how this works with usebatchmode
    
    acc = 0
    tModel.eval()

    timeStart = time.time()
    for i in range(X.shape[0] // iTBS):
        with torch.no_grad():
            P = tModel(X[i*iTBS:(i+1)*iTBS,...].to(device)).detach()
        acc += torch.sum(torch.where(torch.argmax(P, dim=1) == YA[i*iTBS:(i+1)*iTBS,...].to(device), 1, 0))
    
    fAvgTestTime = time.time() - timeStart

    tacc = 0
    for i in range(TX.shape[0] // iTBS):
        with torch.no_grad():
            P = tModel(TX[i*iTBS:(i+1)*iTBS,...].to(device)).detach()
        tacc += torch.sum(torch.where(torch.argmax(P, dim=1) == TYA[i*iTBS:(i+1)*iTBS,...].to(device), 1, 0))
        
    return {
        "LossThreshold": fT,
        "FinalLoss": fLoss,
        "EpochsRequired": iE,
        "TrainAccuracy": (acc / X.shape[0]).to("cpu").item(),
        "TestAccuracy": (tacc / TX.shape[0]).to("cpu").item(),
        "GeneralizationGap": (acc / X.shape[0]).to("cpu").item() - (tacc / TX.shape[0]).to("cpu").item(),
        "Params": CountParams(tModel),
        "TrainTime": fAvgTrainTime,
        "TestTime": fAvgTestTime,
    }


def main():
    X = torch.rand((1000, 512))
    tDepth = ComputeDPSCC(X, fEps = 1e-8)
    print(torch.mean(tDepth), torch.max(tDepth))

    # tS = time.time()
    # R = ComputeTernaryDPMatrix(X)
    # tE1 = time.time()

    # print(R.shape)
    # print(tE1 - tS)

    #Rc = ComputeTernaryDPMatirxTorch(X)
    #tE2 = time.time()

    #print(R[0, 0, :])
    #print(Rc[0, 0, :])
    #print(R.shape, Rc.shape)
    #print("Worst Error:", torch.max(torch.abs(Rc - R)))
    #print(tE1 - tS, tE2 - tE1)

    return

if __name__ == "__main__":
    main()