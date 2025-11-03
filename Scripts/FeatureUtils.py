'''
Code written by Nicholas J. Cooper.
Released under the MIT license, see the GitHub page for the full legal boilerplate.
tldr: you freely can do whatever you like with this code, but please cite the GitHub at: https://github.com/Thegolfingocto/KDwoCE.git
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
'''

import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import os
#from cuml.decomposition import PCA
#from qrpca.decomposition import qrpca as qPCA

from Arch.Analysis.MathUtils import *
from Arch.Analysis.MathUtils import IMeasureSeperability
from Arch.Models.Resnet import *
from Arch.Models.ViT import VisionTransformer
from Arch.Models.IModel import IModel
from Arch.Models.ModelUtils import *
from Arch.Models.Misc import LinearHead

#NOTE: https://github.com/chenyaofo/pytorch-cifar-models github link for pre-trained baseline (ResNet 56)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def NormalizeFeatures(FeaturesPath):
    with open(FeaturesPath, "rb") as f:
        X = torch.load(f)
    for i in tqdm.tqdm(range(X.shape[0])):
        X[i,:] = X[i,:] / torch.norm(X[i,:])
        
    OutPath = FeaturesPath.split(".")[0]
    OutPath += "_N1.pkl"
    with open(OutPath, "wb") as f:
        torch.save(X, f)
    return

def IGenerateFeatures(tModel: IModel, X: torch.tensor, Y: torch.tensor, vecFeatureLayers: list[int] = [-1]):
    with torch.no_grad():
        _, vecTF = tModel(X[0:1,...].to(device), vecFeatureLayers = vecFeatureLayers)
        vecTF = [tF.detach().view(tF.shape[0], -1) for tF in vecTF]
        
    vecFeatures = [torch.zeros((X.shape[0], tF.shape[1])) for tF in vecTF]

    #Run the dataset thru the model in chunks to avoid OOM issues
    with torch.no_grad():
        for i in range(X.shape[0] // 2000):
            _, vecF = tModel(X[i*2000:(i+1)*2000,...].to(device), vecFeatureLayers = vecFeatureLayers)
            vecF = [tF.detach().view(tF.shape[0], -1) for tF in vecF]
            for j in range(len(vecFeatures)):
                vecFeatures[j][i*2000:(i+1)*2000,...] = vecF[j]
    return vecFeatures
    
def IGenerateSoftenedLogits(X: torch.tensor, Y: torch.tensor, dTemperature: float = 2):
    _, tLogits, _ = IMeasureSeperability(X, Y)
    tLogits /= dTemperature
    return tLogits
