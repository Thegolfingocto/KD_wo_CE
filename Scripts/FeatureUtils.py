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
from sklearn.decomposition import PCA
#from cuml.decomposition import PCA
#from qrpca.decomposition import qrpca as qPCA

from LoadData import *
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
#print(f"Using device '{device}'")

# Path setup
if os.environ["USER"] == "nickubuntuworkstation" or os.environ["USER"] == "nicklaptop":
    strBasePath = "/home/" + os.environ["USER"] + "/Repos/HSKD/"
elif os.environ["USER"] == "nickubuntu":
    strBasePath = "/home/" + os.environ["USER"] + "/ReposWSL/HSKD/"
else:
    print("NOT SET UP YET!")

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

def GenerateSoftenedLogits(strModel: str = "cifar10_resnet56", dTemperature: float = 2, iDataset: int = 0) -> torch.tensor:
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", strModel, pretrained=True)
    model.to(device)
    model.eval()
    
    if not iDataset:
        X, Y = LoadCIFAR10Train()
    else:
        X, Y = LoadCIFAR100Train()
    
    with torch.no_grad():
        F = model(X)
    
    return IGenerateSoftenedLogits(F, Y, dTemperature)
    
def IGenerateSoftenedLogits(X: torch.tensor, Y: torch.tensor, dTemperature: float = 2):
    _, tLogits, _ = IMeasureSeperability(X, Y)
    tLogits /= dTemperature
    return tLogits

def EvaluateLatentPCA(strModel: str, nComps: List[int], strFeaturePath: str = "None") -> None:
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", strModel, pretrained=True)
    model = model.to(device)
    model.eval()
    
    vecLayers = LinearizeModel(model.children())
    feature_extractor = torch.nn.Sequential(*vecLayers[:-1])
    classifier = vecLayers[-1]
    print(classifier.weight.shape)
    
    if strFeaturePath == "None":
        features = IGenerateFeatures(model=model, iDataset=0)
    else:
        with open(strFeaturePath, "rb") as f:
            features = torch.load(f)
    
    features = features.to("cpu").numpy()
    
    Xt, Yt = LoadCIFAR10Test()
    
    C = torch.argmax(Yt, dim=1)
    
    with torch.no_grad():
        test_features = feature_extractor(Xt)
    print(features.shape, test_features.shape)
    
    test_features = test_features.to("cpu").numpy()
    
    vecPCA = []
    vecEV = []
    vecPredPCA = []
    for nC in nComps:
        vecPCA.append(PCA(n_components = nC))
        vecPCA[-1].fit(features)
        print("Explained Variance:", np.sum(vecPCA[-1].explained_variance_ratio_))
        vecEV.append(np.sum(vecPCA[-1].explained_variance_ratio_))
        temp = vecPCA[-1].inverse_transform(vecPCA[-1].transform(test_features))
        vecPredPCA.append(torch.softmax(classifier(torch.tensor(temp).to(device)), dim=1))
        vecPredPCA[-1] = torch.argmax(vecPredPCA[-1], dim=1)
    
    test_logits = classifier(torch.tensor(test_features).to(device))
    test_preds = torch.argmax(torch.softmax(test_logits, dim=1), dim=1)
    
    acc = torch.sum(torch.where(C == test_preds, 1, 0)) / Xt.shape[0]
    acc_pca = [torch.sum(torch.where(C == vecPredPCA[i], 1, 0)) / Xt.shape[0] for i in range(len(nComps))]
    #print("Original Accuracy: {:.4f}, Accuracy after PCA with {} components: {:.4f}".format(acc * 100 / Yt.shape[0], nComps, acc_pca * 100 / Yt.shape[0]))
    
    plt.plot(nComps, [acc.to("cpu").numpy() for _ in range(len(nComps))], linewidth=2, linestyle="dashed", color="black", label="Original Accuracy")
    plt.plot(nComps, [acc_pca[i].to("cpu").numpy() for i in range(len(nComps))], linewidth=2, label="PCA Corrupted Accuracy")
    plt.plot(nComps, vecEV, linewidth=2, label="Explained Variance")
    plt.plot(nComps, [1 + np.log10(vecEV[i]) for i in range(len(nComps))], linewidth=2, label="1 + Log10 Explained Variance")
    
    plt.legend(fontsize=20)
    
    plt.xlabel("Number of Principal Components", fontsize=18)
    plt.ylabel("Accuracy/Explained Variance as [0,1] Fraction", fontsize=18)
    
    plt.title("ResNet56 Response to PCA Corruption", fontsize=24)
    
    plt.show()
    plt.close()
    
    return

def PlotPerLayerPCACorruption(strModel: str, nComps: List[float], strFeatureDir: str, iStartLayer: int = 0):
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", strModel, pretrained=True)
    model = model.to(device)
    model.eval()
    
    vecLayers = LinearizeModel(model.children())
    
    Xt, Yt = LoadCIFAR10Test()

    cnt = 0
    nL = 0
    for f in os.scandir(strFeatureDir): nL += 1
    imPCAcc = np.zeros((nL, len(nComps)))
    imAcc = np.zeros_like(imPCAcc)
    imEV = np.zeros_like(imPCAcc)
    
    with torch.no_grad():
        preds = torch.argmax(model(Xt), dim=1)
    C = torch.argmax(Yt, dim=1)
    a = torch.sum(torch.where(preds == C, 1, 0)).to("cpu").numpy()
    print(a)
    imAcc += a
    del preds
    
    if os.path.exists(strBasePath + "Results/PerLayerPCA/" + strModel + ".pkl"):
        with open(strBasePath + "Results/PerLayerPCA/" + strModel + ".pkl", "rb") as f:
            imEV = torch.load(f).numpy()
    
    with torch.no_grad():
        for i in range(len(vecLayers)):
            layer = vecLayers[i]
            Xt = layer(Xt)
            print(Xt.shape)
            if IsGeneralizedLayer(layer):
                if cnt < iStartLayer:
                    cnt += 1 
                    continue
                
                features = None
                    
                for j in range(len(nComps)):
                    #Use pre-computed PCA features if they exist
                    if os.path.exists(strFeatureDir + str(cnt) + "_" + str(j) + ".pkl"):
                        with open(strFeatureDir + str(cnt) + "_" + str(j) + ".pkl", "rb") as f:
                            test_features = torch.load(f).to(device)
                    #otherwise, we have to compute the PCA
                    else:
                        if features is None:
                            with open(strFeatureDir + str(cnt) + ".pkl", "rb") as f:
                                features = torch.load(f)
                                
                        nC = nComps[j]
                        for k in range(1, features.dim()): nC *= features.shape[k]
                        nC = int(nC + 1)
                        
                        print("Starting PCA Fit")
                        #pca = qPCA(n_component_ratio=nC, device=device)
                        #pca.fit(features.view(features.shape[0], -1))
                        #test_features = pca.inverse_transform(pca.transform(Xt.view(Xt.shape[0], -1)))
                        
                        pca = PCA(n_components=nC)
                        pca.fit(features.view(features.shape[0], -1).to("cpu").numpy())
                        
                        imEV[cnt, j] = np.sum(pca.explained_variance_ratio_)
                        print("Fit complete")
                        
                        test_features = torch.tensor(pca.inverse_transform(pca.transform(Xt.view(Xt.shape[0], -1).to("cpu").numpy()))).to(device)
                        test_features = torch.reshape(test_features, Xt.shape)
                        print("Transform complete")
                        
                        #Save the result so we don't redo the compute next time
                        with open(strFeatureDir + str(cnt) + "_" + str(j) + ".pkl", "wb") as f:
                            torch.save(test_features.to("cpu"), f)
                        
                        del pca
                    
                    preds = torch.argmax(torch.nn.Sequential(*vecLayers[i+1:])(test_features), dim=1)
                    tacc = torch.sum(torch.where(preds == C, 1, 0))
                    imPCAcc[cnt, j] = tacc
                    print(imEV[cnt, j], tacc * 100 / Xt.shape[0])
                    
                    del test_features
                    del preds
                cnt += 1
    
    with open(strBasePath + "Results/PerLayerPCA/" + strModel + ".pkl", "wb") as f:
        torch.save(torch.tensor(imEV), f)
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    
    imAcc /= Xt.shape[0]
    imPCAcc /= Xt.shape[0]
    
    for i in range(cnt):
        ax.plot(nComps, [i for _ in range(len(nComps))], imAcc[i,:], color="black", linestyle="dashed", linewidth=2)
        ax.plot(nComps, [i for _ in range(len(nComps))], imPCAcc[i,:], color="blue", linewidth=2)
        ax.plot(nComps, [i for _ in range(len(nComps))], imEV[i,:], color="orange", linewidth=2)
        
    plt.show()
    plt.close()
        
    return

def main():
    #GeneratePerLayerFeatures("cifar10_vgg19_bn", strBasePath + "ModelFeatures/VGG19PerLayer/", iDataset = 0)
    #PlotPerLayerPCACorruption("cifar10_vgg19_bn", [0.025, 0.02, 0.015, 0.01, 0.0075, 0.005, 0.0025, 0.001], strBasePath + "ModelFeatures/VGG19PerLayer/", iStartLayer = 2)
    #PlotPerLayerPCACorruption("cifar10_resnet56", [0.5, 0.25, 0.1, 0.05, 0.01], strBasePath + "ModelFeatures/Resnet56PerLayer/")
    #0.5, 0.25, 0.1, 0.05, 
    PrintModelSummary("cifar10_vgg19_bn")
    #EvaluateLatentPCA("cifar10_vgg19_bn", [500, 400, 300, 250, 200, 150, 140, 130, 120, 110, 100, 95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2], strBasePath + "ModelFeatures/vgg19_bn.pkl")
    
    #PrintModelSummary("cifar10_resnet56")
    #EvaluateLatentPCA("cifar10_resnet56", [60, 58, 56, 54, 52, 50, 48, 46, 44, 42, 40, 38, 36, 34, 32, 30, 28, 26, 24, 22, 20, 18, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2], strBasePath + "ModelFeatures/resnet56.pkl")
    
    #ComputePerLayerPCA("cifar10_resnet56")
    
    #features = GenerateFeatures("cifar10_vgg19_bn", bPerClass=True)
    #FeatureStats(features)
    # with open(strBasePath + "ModelFeatures/vgg19_bn.pkl", "wb") as f:
    #     torch.save(features, f)
    
    # logits = GenerateSoftenedLogits("cifar10_resnet56", 2, False)
    # with open(strBasePath + "ModelFeatures/resnet56_logits_T2.pkl", "wb") as f:
    #     torch.save(logits, f)
    
    # features = GenerateFeatures("cifar10_resnet56", iDataset=1)
    # with open(strBasePath + "ModelFeatures/resnet56_cifar10->cifar100C.pkl", "wb") as f:
    #     torch.save(features, f)
    
    # logits = GenerateSoftenedLogits("cifar10_vgg19_bn", dTemperature=2, iDataset=1, bPerClass=False)
    # with open(strBasePath + "ModelFeatures/vgg19_bn_cifar10->cifar100C_logits_T2.pkl", "wb") as f:
    #     torch.save(logits, f)
    
    return

if __name__ == "__main__":
    main()
    