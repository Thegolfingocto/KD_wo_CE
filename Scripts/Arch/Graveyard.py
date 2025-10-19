'''
This is storage for old code that is kept around as a "gadget farm" of sorts.
'''



def ComputeClassDistanceMatrix(strFeaturePath: str, iMode: int = 0):
    '''
    iMode ~ {0, 1, 2} := cifar10, cifar100C, cifar100F
    '''
    with open(strFeaturePath, "rb") as f:
        pts = torch.load(f).to(device)
    
    norms = torch.norm(pts, dim=1)    
    
    dst = ComputeDistanceMatrix(pts, True).to("cuda")
    dst = dst**0.5
    dp = ComputeDPMatrix(pts, True).to("cuda")
    Y = LoadCIFAR10TrainLabels().to("cuda") if not iMode else LoadCIFAR100TrainLabels(iMode - 1)
    Y = torch.argmax(Y, dim=1)
    cs = torch.max(Y) - torch.min(Y) + 1
    cs = cs.to("cpu").item()
    mtx = torch.zeros((cs, cs, 3)).to("cuda")
    mtx2 = torch.zeros((cs, cs, 3)).to("cuda")
    
    #compute intra/inter class variance ratio
    intraDst = 0
    intraDp = 0
    interDst = 0
    interDp = 0
    
    for i in tqdm.tqdm(range(cs)):
        for j in range(i,cs):
            idx1 = torch.where(Y == i)[0]
            idx2 = torch.where(Y == j)[0]
            
            norms1 = norms[idx1]
            norms2 = norms[idx2]
            normMtx = torch.ones((norms1.shape[0], norms2.shape[0]), device=device)
            for c in range(norms2.shape[0]): normMtx[:,c] *= norms1
            for r in range(norms1.shape[0]): normMtx[r,:] *= norms2
            
            tDst = dst[idx1, :][:, idx2]
            tDst /= normMtx
            mtx[i,j,0] = torch.mean(tDst)
            mtx[j,i,0] = mtx[i,j,0]
            mtx[i,j,1] = torch.min(tDst)
            mtx[j,i,1] = mtx[i,j,1]
            mtx[i,j,2] = torch.max(tDst)
            mtx[j,i,2] = mtx[i,j,2]
            
            tDp = dp[idx1, :][:, idx2]
            tDp /= normMtx
            mtx2[i,j,0] = torch.mean(tDp)
            mtx2[j,i,0] = mtx2[i,j,0]
            mtx2[i,j,1] = torch.min(tDp)
            mtx2[j,i,1] = mtx2[i,j,1]
            mtx2[i,j,2] = torch.max(tDp)
            mtx2[j,i,2] = mtx2[i,j,2]
            
            if i == j:
                intraDst += mtx[i,j,0]
                intraDp += mtx2[i,j,0]
            else:
                interDst += 2*mtx[i,j,0]
                interDp += 2*mtx2[i,j,0]
                
    mtx = mtx.to("cpu")
    mtx2 = mtx2.to("cpu")
    
    intraDst /= cs
    intraDp /= cs
    interDst /= (cs**2 - cs)
    interDp /= (cs**2 - cs)
    
    fig, ((ax1, ax2)) = plt.subplots(ncols=2, nrows=1, figsize=(38, 20))
    
    ax1.imshow(mtx[:,:,0])
    
    for i in range(cs):
        for j in range(cs):
            text = str(int(mtx[i, j, 0].item() * 1000) / 1000)
            text2 = "[" + str(int(mtx[i, j, 1].item() * 100) / 100) + "," + str(int(mtx[i, j, 2].item() * 100) / 100) + "]"
            ax1.text(j, i, text, ha="center", va="bottom", color="black", fontsize=18)
            ax1.text(j, i, text2, ha="center", va="top", color="black", fontsize=15)
    
    ax1.set_title("Distance Matrix", fontsize=20)
    
    ax2.imshow(mtx2[:,:,0])
    
    for i in range(cs):
        for j in range(cs):
            text = str(int(mtx2[i, j, 0].item() * 1000) / 1000)
            text2 = "[" + str(int(mtx2[i, j, 1].item() * 100) / 100) + "," + str(int(mtx2[i, j, 2].item() * 100) / 100) + "]"
            ax2.text(j, i, text, ha="center", va="bottom", color="black", fontsize=18)
            ax2.text(j, i, text2, ha="center", va="top", color="black", fontsize=15)
    
    ax2.set_title("Dot-Product Matrix", fontsize=20)
    
    # for i in range(10):
    #     idx = torch.where(Y == i)[0]
    #     NormHistogram(pts[idx,...], ax = ax3, line=True)
    
    # ax3.set_title("Norms per Class", fontsize=20)
    
    fig.suptitle("CIFAR100C Class Distances: VGG19", fontsize=32)
    fig.tight_layout()
    
    print("Average Intra-Class Variance (distance, dot-product): {:.3f}, {:.3f}".format(intraDst, intraDp))
    print("Average Inter-Class Variance (distance, dot-product): {:.3f}, {:.3f}".format(interDst, interDp))
    print("Average Intra/Inter-Class Variance Ratio (distance, dot-product): {:.3f}, {:.3f}".format(intraDst / interDst, intraDp / interDp))
    
    plt.show()
    plt.close()
    return








def ComputeOODistanceMatrix(strIDFeatures: str, strODFeatures: str):
    with open(strIDFeatures, "rb") as f:
        ptsID = torch.load(f).to(device)
    with open(strODFeatures, "rb") as f:
        ptsOD = torch.load(f).to(device)
        
    normsID = torch.norm(ptsID, dim=1)
    normsOD = torch.norm(ptsOD, dim=1)
    
    dst = ComputeDisjointDistanceMatrix(ptsID, ptsOD).to("cuda")
    dst = dst**0.5
    
    dp = ComputeDisjointDPMatrix(ptsID, ptsOD).to("cuda")
    
    YID = LoadCIFAR10TrainLabels()
    YID = torch.argmax(YID, dim=1)
    YOD = LoadCIFAR100TrainLabels(iMode = 0)
    YOD = torch.argmax(YOD, dim=1)
    
    mtx = torch.zeros((10, 20, 3)).to("cuda")
    mtx2 = torch.zeros((10, 20, 3)).to("cuda")
    
    for i in tqdm.tqdm(range(10)):
        for j in range(20):
            idx1 = torch.where(YID == i)[0]
            idx2 = torch.where(YOD == j)[0]
            
            norms1 = normsID[idx1]
            norms2 = normsOD[idx2]
            normMtx = torch.ones((norms1.shape[0], norms2.shape[0]), device=device)
            for c in range(norms2.shape[0]): normMtx[:,c] *= norms1
            for r in range(norms1.shape[0]): normMtx[r,:] *= norms2
            
            tDst = dst[idx1, :][:, idx2]
            tDst /= normMtx
            mtx[i,j,0] += torch.mean(tDst)
            mtx[i,j,1] = torch.min(tDst)
            mtx[i,j,2] = torch.max(tDst)
            
            tDp = dp[idx1, :][:, idx2]
            tDp /= normMtx
            mtx2[i,j,0] += torch.mean(tDp)
            mtx2[i,j,1] = torch.min(tDp)
            mtx2[i,j,2] = torch.max(tDp)

    mtx = mtx.to("cpu")
    mtx2 = mtx2.to("cpu")
    
    fig, ((ax1, ax2)) = plt.subplots(ncols=1, nrows=2, figsize=(40, 20))
    
    ax1.imshow(mtx[:,:,0])
    
    for i in range(10):
        for j in range(20):
            text = str(int(mtx[i, j, 0].item() * 1000) / 1000)
            text2 = "[" + str(int(mtx[i, j, 1].item() * 100) / 100) + "," + str(int(mtx[i, j, 2].item() * 100) / 100) + "]"
            ax1.text(j, i, text, ha="center", va="bottom", color="black", fontsize=18)
            ax1.text(j, i, text2, ha="center", va="top", color="black", fontsize=15)
    
    ax1.set_title("Distance Matrix", fontsize=20)
    
    ax2.imshow(mtx2[:,:,0])
    
    for i in range(10):
        for j in range(20):
            text = str(int(mtx2[i, j, 0].item() * 1000) / 1000)
            text2 = "[" + str(int(mtx2[i, j, 1].item() * 100) / 100) + "," + str(int(mtx2[i, j, 2].item() * 100) / 100) + "]"
            ax2.text(j, i, text, ha="center", va="bottom", color="black", fontsize=18)
            ax2.text(j, i, text2, ha="center", va="top", color="black", fontsize=15)
    
    ax2.set_title("Dot-Product Matrix", fontsize=20)
    
    # for i in range(10):
    #     idx = torch.where(Y == i)[0]
    #     NormHistogram(pts[idx,...], ax = ax3, line=True)
    
    # ax3.set_title("Norms per Class", fontsize=20)
    
    fig.suptitle("CIFAR10->CIFAR100C: VGG19", fontsize=32)
    fig.tight_layout()
    
    plt.show()
    plt.close()
    return









def ComputeAUROC(strIDLogits, strODLogits, iResolution = 100, bShow = True):
    with open(strIDLogits, "rb") as f:
        IDL = torch.load(f)
    IDL = torch.max(torch.softmax(IDL, dim=1), dim=1)[0]
    
    with open(strODLogits, "rb") as f:
        ODL = torch.load(f)
    ODL = torch.max(torch.softmax(ODL, dim=1), dim=1)[0]
    
    strModel = strIDLogits.split("/")[-1].split("_")[0]
    
    X = np.array([i / (iResolution) for i in range(iResolution + 1)])
    TPR = np.zeros((iResolution + 1))
    FPR = np.zeros((iResolution + 1))
    TPRvFPR = np.zeros((iResolution + 1))
    
    TNR = np.zeros((iResolution + 1))
    FNR = np.zeros((iResolution + 1))
    TNRvFNR = np.zeros((iResolution + 1))
    
    for i in range(X.shape[0]):
        t = X[i]
        tpr = torch.sum(torch.where(IDL >= t, 1, 0)) / IDL.shape[0]
        TPR[i] = tpr.to("cpu").item()
        fpr = torch.sum(torch.where(ODL >= t, 1, 0)) / ODL.shape[0]
        FPR[i] = fpr.to("cpu").item()
        tnr = torch.sum(torch.where(ODL < t, 1, 0)) / ODL.shape[0]
        TNR[i] = tnr.to("cpu").item()
        fnr = torch.sum(torch.where(IDL < t, 1, 0)) / IDL.shape[0]
        FNR[i] = fnr.to("cpu").item()
        
    for i in range(X.shape[0]):
        t = X[i]
        idx = np.argmin(np.abs(FPR - t))
        TPRvFPR[i] = TPR[idx]
        
        idx = np.argmin(np.abs(FNR - t))
        TNRvFNR[i] = TNR[idx]
    
    if bShow:
        plt.plot(X, TPR, label="True Positive Rate " + strModel, color='black', linewidth=3)
        plt.plot(X, FPR, label="False Positive Rate " + strModel, color='black', linewidth=3, linestyle="dashed")
        plt.plot(X, TNR, label="True Negative Rate " + strModel, color='red', linewidth=3)
        plt.plot(X, FNR, label="False Negative Rate " + strModel, color='red', linewidth=3, linestyle="dashed")
        plt.plot(X, TPRvFPR, label="TPR vs. FPR " + strModel, color='blue', linewidth=3)
        plt.plot(X, TNRvFNR, label="TNR vs. FNR " + strModel, color='blue', linewidth=3, linestyle="dashed")
        plt.legend(fontsize=18)
        plt.title("TPR and FPR vs. Threshold", fontsize=24)
        plt.show()
        plt.close()
        
    return TPRvFPR, TNRvFNR





def IComputePerLayerStats(model: IModel, X: torch.Tensor, Y: torch.Tensor, vecClassSelection: list[int] = [0], iBatchSize: int = 2000):
    if isinstance(model, VisionTransformer) or X.shape[2] > 32:
        tIdx = GetRandomSubset(Y, 10000)
        X = X[tIdx,...]
        Y = Y[tIdx,...]
    
    #X = X.to(device)
    #Y = Y.to(device)
    
    gIdx = GetRandomSubset(Y, 5000)
    print(gIdx.shape)
    
    #Storage for intra-class stats
    vecExplainedVariance = [[] for _ in range(len(vecClassSelection))]
    vecIntraClassVariance = [[] for _ in range(len(vecClassSelection))]
    vecICVDim = [[] for _ in range(len(vecClassSelection))]
    vecTwoNNID = [[] for _ in range(len(vecClassSelection))]
    vecDMtxDiff = [[0] for _ in range(len(vecClassSelection))]
    vecPrevDMtx = [None for _ in range(len(vecClassSelection))]
    #Storage for inter-class stats
    vecLinSep = []
    vecGlobalEV = []
    vecHostDims = []
    
    #compute global stats
    acc, _ = IMeasureSeperability(X, Y)
    vecLinSep.append(acc)
    T = X.to("cpu")
    T = T.view(T.shape[0], -1)
    gpca = PCA(n_components = min([T.shape[1], T.shape[0], 100]))
    gpca.fit(T.numpy())
    vecGlobalEV.append(gpca.explained_variance_ratio_.tolist())
    vecHostDims.append(T.shape[1])
    
    del gpca
    del T
    
    #compute intra-class stats
    #XC = SplitTrainFeaturesByClass(X, Y)
    for i in range(len(vecClassSelection)):
        cidx = vecClassSelection[i]
        x = X[torch.where(torch.argmax(Y, dim=1) == cidx)[0],...]
        #PCA
        x = x.view(x.shape[0], -1)
        temp = x.to("cpu")
        pca = PCA(n_components = min([temp.shape[1], temp.shape[0], 100]))
        pca.fit(temp.numpy())
        vecExplainedVariance[i].append(pca.explained_variance_ratio_.tolist())
        
        del pca
        del temp
        
        #intra-class distance
        norms = torch.norm(x, dim=1, keepdim=True)
        dmtx = ComputeDistanceMatrix(x / norms, True)
        vecTwoNNID[i].append(ComputeNNID(dmtx))
        dmtx = dmtx.flatten()[1:].view(norms.shape[0] - 1, norms.shape[0] + 1)[:,:-1].reshape(norms.shape[0], norms.shape[0] - 1)
        vecIntraClassVariance[i].append(torch.mean(dmtx).item())
        vecICVDim[i].append(vecIntraClassVariance[i][-1]**2 / (2 * torch.var(dmtx).item()))
        vecPrevDMtx[i] = dmtx
        del dmtx
        del norms
    
    for layer in model.vecLayers:
        #quit out at the classification head
        if isinstance(layer, torch.nn.modules.linear.Linear):
            break
        print(type(layer))
        
        
        #only compute stats for post-activation/block features
        if not IsGeneralizedLayer(layer):
            continue
        
        #compute global stats
        acc, _ = IMeasureSeperability(X, Y)
        vecLinSep.append(acc)
        T = X.to("cpu")
        T = T.view(T.shape[0], -1)
        gpca = PCA(n_components = min([T.shape[1], T.shape[0], 100]))
        if T.shape[1] > 16384:
            gpca.fit(T[gIdx,...].numpy())
        else:
            gpca.fit(T.numpy())
        vecGlobalEV.append(gpca.explained_variance_ratio_.tolist())
        vecHostDims.append(T.shape[1])
        
        del gpca
        del T
        #continue #comment out to just to global stats
        #compute intra-class stats
        #XC = SplitTrainFeaturesByClass(X, Y)
        for i in range(len(vecClassSelection)):
            cidx = vecClassSelection[i]
            x = X[torch.where(torch.argmax(Y, dim=1) == cidx)[0],...]
            #PCA
            x = x.view(x.shape[0], -1)
            temp = x.to("cpu")
            temp = temp
            pca = PCA(n_components = min([temp.shape[1], temp.shape[0], 100]))
            pca.fit(temp.numpy())
            vecExplainedVariance[i].append(pca.explained_variance_ratio_.tolist())
            
            del pca
            del temp
            
            #intra-class distance
            norms = torch.norm(x, dim=1, keepdim=True)
            dmtx = ComputeDistanceMatrix(x / norms, True)
            vecTwoNNID[i].append(ComputeNNID(dmtx))
            dmtx = dmtx.flatten()[1:].view(norms.shape[0] - 1, norms.shape[0] + 1)[:,:-1].reshape(norms.shape[0], norms.shape[0] - 1)
            vecIntraClassVariance[i].append(torch.mean(dmtx).item())
            vecICVDim[i].append(vecIntraClassVariance[i][-1]**2 / (2 * torch.var(dmtx).item()))
            vecDMtxDiff[i].append(torch.nn.functional.mse_loss(vecPrevDMtx[i], dmtx, reduction = "sum").to("cpu").item())
            vecDMtxDiff[i][-1] /= (norms.shape[0]**2 - norms.shape[0])
            vecPrevDMtx[i] = dmtx
            print(vecIntraClassVariance[i][-1], vecICVDim[i][-1], vecDMtxDiff[i][-1])
            del dmtx
            del norms
            
        # if isinstance(layer, ViT.EncoderBlock):
        #     del X
        #     X = XB
        #     del XB
            
    vecHostDims = [vecHostDims[i] / max(vecHostDims) for i in range(len(vecHostDims))]
    
    dRet = {
        "HostDim": vecHostDims,
        "LinearSeparability": vecLinSep,
        "GlobalPCA": vecGlobalEV,
        "ClassPCA": vecExplainedVariance,
        "IntraClassVariance": vecIntraClassVariance,
        "IntrinsicDim": vecICVDim,
        "TwoNNID": vecTwoNNID,
        "DistMtxDiff": vecDMtxDiff,
        "ClassIDs": vecClassSelection,
    }
    
    return dRet

def PlotPerLayerStats(dData: dict, vecEVC: list[int], strTitle: str) -> None:
    return IPlotPerLayerStats(dData["HostDim"], dData["LinearSeparability"], dData["GlobalPCA"], dData["ClassPCA"],
                              dData["IntraClassVariance"], dData["IntrinsicDim"], dData["TwoNNID"], dData["DistMtxDiff"],
                              vecEVC, dData["ClassIDs"], strTitle)

def IPlotPerLayerStats(vecHostDims: list[float], vecLinSep: list[float], vecGlobalEV: list[float], vecClassEV: list[list[float]],
                       vecIntraClassVariance: list[list[float]], vecID: list[list[float]], vecTwoNNID: list[list[float]],
                       vecDstMtxDiff: list[list[float]], vecPCAC: list[int], vecClsIDs: list[int], strTitle: str) -> None:
    L = [i for i in range(-1, len(vecHostDims) - 1)]
    
    fig, axes = plt.subplots(ncols=1 + len(vecClsIDs), nrows=1, figsize=(40 / 1 + len(vecClsIDs), 5))
    
    vecPCAColors = ["darkorange", "green", "yellow", "firebrick", "darkslategrey"]
    
    #plot global stuff first
    axes[0].plot(L, vecHostDims, color="black", linewidth=2, label="Normalized Latent Dimension")
    axes[0].plot(L, vecLinSep, color="blue", linewidth=2, label="Linear SVM Accuracy")
    for j in range(len(vecPCAC)):
        evc = vecPCAC[j]
        axes[0].plot(L, [sum(vecGlobalEV[i][:evc]) for i in range(len(vecGlobalEV))], linewidth=2,
                     color=vecPCAColors[j % len(vecPCAColors)], label="Explained Variance From Top " + str(evc) + " Components")
    
    axes[0].legend(fontsize=18)
    axes[0].set_xlabel("Number of Non-linear activations (VGG) / residual blocks (ResNet)", fontsize=18)
    axes[0].set_ylabel("Explained Variance Ratio / Linear Seperability", fontsize=18)
    axes[0].set_xticks(L)
    axes[0].set_yticks([i/10 for i in range(11)])
    axes[0].grid(color="black", linestyle="dashed", linewidth = 1)
    axes[0].set_title("Dataset Level Stats", fontsize=24)
    
    for i in range(len(vecClsIDs)):
        ax = axes[i+1]
        ax.plot(L, vecIntraClassVariance[i], color="black", linewidth=2, label="Average Intra-Class Variance")
        ax.plot(L, vecID[i], color="blue", linewidth=2, label="Intra-Class Var. Intrinsic Dim")
        ax.plot(L, vecTwoNNID[i], color="blue", linewidth = 2, linestyle = "dashed", label = "TwoNN Intrinsic Dim")
        ax.plot(L, vecDstMtxDiff[i], color="red", linewidth=2, label="Distance Matrix MSE")
        for j in range(len(vecPCAC)):
            evc = vecPCAC[j]
            ax.plot(L, [sum(vecClassEV[i][k][:evc]) for k in range(len(vecClassEV[i]))], linewidth=2,
                        color=vecPCAColors[j % len(vecPCAColors)], label="Explained Variance From Top " + str(evc) + " Components")
            
        ax.legend(fontsize=18)
        ax.set_xlabel("Number of Non-linear activations (VGG) / residual blocks (ResNet)", fontsize=18)
        ax.set_ylabel("E.V Ratio / Avg Intra-Class Distnace / Intrinsic Dim / DMtx MSE", fontsize=18)
        ax.set_xticks(L)
        ax.set_yticks([j/10 for j in range(11)] + [j for j in range(2, 11)])
        ax.grid(color="black", linestyle="dashed", linewidth = 1)
        ax.set_title("Stats for Class " + str(vecClsIDs[i]), fontsize=24)
    
    fig.suptitle("Latent Feature Characteristics: " + strTitle, fontsize=28)
    fig.tight_layout()
    
    plt.show()
    plt.close()
    
    return




def IPlotFeaturePCA(tModel: torch.nn.Module, iDataset: int, iLayer: int, iComps: int, strTitle: str) -> None:
    vecLayers = LinearizeModel(tModel.children())
    X, Y = LoadDataset(iDataset)
    YC = SplitLabelsByClass(Y)
    
    cnt = 0
    iStop = -1
    if iLayer < 0: iStop = iLayer
    for layer in vecLayers[:iStop]:
        #quit out at the classification head
        if isinstance(layer, torch.nn.modules.linear.Linear):
            break
        if cnt == iLayer:
            break
        if IsGeneralizedLayer(layer):
            cnt += 1
        with torch.no_grad():
            X = layer(X)
            
    sPCA = PCA(n_components = iComps)
    
    nX = sPCA.fit_transform(X.to("cpu").numpy())
    print(nX.shape)
    
    if iComps == 2:
        for i in range(Y.shape[1]):
            plt.scatter(nX[YC[i],0], nX[YC[i],1], label = "Class " + str(i+1))
    elif iComps == 3:
        print("TODO")
    else:
        print("Plotting doesn't work that way")
    
    plt.legend(fontsize=20)
    plt.title(strTitle, fontsize=24)
    
    plt.show()
    plt.close()
    
    return