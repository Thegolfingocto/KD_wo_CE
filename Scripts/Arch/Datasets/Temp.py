import datasets
from torchvision.transforms import ToTensor, Normalize, Compose, Resize
import torchvision
#import torchtext
#torchtext.disable_torchtext_deprecation_warning()
#import torchtext.vocab
import torch
import numpy as np
import pandas as pd
from PIL import Image
import io
import pickle
import tqdm
#from sklearn.datasets import fetch_20newsgroups_vectorized
import os
import scipy.io
import re
import string

import nltk
from nltk.corpus import stopwords
#from nltk.stem import PorterStemmer
#from nltk import word_tokenize
#from sklearn.model_selection import train_test_split

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if os.environ["USER"] == "nickubuntu" or os.environ["USER"] == "nickubuntuworkstation":
    strDataDir = "/home/" + os.environ["USER"] + "/Datasets/"
elif os.environ["USER"] == "nicklaptop":
    strDataDir = "/home/nicklaptop/Datasets/"
else:
    strDataDir = "NOT SET UP YET!"

def LoadImagenet():
    trnsNormalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])
    trnsTrain = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(224),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        trnsNormalize,
    ])
    trnsTest = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        trnsNormalize,
    ])
    tdsTrain = torchvision.datasets.ImageFolder(strDataDir + "ILSVRC/Data/CLS-LOC/train", transform = trnsTrain)
    tdsTest = torchvision.datasets.ImageFolder(strDataDir + "ILSVRC/Data/CLS-LOC/val", transform = trnsTest)

    return tdsTrain, tdsTest

def UnpackCIFAR100():
    vecPaths = [strDataDir + "CIFAR100/cifar-100-python/train", strDataDir + "CIFAR100/cifar-100-python/test"]

    if not os.path.isdir(strDataDir + "CIFAR100/Unpacked"):
        os.mkdir(strDataDir + "CIFAR100/Unpacked")
    
    for i in range(len(vecPaths)):
        path = vecPaths[i]
        o = 0 if not i else 50000
        with open(path, 'rb') as f:
            d = pickle.load(f, encoding='bytes')
            data = np.array(d[b'data'])
            for n in tqdm.tqdm(range(data.shape[0])):
                r = data[n, :1024]
                g = data[n, 1024:2048]
                b = data[n, 2048:]
            
                r = r.reshape((32, 32)).astype(np.float64)
                g = g.reshape((32, 32)).astype(np.float64)
                b = b.reshape((32, 32)).astype(np.float64)

                r /= 255
                g /= 255
                b /= 255
    #stats taken from: https://gist.github.com/weiaicunzai/e623931921efefd4c331622c344d8151
    #TODO: sanity check?
                r -= 0.5071
                g -= 0.4865
                b -= 0.4409

                r /= 0.2673
                g /= 0.2564
                b /= 0.2762

                im = np.zeros((3, 32, 32), dtype=np.float64)
                im[0,...] = r
                im[1,...] = g
                im[2,...] = b
                with open(strDataDir + "CIFAR100/Unpacked/" + str(o + n).zfill(6) + ".npz", "wb") as f:
                    np.save(f, im)
                    
    #label unpacking
    coarse_labels = np.zeros((0), np.uint8)
    fine_labels = np.zeros_like(coarse_labels)
    for path in vecPaths:
        with open(path, 'rb') as f:
            d = pickle.load(f, encoding='bytes')
        coarse_labels = np.concatenate((coarse_labels, np.array(d[b'coarse_labels'])))
        fine_labels = np.concatenate((fine_labels, np.array(d[b'fine_labels'])))
    with open(strDataDir + "CIFAR100/Unpacked/" + "coarse_labels.npz", 'wb') as f:
        np.save(f, coarse_labels)
    with open(strDataDir + "CIFAR100/Unpacked/" + "fine_labels.npz", 'wb') as f:
        np.save(f, fine_labels)
        
    return

def UnpackCIFAR10():
    vecPaths = [strDataDir + "CIFAR10/cifar-10-batches-py/data_batch_1", strDataDir + "CIFAR10/cifar-10-batches-py/data_batch_2", 
                strDataDir + "CIFAR10/cifar-10-batches-py/data_batch_3", strDataDir + "CIFAR10/cifar-10-batches-py/data_batch_4", 
                strDataDir + "CIFAR10/cifar-10-batches-py/data_batch_5", strDataDir + "CIFAR10/cifar-10-batches-py/test_batch"]

    if not os.path.isdir(strDataDir + "CIFAR10/Unpacked"):
        os.mkdir(strDataDir + "CIFAR10/Unpacked")
    
    #image unpacking
    o = 0
    for path in vecPaths:
        with open(path, 'rb') as f:
            d = pickle.load(f, encoding='bytes')
        data = np.array(d[b'data'])
        for n in tqdm.tqdm(range(data.shape[0])):
            r = data[n, :1024]
            g = data[n, 1024:2048]
            b = data[n, 2048:]
        
            r = r.reshape((32, 32)).astype(np.float64)
            g = g.reshape((32, 32)).astype(np.float64)
            b = b.reshape((32, 32)).astype(np.float64)

            r /= 255
            g /= 255
            b /= 255
#stats taken from: https://gist.github.com/weiaicunzai/e623931921efefd4c331622c344d8151
#TODO: sanity check?
            r -= 0.4914
            g -= 0.4822
            b -= 0.4465

            r /= 0.2470
            g /= 0.2435
            b /= 0.2616

            #r /= 0.2023
            #g /= 0.1994 
            #b /= 0.2010

            im = np.zeros((3, 32, 32), dtype=np.float64)
            im[0,...] = r
            im[1,...] = g
            im[2,...] = b
            with open(strDataDir + "CIFAR10/Unpacked/" + str(o + n).zfill(6) + ".npz", "wb") as f:
                np.save(f, im)
        o += 10000

    #label unpacking
    labels = np.zeros((0), np.uint8)
    for path in vecPaths:
        with open(path, 'rb') as f:
            d = pickle.load(f, encoding='bytes')
        labels = np.concatenate((labels, np.array(d[b'labels'])))
    with open(strDataDir + "CIFAR10/Unpacked/" + "labels.npz", 'wb') as f:
        np.save(f, labels)
    return

def UnpackEuroSAT() -> None:
    vecDirs = [strDataDir + "EuroSAT_RGB/AnnualCrop/", strDataDir + "EuroSAT_RGB/Forest/", strDataDir + "EuroSAT_RGB/HerbaceousVegetation/",
               strDataDir + "EuroSAT_RGB/Highway/", strDataDir + "EuroSAT_RGB/Industrial/", strDataDir + "EuroSAT_RGB/Pasture/",
               strDataDir + "EuroSAT_RGB/PermanentCrop/", strDataDir + "EuroSAT_RGB/Residential/", strDataDir + "EuroSAT_RGB/River/",
               strDataDir + "EuroSAT_RGB/SeaLake/"]
    
    if not os.path.isdir(strDataDir + "EuroSAT_RGB/Consolidated/"):
        os.mkdir(strDataDir + "EuroSAT_RGB/Consolidated/")

    X = torch.zeros((27000, 3, 64, 64))
    Y = torch.zeros((27000, 10))

    idx = 0
    for i in range(len(vecDirs)):
        strDir = vecDirs[i]
        for f in tqdm.tqdm(os.listdir(strDir)):
            strF = os.fsdecode(f)
            im = Image.open(strDir + strF)
            tIm = torch.tensor(np.array(im))
            X[idx, ...] = torch.permute(tIm, (2, 0, 1))
            Y[idx, i] = 1
            idx += 1

    vecIdx = SplitLabelsByClass(Y)

    XTr = torch.zeros((21600, 3, 64, 64))
    YTr = torch.zeros((21600, 10))
    XTs = torch.zeros((5400, 3, 64, 64))
    YTs = torch.zeros((5400, 10))

    idx = 0
    idx2 = 0
    for i in range(len(vecDirs)):
        sz = vecIdx[i].shape[0]
        XTr[idx:idx + int(sz * 0.8), ...] = X[vecIdx[i][:int(sz * 0.8)], ...]
        YTr[idx:idx + int(sz * 0.8), i] = 1
        XTs[idx2:idx2 + int(sz * 0.2), ...] = X[vecIdx[i][int(sz * 0.8):], ...]
        YTs[idx2:idx2 + int(sz * 0.2), i] = 1
        idx += int(sz * 0.8)
        idx2 += int(sz * 0.2)

    for i in range(3):
        XTr[:,i,:,:] -= torch.mean(X[:,i,:,:])
        XTr[:,i,:,:] /= torch.std(X[:,i,:,:])

        XTs[:,i,:,:] -= torch.mean(X[:,i,:,:])
        XTs[:,i,:,:] /= torch.std(X[:,i,:,:])

    with open(strDataDir + "EuroSAT_RGB/Consolidated/TrainImages.pkl", "wb") as f:
        torch.save(XTr, f)

    with open(strDataDir + "EuroSAT_RGB/Consolidated/TrainLabels.pkl", "wb") as f:
        torch.save(YTr, f)

    with open(strDataDir + "EuroSAT_RGB/Consolidated/TestImages.pkl", "wb") as f:
        torch.save(XTs, f)

    with open(strDataDir + "EuroSAT_RGB/Consolidated/TestLabels.pkl", "wb") as f:
        torch.save(YTs, f)

    return

def LoadEuroSAT() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    with open(strDataDir + "EuroSAT_RGB/Consolidated/TrainImages.pkl", "rb") as f:
        X = torch.load(f)

    with open(strDataDir + "EuroSAT_RGB/Consolidated/TrainLabels.pkl", "rb") as f:
        Y = torch.load(f)

    with open(strDataDir + "EuroSAT_RGB/Consolidated/TestImages.pkl", "rb") as f:
        Xt = torch.load(f)

    with open(strDataDir + "EuroSAT_RGB/Consolidated/TestLabels.pkl", "rb") as f:
        Yt = torch.load(f)

    return X, Y, Xt, Yt

def UnpackCUB200() -> None:
    with open(strDataDir + "CUB_200_2011/train_test_split.txt", "r") as f:
        tSplit = torch.tensor([int(row[-2]) for row in f.readlines()])

    iTr = torch.sum(tSplit)
    iTs = tSplit.shape[0] - iTr

    X = torch.zeros((iTr, 3, 224, 224))
    Y = torch.zeros((iTr, 200))
    Xt = torch.zeros((iTs, 3, 224, 224))
    Yt = torch.zeros((iTs, 200))

    idxTr = 0
    idxTs = 0
    tTransform = Resize((224, 224))

    with open(strDataDir + "CUB_200_2011/images.txt", "r") as f:
        vecPaths = [l[:-1] for l in f.readlines()]

    for i in tqdm.tqdm(range(len(vecPaths))):
        strPath = vecPaths[i].split(" ")[1]
        iCls = int(strPath[:3])
        
        im = Image.open(strDataDir + "CUB_200_2011/images/" + strPath)
        tIm = torch.tensor(np.array(im))
        
        #plt.imshow(tIm)
        #plt.show()
        #plt.close()
        
        if len(tIm.shape) == 2:
            tIm = torch.stack((tIm, tIm, tIm), dim = 2)
        tIm = tTransform(torch.permute(tIm, (2, 0, 1)).unsqueeze(0))
        if tSplit[i]:
            X[idxTr,...] = tIm
            Y[idxTr, iCls - 1] = 1
            idxTr += 1
        else:
            Xt[idxTs,...] = tIm
            Yt[idxTs, iCls - 1] = 1
            idxTs += 1

    if idxTr != iTr or idxTs != iTs:
        print("Something weird happened!")
        print(idxTr, iTr, idxTs, iTs)

    for i in range(3):
        X[:,i,:,:] -= torch.mean(X[:,i,:,:])
        X[:,i,:,:] /= torch.std(X[:,i,:,:])

        Xt[:,i,:,:] -= torch.mean(Xt[:,i,:,:])
        Xt[:,i,:,:] /= torch.std(Xt[:,i,:,:])

    if not os.path.exists(strDataDir + "CUB_200_2011/Consolidated/"): os.mkdir(strDataDir + "CUB_200_2011/Consolidated/")

    with open(strDataDir + "CUB_200_2011/Consolidated/TrainImages.pkl", "wb") as f:
        torch.save(X, f)

    with open(strDataDir + "CUB_200_2011/Consolidated/TrainLabels.pkl", "wb") as f:
        torch.save(Y, f)

    with open(strDataDir + "CUB_200_2011/Consolidated/TestImages.pkl", "wb") as f:
        torch.save(Xt, f)

    with open(strDataDir + "CUB_200_2011/Consolidated/TestLabels.pkl", "wb") as f:
        torch.save(Yt, f)

    return

def LoadCUB_200_2011() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    with open(strDataDir + "CUB_200_2011/Consolidated/TrainImages.pkl", "rb") as f:
        X = torch.load(f)

    with open(strDataDir + "CUB_200_2011/Consolidated/TrainLabels.pkl", "rb") as f:
        Y = torch.load(f)

    with open(strDataDir + "CUB_200_2011/Consolidated/TestImages.pkl", "rb") as f:
        Xt = torch.load(f)

    with open(strDataDir + "CUB_200_2011/Consolidated/TestLabels.pkl", "rb") as f:
        Yt = torch.load(f)

    return X, Y, Xt, Yt

def UnpackStanfordCars() -> None:
    X = torch.zeros((8144, 3, 224, 224))
    Y = torch.zeros((8144, 196))
    Xt = torch.zeros((8041, 3, 224, 224))
    Yt = torch.zeros((8041, 196))

    tTransform = Resize((224, 224))

    L = scipy.io.loadmat(strDataDir + "StanfordCars/cars_train_annos.mat")
    Lt = scipy.io.loadmat(strDataDir + "StanfordCars/cars_test_annos.mat")

    if not os.path.isdir(strDataDir + "StanfordCars/Consolidated/"):
        os.mkdir(strDataDir + "StanfordCars/Consolidated/")

    for i in tqdm.tqdm(range(8144)):
        im = Image.open(strDataDir + "StanfordCars/cars_train/" + str(i + 1).zfill(5) + ".jpg")
        tIm = torch.tensor(np.array(im))
        if len(tIm.shape) == 2:
            tIm = torch.stack((tIm, tIm, tIm), dim = 2)
        tIm = tTransform(torch.permute(tIm, (2, 0, 1)).unsqueeze(0))
        X[i,...] = tIm
        Y[i, L["annotations"][0][i][-2].item() - 1] = 1

    for i in tqdm.tqdm(range(8041)):
        im = Image.open(strDataDir + "StanfordCars/cars_test/" + str(i + 1).zfill(5) + ".jpg")
        tIm = torch.tensor(np.array(im))
        if len(tIm.shape) == 2:
            tIm = torch.stack((tIm, tIm, tIm), dim = 2)
        tIm = tTransform(torch.permute(tIm, (2, 0, 1)).unsqueeze(0))
        Xt[i,...] = tIm
        Yt[i, Lt["annotations"][0][i][-2].item() - 1] = 1

    m = torch.mean(torch.cat((X, Xt), dim = 0))
    sd = torch.std(torch.cat((X, Xt), dim = 0))
    for i in range(3):
        X[:,i,:,:] -= m
        X[:,i,:,:] /= sd

        Xt[:,i,:,:] -= m
        Xt[:,i,:,:] /= sd

    with open(strDataDir + "StanfordCars/Consolidated/TrainImages.pkl", "wb") as f:
        torch.save(X, f)

    with open(strDataDir + "StanfordCars/Consolidated/TrainLabels.pkl", "wb") as f:
        torch.save(Y, f)

    with open(strDataDir + "StanfordCars/Consolidated/TestImages.pkl", "wb") as f:
        torch.save(Xt, f)

    with open(strDataDir + "StanfordCars/Consolidated/TestLabels.pkl", "wb") as f:
        torch.save(Yt, f)

    return

def LoadStanfordCars() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    with open(strDataDir + "StanfordCars/Consolidated/TrainImages.pkl", "rb") as f:
        X = torch.load(f)

    with open(strDataDir + "StanfordCars/Consolidated/TrainLabels.pkl", "rb") as f:
        Y = torch.load(f)

    with open(strDataDir + "StanfordCars/Consolidated/TestImages.pkl", "rb") as f:
        Xt = torch.load(f)

    with open(strDataDir + "StanfordCars/Consolidated/TestLabels.pkl", "rb") as f:
        Yt = torch.load(f)

    return X, Y, Xt, Yt

def LoadStanfordCarsSmall() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    X, Y, Xt, Yt = LoadStanfordCars()
    X = torch.nn.functional.max_pool2d(X, (2, 2), (2, 2))
    Xt = torch.nn.functional.max_pool2d(Xt, (2, 2), (2, 2))

    return X, Y, Xt, Yt

def GroupFineLabels():
    with open(strDataDir + "CIFAR100/Unpacked/fine_labels.npz", "rb") as f:
        fLabels = np.load(f)
    with open(strDataDir + "CIFAR100/Unpacked/coarse_labels.npz", "rb") as f:
        cLabels = np.load(f)
        
    #print(fLabels.shape, cLabels.shape)
    vecLabelIds = [[] for _ in range(20)]
    for i in range(50000):
        if fLabels[i] not in vecLabelIds[cLabels[i]]:
            vecLabelIds[cLabels[i]].append(fLabels[i])
    
    # for i in range(20):    
    #     print(vecLabelIds[i])
        
    return vecLabelIds

def LoadCIFAR10():
    X, Y = LoadCIFAR10Train()
    Xt, Yt = LoadCIFAR10Test()
    return X.to("cpu"), Y.to("cpu"), Xt.to("cpu"), Yt.to("cpu")

def LoadCIFAR100C():
    X, Y = LoadCIFAR100Train(iMode = 0)
    Xt, Yt = LoadCIFAR100Test(iMode = 0)
    return X.to("cpu"), Y.to("cpu"), Xt.to("cpu"), Yt.to("cpu")

def LoadCIFAR100F():
    X, Y = LoadCIFAR100Train(iMode = 1)
    Xt, Yt = LoadCIFAR100Test(iMode = 1)
    return X.to("cpu"), Y.to("cpu"), Xt.to("cpu"), Yt.to("cpu")

def LoadCIFAR100S(iSubset: int = -1):
    if iSubset == -1:
        iSubset = np.random.randint(0, 10)
    elif iSubset < -1 or iSubset > 9:
        print("Error! Subset must be in [0,9]")
    X, Y = LoadCIFAR100SubsetTrain(iSubset)
    Xt, Yt = LoadCIFAR100SubsetTest(iSubset)
    
    return X.to("cpu"), Y.to("cpu"), Xt.to("cpu"), Yt.to("cpu")

def LoadNewsTrain(iDim: int = 100, iCls: int = 20):
    return ILoadNews("train", iDim, iCls)

def LoadNewsTest(iDim: int = 100, iCls: int = 20):
    return ILoadNews("test", iDim, iCls)

def LoadNews(iDim: int = 100, iCls: int = 20):
    X, Y = LoadNewsTrain(iDim, iCls)
    Xt, Yt = LoadNewsTest(iDim, iCls)
    return X.to("cpu"), Y.to("cpu"), Xt.to("cpu"), Yt.to("cpu")

def ILoadNews(strSplit: str = "train", iDim: int = 100, iCls: int = 20, iNumChunks: int = 128):
    if iCls not in [10, 20]:
        print("Invalid number of classes selected! {}".format(iCls))
        return

    #re_url = re.compile(r'(?:http|ftp|https)://(?:[\w_-]+(?:(?:\.[\w_-]+)+))(?:[\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?')
    #re_email = re.compile('(?:[a-z0-9!#$%&\'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&\'*+/=?^_`{|}~-]+)*|"(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21\x23-\x5b\x5d-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])*")@(?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?|\[(?:(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9]))\.){3}(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9])|[a-z0-9-]*[a-z0-9]:(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21-\x5a\x53-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])+)\])')
    
    def clean_header(text):
        text = re.sub(r'(From:\s+[^\n]+\n)', '', text)
        text = re.sub(r'(Subject:[^\n]+\n)', '', text)
        text = re.sub(r'(([\sA-Za-z0-9\-]+)?[A|a]rchive-name:[^\n]+\n)', '', text)
        text = re.sub(r'(Last-modified:[^\n]+\n)', '', text)
        text = re.sub(r'(Version:[^\n]+\n)', '', text)

        return text

    def clean_text(text: str) -> str:        
        text = text.lower()
        text = text.strip()
        text = re.sub(re_url, '', text)
        text = re.sub(re_email, '', text)
        text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
        text = re.sub(r'(\d+)', ' ', text)
        text = re.sub(r'(\s+)', ' ', text)
    
        return text
    
    #nltk.download('stopwords')
    nStopWords = stopwords.words('english')
    #stemmer = PorterStemmer()

    strPath = strDataDir + "20News/20news-bydate-" + strSplit + "/GloVe_" + str(iDim) + "_" + str(iCls) + "Cls_" + str(iNumChunks) + "Chk.pkl"
    #strPath = strDataDir + "20News/20news-bydate-train/Vectorized_10.pkl"
    strLabelPath = strDataDir + "20News/20news-bydate-" + strSplit + "/Labels_" + str(iCls) + "C.pkl"
    if os.path.exists(strPath) and os.path.exists(strLabelPath):
        with open(strPath, "rb") as f:
            X = torch.load(f)
        with open(strLabelPath, "rb") as f:
            Y = torch.load(f)
        return X, Y
    
    print("Generating GloVe embeddings ({}D) for 20News Train".format(iDim))
    X = torch.zeros((0, iNumChunks, iDim))
    Y = torch.zeros((0, iCls))
    G = torchtext.vocab.GloVe(name = "6B", dim = iDim)
    
    if iCls == 20:
        vecCats = ["alt.atheism", "comp.graphics", "comp.os.ms-windows.misc", "comp.sys.ibm.pc.hardware", "comp.sys.mac.hardware", "comp.windows.x", 
                   "misc.forsale", "rec.autos", "rec.motorcycles", "rec.sport.baseball", "rec.sport.hockey", 
                    "sci.crypt", "sci.electronics", "sci.med", "sci.space", "soc.religion.christian", "talk.politics.guns", 
                    "talk.politics.mideast", "talk.politics.misc", "talk.religion.misc"]
    elif iCls == 10:
        vecCats = ["alt.atheism", "comp.sys.ibm.pc.hardware", "misc.forsale", "rec.motorcycles", "rec.sport.hockey", 
                "sci.med", "sci.space", "talk.politics.guns", "talk.politics.misc", "talk.religion.misc"]
    
    for i in range(len(vecCats)):
        strDir = strDataDir + "20News/20news-bydate-" + strSplit + "/" + vecCats[i] + "/"
        for file in tqdm.tqdm(os.listdir(strDir)):
            strF = strDir + os.fsdecode(file)
            try:
                with open(strF, "r") as f:
                    vecL = f.readlines()
            except:
                print("Could not decode file {}".format(strF))
            strD = "".join([l.replace("\n", " ") for l in vecL])
            #print(strD)
            strD = clean_header(strD)
            strD = clean_text(strD)
            #vecT = [stemmer.stem(word) for word in strD.split()]
            vecT = [word for word in strD.split() if word not in nStopWords]
            #print(vecT)
            
            #if len(vecT) < iNumChunks:
                #print("Warning! Document {} is too short ({})! Skipping...".format(strF, len(vecT)))
                #continue
            
            tX = torch.zeros((1, iNumChunks, iDim))
            iChunkSize = (len(vecT) + iNumChunks) // iNumChunks
            #if len(vecT) < iNumChunks: print(len(vecT), iNumChunks, iChunkSize)
            for j in range(len(vecT)):
                t = vecT[j]
                if t not in G.stoi.keys():
                    #print(t)
                    continue
                else:
                    tX[0, j // iChunkSize, :] += G.vectors[G.stoi[t]]
            X = torch.cat((X, tX), dim = 0)
            tY = torch.zeros((1, iCls))
            tY[0, i] = 1
            Y = torch.cat((Y, tY), dim = 0)

    with open(strPath, "wb") as f:
        torch.save(X.to("cpu"), f)
    with open(strLabelPath, "wb") as f:
        torch.save(Y.to("cpu"), f)
    
    return X.to("cpu"), Y.to("cpu")

def LoadDBpediaTrain(iDim: int = 100, iNumChunks: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    return ILoadDBpedia("train", iDim, iNumChunks)

def LoadDBpediaTest(iDim: int = 100, iNumChunks: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    return ILoadDBpedia("test", iDim, iNumChunks)

def LoadDBpedia(iDim: int = 100, iNumChunks: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    X, Y = LoadDBpediaTrain(iDim, iNumChunks)
    Xt, Yt = LoadDBpediaTest(iDim, iNumChunks)
    return X.to("cpu"), Y.to("cpu"), Xt.to("cpu"), Yt.to("cpu")

def ILoadDBpedia(strSplit: str, iDim: int = 100, iNumChunks: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    if strSplit not in ["train", "test"]:
        print("Invalid strSplit: {}".format(strSplit))
        return

    #re_url = re.compile(r'(?:http|ftp|https)://(?:[\w_-]+(?:(?:\.[\w_-]+)+))(?:[\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?')
    #re_email = re.compile('(?:[a-z0-9!#$%&\'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&\'*+/=?^_`{|}~-]+)*|"(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21\x23-\x5b\x5d-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])*")@(?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?|\[(?:(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9]))\.){3}(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9])|[a-z0-9-]*[a-z0-9]:(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21-\x5a\x53-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])+)\])')

    def clean_text(text: str) -> str:        
        text = text.lower()
        text = text.strip()
        text = re.sub(re_url, '', text)
        text = re.sub(re_email, '', text)
        text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)
        text = re.sub(r'(\d+)', ' ', text)
        text = re.sub(r'(\s+)', ' ', text)
    
        return text
    
    strPath = strDataDir + "dbpedia_14/" + strSplit + "_GloVe_" + str(iDim) + "_" + str(iNumChunks) + "Chk.pkl"
    strLabelPath = strDataDir + "dbpedia_14/" + strSplit + "_Labels.pkl"
    if os.path.exists(strPath) and os.path.exists(strLabelPath):
        with open(strPath, "rb") as f:
            X = torch.load(f)
        with open(strLabelPath, "rb") as f:
            Y = torch.load(f)
        return X, Y
    
    print("Generating GloVe embeddings ({}D) for DBpedia".format(iDim))
    strLoad = strDataDir + "dbpedia_14/dbpedia_14/" + strSplit + "-00000-of-00001.parquet"
    df = pd.read_parquet(strLoad)

    X = torch.zeros((df.shape[0], iNumChunks, iDim))
    Y = torch.zeros((df.shape[0], 14))
    G = torchtext.vocab.GloVe(name = "6B", dim = iDim)
    nStopWords = stopwords.words('english')
    idx = 0
    for i in tqdm.tqdm(range(df.shape[0])):
        row = df.iloc[i]
        strText = row["content"]
        strText = "".join([l.replace("\n", " ") for l in strText])
        #print(strText)
        strText = clean_text(strText)
        #print(strText)
        #input()
        vecT = [word for word in strText.split() if word not in nStopWords]

        iChunkSize = (len(vecT) + iNumChunks) // iNumChunks
        cnt = 0
        missed = 0
        for j in range(len(vecT)):    
            t = vecT[j]
            if t not in G.stoi.keys():
                #print("Missed token:", t)
                missed += 1
                continue
            else:
                X[idx, j // iChunkSize, :] += G.vectors[G.stoi[t]]
                cnt += 1
        if missed >= 0.1 * len(vecT):
            #print("Skipping row due to high miss rate")
            continue
        if cnt == 0: 
            #print("WARNING! Found no embeddings!")
            continue
        else:
            X[idx,...] /= iChunkSize
        Y[idx, row["label"]] = 1
        idx += 1

    print("Generated embedding tensors for {} / {} rows".format(idx, df.shape[0]))
    X = X[:idx,...]
    Y = Y[:idx,...]

    with open(strPath, "wb") as f:
        torch.save(X.to("cpu"), f)
    with open(strLabelPath, "wb") as f:
        torch.save(Y.to("cpu"), f)

    return X, Y

def LoadCIFAR10Train():
    X = torch.zeros((50000, 3, 32, 32))
    Y = torch.zeros((50000, 10))
    with open(strDataDir + "CIFAR10/Unpacked/" + "labels.npz", "rb") as f:
        labels = np.load(f)
    for i in tqdm.tqdm(range(50000)):
        with open(strDataDir + "CIFAR10/Unpacked/" + str(i).zfill(6) + ".npz", "rb") as f:
            X[i, :, :, :] = torch.tensor(np.load(f))
        Y[i, labels[i]] = 1
    return X, Y

def LoadCIFAR100Train(iMode: int = 0):
    '''
    iMode ~ {0, 1} := coarse, fine
    '''
    X = torch.zeros((50000, 3, 32, 32))
    Y = torch.zeros((50000, 20)) if not iMode else torch.zeros((50000, 100))
    prefix = "coarse_" if not iMode else "fine_"
    with open(strDataDir + "CIFAR100/Unpacked/" + prefix + "labels.npz", "rb") as f:
        labels = np.load(f)
    for i in tqdm.tqdm(range(50000)):
        with open(strDataDir + "CIFAR100/Unpacked/" + str(i).zfill(6) + ".npz", "rb") as f:
            X[i, :, :, :] = torch.tensor(np.load(f))
        Y[i, labels[i]] = 1
    return X, Y

def LoadCIFAR100SubsetTrain(iSubset: int = -1):
    if iSubset == -1:
        iSubset = np.random.randint(0, 10)
    elif iSubset < -1 or iSubset > 9:
        print("Error! Subset must be in [0,9]")
        
    iC = iSubset % 2
    iF = iSubset % 5
    
    vecLabelsGrouped = GroupFineLabels()
    vecLabels = []
    for i in range(10):
        vecLabels.append(vecLabelsGrouped[2*i + iC][iF])
        
    #print(vecLabels)
    
    X = torch.zeros((5000, 3, 32, 32))
    Y = torch.zeros((5000, 10))
    
    tIdx = torch.zeros((5000), dtype=torch.int32)
    with open(strDataDir + "CIFAR100/Unpacked/fine_labels.npz", "rb") as f:
        fLabels = np.load(f)
    idx = 0
    for i in range(50000):
        if fLabels[i] in vecLabels:
            tIdx[idx] = i
            Y[idx, vecLabels.index(fLabels[i])] = 1
            idx += 1
            
    if idx != 5000:
        print("Error! Could not find 5k images for subset!")
        return
    
    for i in tqdm.tqdm(range(5000)):
        with open(strDataDir + "CIFAR100/Unpacked/" + str(tIdx[i].item()).zfill(6) + ".npz", "rb") as f:
            X[i, :, :, :] = torch.tensor(np.load(f))
            
    return X, Y

def LoadCIFAR100SubsetTest(iSubset: int = -1):
    if iSubset == -1:
        iSubset = np.random.randint(0, 10)
    elif iSubset < -1 or iSubset > 9:
        print("Error! Subset must be in [0,9]")
        
    iC = iSubset % 2
    iF = iSubset % 5
    
    vecLabelsGrouped = GroupFineLabels()
    vecLabels = []
    for i in range(10):
        vecLabels.append(vecLabelsGrouped[2*i + iC][iF])
        
    #print(vecLabels)
    
    X = torch.zeros((1000, 3, 32, 32))
    Y = torch.zeros((1000, 10))
    
    tIdx = torch.zeros((1000), dtype=torch.int32)
    with open(strDataDir + "CIFAR100/Unpacked/fine_labels.npz", "rb") as f:
        fLabels = np.load(f)
    idx = 0
    for i in range(10000):
        if fLabels[50000 + i] in vecLabels:
            tIdx[idx] = 50000 + i
            Y[idx, vecLabels.index(fLabels[50000 + i])] = 1
            idx += 1
            
    if idx != 1000:
        print("Error! Could not find 1k images for subset!")
        return
    
    for i in tqdm.tqdm(range(1000)):
        with open(strDataDir + "CIFAR100/Unpacked/" + str(tIdx[i].item()).zfill(6) + ".npz", "rb") as f:
            X[i, :, :, :] = torch.tensor(np.load(f))
            
    return X, Y

def LoadCIFAR10TrainLabels():
    Y = torch.zeros((50000, 10))
    with open(strDataDir + "CIFAR10/Unpacked/" + "labels.npz", "rb") as f:
        labels = np.load(f)
    for i in tqdm.tqdm(range(50000)):
        Y[i, labels[i]] = 1
    return Y

def LoadCIFAR100TrainLabels(iMode: int = 0):
    '''
    iMode ~ {0, 1} := coarse, fine
    '''
    Y = torch.zeros((50000, 20)) if not iMode else torch.zeros((50000, 100))
    prefix = "coarse_" if not iMode else "fine_"
    with open(strDataDir + "CIFAR100/Unpacked/" + prefix + "labels.npz", "rb") as f:
        labels = np.load(f)
    for i in tqdm.tqdm(range(50000)):
        Y[i, labels[i]] = 1
    return Y

def LoadCIFAR10Test():
    X = torch.zeros((10000, 3, 32, 32))
    Y = torch.zeros((10000, 10))
    with open(strDataDir + "CIFAR10/Unpacked/" + "labels.npz", "rb") as f:
        labels = np.load(f)
    for i in tqdm.tqdm(range(10000)):
        with open(strDataDir + "CIFAR10/Unpacked/" + str(50000 + i).zfill(6) + ".npz", "rb") as f:
            X[i, :, :, :] = torch.tensor(np.load(f))
        Y[i, labels[50000 + i]] = 1
    return X, Y

def LoadCIFAR100Test(iMode: int = 0):
    '''
    iMode ~ {0, 1} := coarse, fine
    '''
    X = torch.zeros((10000, 3, 32, 32))
    Y = torch.zeros((10000, 20)) if not iMode else torch.zeros((50000, 100))
    prefix = "coarse_" if not iMode else "fine_"
    with open(strDataDir + "CIFAR100/Unpacked/" + prefix + "labels.npz", "rb") as f:
        labels = np.load(f)
    for i in tqdm.tqdm(range(10000)):
        with open(strDataDir + "CIFAR100/Unpacked/" + str(50000 + i).zfill(6) + ".npz", "rb") as f:
            X[i, :, :, :] = torch.tensor(np.load(f))
        Y[i, labels[50000 + i]] = 1
    return X, Y

def LoadCIFAR10TrainByClass():
    X, Y = LoadCIFAR10Train()
    ret = []
    idx = [[] for _ in range(10)]
    for i in range(50000):
        idx[torch.argmax(Y[i,:])].append(i)
    for i in range(10):
        ret.append(X[idx[i],...])
    return ret

def SplitTrainFeaturesByClass(X: torch.tensor, Y: torch.tensor) -> list[torch.Tensor]:
    ret = []
    c = Y.shape[1]
    Y = torch.argmax(Y, dim = 1).to("cpu")
    vecIdx = [torch.where(Y == i)[0] for i in range(c)]
    for idx in vecIdx:
        ret.append(X[idx,...])
    return ret

def SplitLabelsByClass(Y: torch.Tensor) ->list[torch.Tensor]:
    c = Y.shape[1]
    Y = torch.argmax(Y, dim = 1).to("cpu")
    vecIdx = [torch.where(Y == i)[0] for i in range(c)]
    return vecIdx

def GetRandomSubset(Y: torch.tensor, iN: int):
    fP = iN / Y.shape[0]
    c = Y.shape[1]
    Y = torch.argmax(Y, dim = 1)
    vecIdx = [torch.where(Y == i)[0] for i in range(c)]
    rng = np.random.default_rng()
    
    ret = torch.zeros((0), dtype=torch.int32)
    for i in range(c):
        Idx = rng.permutation(vecIdx[i].shape[0])[:int(vecIdx[i].shape[0] * fP)]
        ret = torch.cat((ret, vecIdx[i][Idx]), dim = 0)
    return ret

def LoadDataset(iDataset: int, iSubset: int = -1) -> tuple[torch.Tensor, torch.Tensor]:
    if iDataset == 0:
        X, Y = LoadCIFAR10Train()
    elif iDataset == 1:
        X, Y = LoadCIFAR100Train()
    elif iDataset == 2:
        X, Y = LoadCIFAR100SubsetTrain(iSubset = iSubset)
        
    return X, Y

def LoadTinyImagenet():
    X, Y = LoadTinyImagenetTrain()
    Xt, Yt = LoadTinyImagenetTest()
    
    return X, Y, Xt, Yt

def LoadTinyImagenetTrain():
    return ILoadTinyImagenet(strDataDir + "tiny-imagenet/data/train-00000-of-00001-1359597a978bc4fa.parquet")

def LoadTinyImagenetTest():
    return ILoadTinyImagenet(strDataDir + "tiny-imagenet/data/valid-00000-of-00001-70d52db3c749a935.parquet")

def ILoadTinyImagenet(strPath: str):
    df = pd.read_parquet(strPath)
    
    X = torch.zeros((df.shape[0], 3, 64, 64), device=device)
    Y = torch.zeros((df.shape[0], 200), device=device)
    
    cnt = 0
    
    for i in tqdm.tqdm(range(df.shape[0])):
        im = Image.open(io.BytesIO(df.iloc[i]["image"]["bytes"]))
        tim = torch.tensor(np.array(im))
        if len(tim.shape) == 2:
            tim = torch.stack((tim, tim, tim), dim = 2)
            cnt += 1
        tim = torch.permute(tim, (2, 0, 1))
        X[i,...] = tim
        
        Y[i, df.iloc[i]["label"]] = 1
    
    #print("Found {} grayscale images".format(cnt))
    #means: 121.9618, 113.9172, 100.8703
    #stds: 70.2483, 68.1253, 71.3577

    m = torch.tensor([121.9618, 113.9172, 100.8703])
    sd = torch.tensor([70.2483, 68.1253, 71.3577])

    for i in range(3):
        X[:,i,:,:] -= m[i]
        X[:,i,:,:] /= sd[i]
    
    return X.to("cpu"), Y.to("cpu")

def FormatImagenet(strSplit: str = "val") -> None:
    import xml.etree.ElementTree as ET
    strDir = strDataDir + "ILSVRC/Annotations/CLS-LOC/" + strSplit + "/"
    strModifyDir = strDataDir + "ILSVRC/Data/CLS-LOC/" + strSplit + "/"
    for f in tqdm.tqdm(os.listdir(strDir)):
        strF = os.fsdecode(f)
        xmlLabel = ET.parse(strDir + strF).getroot()
        strLabel = xmlLabel[5][0].text
        strPath = xmlLabel[1].text

        strLabelDir = strModifyDir + strLabel + "/"

        if not os.path.exists(strLabelDir): os.mkdir(strLabelDir)
        if os.path.exists(strLabelDir + strPath + ".JPEG"):
            os.rename(strLabelDir + strPath + ".JPEG", strLabelDir + strPath + ".jpeg")
        if not os.path.exists(strModifyDir + strPath + ".JPEG"): continue

        if strPath[0] != "I":
            strPath2 = strPath[8:]
            os.rename(strModifyDir + strPath + ".JPEG", strModifyDir + strPath2 + ".JPEG")

        os.rename(strModifyDir + strPath + ".JPEG", strLabelDir + strPath + ".jpeg")

    return

def temp():
    strDir = strDataDir + "ILSVRC/Data/CLS-LOC/train/"
    for f in tqdm.tqdm(os.listdir(strDir)):
        strDir2 = os.fsdecode(f)
        for f2 in os.listdir(strDir + strDir2):
            strF = os.fsdecode(f2)
            strF2 = strF.split(".")[0] + ".jpeg"
            os.rename(strDir + strDir2 + "/" + strF, strDir + strDir2 + "/" + strF2)

def main():
    #UnpackCIFAR()
    # test = LoadCIFAR10TrainByClass()
    # print(test)
    
    #UnpackCIFAR10()
    #UnpackCIFAR100()
    
    #GroupFineLabels()
    #X, Y = LoadCIFAR100SubsetTrain()
    
    #X, Y = Load10NewsTrain(300)
    #Xt, Yt = Load10NewsTest(300)
    
    #X, Y = LoadTinyImagenetTest()
    
    #print(X.shape, Y.shape)

    #UnpackEuroSAT()

    #UnpackCUB200()

    temp()

    return

if __name__ == "__main__":
   main()