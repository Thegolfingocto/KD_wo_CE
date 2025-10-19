'''
Code written by Nicholas J. Cooper.
Released under the MIT license, see the GitHub page for the full legal boilerplate.
tldr: you freely can do whatever you like with this code, but please cite the GitHub at: https://github.com/Thegolfingocto/KDwoCE.git
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
'''



#library deps
import torch
import tqdm
import os
import copy
import json

#App deps
from LoadData import *
from FeatureUtils import *
from FKD import FKDLoss
#models
from Arch.Models.Resnet import *
from Arch.Models.MobileNetV2 import *
from Arch.Models.ShuffleNetV2 import *
from Arch.Models.VGG import *
from Arch.Models.ModelUtils import *
from Arch.Models.Misc import *
from Arch.Models.IModel import ISeq
#arch
from Arch.Datasets.IDataset import IDataset
from Arch.ITrainer import ITrainer
from Arch.Templates.ImageTrainer import ImageTrainer
from Arch.Utils.Utils import *
from Arch.Logger import *


class KDTrainer(ImageTrainer):
    def __init__(self, strCacheDir: str = "", dConfig: dict = {}, bStartLog: bool = False):
        super().__init__(strCacheDir, dConfig, bStartLog)

        self.YA = None
        
        #extra stuff for KD specifics
        self.fkd: FKDLoss = None
        self.logits: torch.tensor = None
        self.kdTeacher = None
        self.iDetachIdx = None
        self.tAugTransform = None

        self.iBackboneGroupIdx = None
        
        self.logit_crit = None #TODO: create logit loss class or integrate logit stuff into FKDLoss
        
        self.iTestBatchSize = 1000 if self.GetValue("Dataset") not in ["StanfordCars", "CUB_200"] else 256
        
        self.SetEvalMetric(iIdx = 1, bHB = True) #set test accuracy (higher better) as the reference metric for best model tracking
        
        return
    
    def SetTeacher(self, kdT: ITrainer) -> None:
        self.kdTeacher = kdT
        self.kdTeacher.ILoadModel()
        #translate any negative indices for ease of use later
        self.ILoadModel()
        if self.GetValue("FeatureKD") and len(self.GetValue("StudentLayers")) > 0:
            self.vecSLidx = [iL + self.tModel.iGenLayers if iL < 0 else iL for iL in self.GetValue("StudentLayers")]
            self.vecTLidx = [iL + self.kdTeacher.tModel.iGenLayers if iL < 0 else iL for iL in self.GetValue("TeacherLayers")]
            if self.GetValue("LayerMappingMethod") == "PreDefined":
                for i in range(len(self.GetValue("LayerMap"))):
                    for j in range(len(self.GetValue("LayerMap")[i])):
                        if self.GetValue("LayerMap")[i][j] < 0:
                            self.dCfg["LayerMap"][i][j] += self.kdTeacher.tModel.iGenLayers
                            
            #find the final distillation layer
            self.iDetachIdx = max(self.vecSLidx)
            
        if self.GetValue("FeatureKD") and not self.GetValue("StoreFeatures"):
            if self.kdTeacher.GetValue("Model") == "ResNet34" and self.GetValue("Dataset") == "Imagenet":
                self.kdTeacher.ILoadModel()
            else:
                self.kdTeacher.LoadTrainedModel()

        return
    
    def IsFKDStudent(self) -> bool:
        return self.GetValue("Model") != "ResNet56" and self.GetValue("Model") != "VGG19" and self.GetValue("FeatureKD")
    
    def IsTransformer(self) -> bool:
        return "ViT" in self.GetValue("Model") or "TcT" in self.GetValue("Model")
    
    #Print config util for DisplayCache
    def PrintConfigSummary(self, dTempCfg: dict = None) -> None:
        if dTempCfg is None: dTempCfg = self.dCfg
        strTrainMethod = self.GenTrainMethod(dTempCfg)
        printf("Model: {}, Dataset: {}, Training Method: {}".format(dTempCfg["Model"], dTempCfg["Dataset"],
                strTrainMethod), NOTICE)
        if dTempCfg["UseCustomReLU"]:
            printf("GenRelu: {}, {}".format(dTempCfg["Kn"], dTempCfg["Kp"]))
        if "PreTrained" in dTempCfg.keys() and dTempCfg["PreTrained"]:
            strB = "Frozen Backbone" if self.GetValue("BackboneLRMultiplier") == -1 else "Backbone LR Mult: " + str(self.GetValue("BackboneLRMultiplier"))
            printf("Imagenet Pretrained, " + strB)
        if dTempCfg["LRScheduler"] == "OneCycle":
            fLR = dTempCfg["MaxLR"]
        else:
            fLR = dTempCfg["LearningRate"]
        printf("Learning Rate: {}, Scheduler: {}, Optimizer: {}".format(fLR, dTempCfg["LRScheduler"], dTempCfg["Optimizer"]))
        if dTempCfg["LRScheduler"] == "MultiStep":
            printf("Steps: {}, Step Mult: {}".format(dTempCfg["LRSteps"], dTempCfg["LRStepMult"]))
        if dTempCfg.get("DataAugmentation"): printf("W/ Data Augmentation")
        if dTempCfg["FeatureKD"]:
            printf(self.GenFKDMethod(dTempCfg) + "\n" + self.GenFKDLayerString(dTempCfg), INFO)
        if dTempCfg["VanillaKD"]:
            printf("Teacher: {}".format(dTempCfg["Teacher"]), INFO)
            printf("Logit Temperature: {}".format(dTempCfg["Temperature"]), INFO)
            if dTempCfg["NormalizeLogits"]: printf("W/ Logit Normalization", INFO)
        printf("Runs: {}, Epochs: {}".format(dTempCfg["NumRuns"], dTempCfg["NumEpochs"]), NOTICE)
        return
    
    def GenTrainMethod(self, dCfg: dict) -> str:
        strTrainMethod = ""
        if dCfg["FeatureKD"]:
            strTrainMethod += "FKD"
            if dCfg["SeperateLinear"]: strTrainMethod += "->[Detach]"
            if dCfg["UseTeacherClassifier"]: strTrainMethod += "->[T-Cls"
            if dCfg["LearnTeacherClassifier"]: strTrainMethod += ", Learned]"
            else: strTrainMethod += "]"
        if dCfg["VanillaKD"]: strTrainMethod += "VKD" if len(strTrainMethod) == 0 else " + VKD"
        if dCfg["UseCELoss"]: strTrainMethod += "Standard CE" if len(strTrainMethod) == 0 else " + CE"
        
        return strTrainMethod

    def GenFKDMethod(self, dCfg: dict = None) -> str:
        if dCfg is None: dCfg = self.dCfg
        strFKDMethod = "Teacher: " + dCfg["Teacher"] + ", Method: " + dCfg["ProjectionMethod"] + " " + dCfg["LayerMappingMethod"]
        if dCfg["ProjectionMethod"] == "PCA":
            strFKDMethod += "\nLMM: "
            strFKDMethod += str(dCfg["LandmarkMethod"])
            strFKDMethod += " LMS: "
            strFKDMethod += str(dCfg["LandmarkPostScaleFactor"])
        elif dCfg["ProjectionMethod"] == "LearnedProjector":
            strFKDMethod += "\nProj. Architecture: "
            strFKDMethod += str(dCfg["ProjectorArchitecture"])
            strFKDMethod += "\nPool Mode: " + str(dCfg["PoolMode"])
        return strFKDMethod
    
    def GenFKDLayerString(self, dCfg: dict = None) -> str:
        if dCfg is None: dCfg = self.dCfg
        strLayers = "TL:"
        strLayers += str(dCfg["TeacherLayers"])
        strLayers += "\nSL:"
        strLayers += str(dCfg["StudentLayers"])
        return strLayers
    
    def DisplayResultCB(self, strFolder: str) -> None:
        if os.path.exists(strFolder + "Result.json"):
            with open(strFolder + "Result.json", "r") as f:
                dR = json.load(f)
            print("Max Test Accuracy: {:.2f}".format(100 * dR["MaxTestAcc"]))
        return
    
    #-------------------Impls of load methods-------------------#
    
    def LoadLossFcn(self) -> any:
        if self.GetValue("VanillaKD"):
            self.logit_crit = KLDiv()
        
        return torch.nn.CrossEntropyLoss()
    
    #-----------------App specific methods for FKD-------------------#
    #LoadComponents is the generic interface for extra component management 
    def LoadComponents(self) -> None:
        self.RunUsageChecks()
        if self.GetValue("FeatureKD"):
            if self.fkd is None:
                self.SetupFeatureKD()
            else:
                self.SetupFKDModules()
        if self.GetValue("VanillaKD") and self.GetValue("Dataset") != "Imagenet":
            self.LoadLogits()

        return
    
    def RunUsageChecks(self) -> None:
        #use-case specific logic check
        if not self.GetValue("FeatureKD") and self.GetValue("SeperateLinear"):
            printf("Config setup without FKD loss AND detached classification head!", WARNING)
            if GetInput("Do you want to re-attach the head? (Y/X)"):
                self.dCfg["SeperateLinear"] = False
                
        if self.GetValue("FeatureKD") and self.GetValue("Dataset") == "TinyImagenet" and self.GetValue("StoreFeatures") and not self.kdTeacher.IsTransformer():
            printf("StoreFeatures mode may not work with tiny imagenet!")
            if GetInput("Disable? (Y/X)"):
                self.dCfg["StoreFeatures"] = False

        if (self.GetValue("FeatureKD") or self.GetValue("VanillaKD")) and (self.GetValue("DataAugmentation") and 
                                                                           self.GetValue("AugmentTeacher") and self.GetValue("StoreFeatures")):
            printf("Must Disable StoreFeatures mode for AugmentTeacher!", WARNING)
            if GetInput("Disable? (Y/X)"):
                self.dCfg["StoreFeatures"] = False
                
        return
    
    def GenFeatureStringBase(self, bTeacher: bool = True) -> str:
        if self.kdTeacher is not None and bTeacher:
            strN = self.kdTeacher.GetValue("Model")
            strN += "_"
            strN += self.kdTeacher.GetValue("Dataset")
        else:
            strN = self.GetValue("Model")
            strN += "_"
            strN += self.GetValue("Dataset")

        return strN

    def GenFeatureString(self, iL: int, bTeacher: bool = True) -> str:
        strN = self.GenFeatureStringBase(bTeacher = bTeacher)
        strN += "_L"
        strN += str(iL)
        return strN
    
    def GenFeatureStringLabel(self, bTeacher: bool = True) -> str:
        strN = self.GenFeatureStringBase(bTeacher = bTeacher)
        strN += "_Y"
        return strN
    
    def GenLogitPath(self) -> str:
        strPath = self.GetCurrentFolder() + "Features/"
        strPath += self.GetValue("Model") + "_"
        strPath += self.GetValue("Dataset") + "_Logits.pkl"
        return strPath
    
    def GenPerLayerFeatures(self, vecLt: list[int] = None, iBatchSize: int = 500) -> None:
        self.LoadTrainedModel()
        self.tModel.eval()

        vecL = copy.deepcopy(vecLt) if vecLt is not None else [i for i in range(self.tModel.iGenLayers)]

        strDir = self.GetCurrentFolder() + "Features/"
        if not os.path.isdir(strDir):
            os.mkdir(strDir)

        if not os.path.isdir("./TensorCache/"):
            os.mkdir("./TensorCache/")
        if len(os.listdir("./TensorCache/")) > 0:
            printf("TensorCache is not empty! Continuing to generate features may erase any leftovers.", WARNING)
            if not GetInput("Overwrite? (Y/X)"): return
            for f in os.listdir("./TensorCache/"):
                strF = "./TensorCache/" + os.fsdecode(f)
                os.remove(strF)
        
        vecR = []
        for iL in vecL:
            strPath = strDir + self.GenFeatureString(iL, bTeacher = False) + ".pkl"
            #print(strPath)
            if os.path.exists(strPath):
                vecR.append(iL)
        for iR in vecR: vecL.remove(iR)

        strLabelPath = strDir + self.GenFeatureStringLabel(bTeacher = False) + ".pkl"
                
        if len(vecL) == 0 and os.path.exists(strLabelPath): return

        if not self.dsData.Loaded(): self.ILoadDataset() #only load dataset if necessary

        bDisableFirstPool = False

        if self.GetValue("Dataset") == "TinyImagenet" and not self.IsTransformer(): #REMEMBER ME!
            X, Y = self.dsData.GetRandomSubset(10000, strSplit = "train")
        elif self.GetValue("Dataset") == "StanfordCars" and not self.IsTransformer():
            X, Y = self.dsData.GetRandomSubset(6000, strSplit = "train")
        elif self.GetValue("Dataset") == "Imagenet":
            X, Y = self.dsData.GetRandomSubset(5000, strSplit = "train")
        else:
            X, Y = self.dsData.X, self.dsData.Y

        torch.save(Y, strLabelPath)

        if len(vecL) == 0: return

        cnt = 0
        
        iN = X.shape[0]
        iL = iN // iBatchSize
        if iN % iBatchSize != 0: iL += 1
        vecX = []
        for i in range(iL):
            iE = (i+1)*iBatchSize
            if iE >= iN: iE = iN
            vecX.append(X[i*iBatchSize:iE,...])

        print("Generating missing features from layers: {}".format(vecL))
        #input()
        bFP = True

        if vecL[0] > 0:
            #skip ahead to avoid OOM errors in some fringe cases
            with torch.no_grad():
                for i in tqdm.tqdm(range(iL)):
                    for layer in self.tModel.vecLayers[:self.tModel.mapGenIdxToLiteralIdx[vecL[0]]]:
                        vecX[i] = layer(vecX[i].to(device))
                        if i == 0 and IsGeneralizedLayer(layer): cnt += 1
                    vecX[i] = vecX[i].to("cpu")
            iStartIdx = self.tModel.mapGenIdxToLiteralIdx[vecL[0]]
        else: iStartIdx = 0

        for layer in self.tModel.vecLayers[iStartIdx:]:
            if cnt > max(vecL): break
            
            if bDisableFirstPool and "pool" in layer.__class__.__name__.lower() and bFP:
                bFP = False
                continue

            with torch.no_grad():
                for i in tqdm.tqdm(range(iL)):
                    vecX[i] = layer(vecX[i].to(device)).to("cpu")

            if IsGeneralizedLayer(layer):
                if cnt in vecL:
                    #extremely jank saving scheme to avoid pytorch's unavoidable memcpy when using the cat() function
                    #save the representations batch-wise
                    print("Saving batched tensor")
                    sz = vecX[0].shape
                    for i in range(iL):
                        with open("./TensorCache/" + "temp" + str(i) + ".pkl", "wb") as f:
                            torch.save(vecX[i], f)
                    #delete this copy
                    del vecX
                    #now allocate a contiguous tensor and reload from disk
                    sz = list(sz)
                    sz[0] = iN
                    X = torch.zeros(sz)
                    print("Allocated contiguous memory for tensor")
                    for i in range(iL):
                        with open("./TensorCache/" + "temp" + str(i) + ".pkl", "rb") as f:
                            iE = (i+1)*iBatchSize
                            if iE >= iN: iE = iN
                            X[i*iBatchSize:iE,...] = torch.load(f) #yes, this way is actually MUCH more memory efficient than cat()
                    print("Loaded batched tensor into contiguous memory")
                    #now save it again in the correct place
                    with open(strDir + self.GenFeatureString(cnt, bTeacher = False) + ".pkl", "wb") as f:
                        torch.save(X.to("cpu"), f)
                    print("Saved contiguous tensor")
                    #delete the contiguous version and reload the batched versions to continue the forward prop.
                    del X
                    vecX = []
                    for i in range(iL):
                        with open("./TensorCache/" + "temp" + str(i) + ".pkl", "rb") as f:
                            vecX.append(torch.load(f))
                        #now we can remove the temporary file
                        os.remove("./TensorCache/" + "temp" + str(i) + ".pkl")
                    print("Reloaded batched tensor")
                cnt += 1
        
        del vecL
        
        return
    
    def GenSoftenedLogits(self, iBatchSize: int = 500) -> None:
        strFeatureDir = self.GetCurrentFolder() + "Features/"
        if not os.path.isdir(strFeatureDir):
            os.mkdir(strFeatureDir)

        strLogitPath = self.GenLogitPath()
        if os.path.exists(strLogitPath):
            print("Warning! Found existing logits: ", strLogitPath)
            if not GetInput("Overwrite? (Y/X)"):
                return

        self.LoadTrainedModel()
        self.tModel.eval()
        if not self.dsData.Loaded(): self.ILoadDataset()

        print("Generating Logits: ", strLogitPath)
        iN = (self.dsData.Size("train") // self.dsData.iBatchSize) + 1
        tLogits = torch.zeros((self.dsData.Size("train"), self.dsData.Classes()))

        with torch.no_grad():
            for i in tqdm.tqdm(range(iN)):
                iE = min([(i+1) * self.dsData.iBatchSize, tLogits.shape[0]])
                idx = torch.arange(i*self.dsData.iBatchSize, iE, 1)
                x, _ = self.dsData.GetSamples(idx, "train")
                tL = self.tModel(x.to(device))
                tLogits[i*self.dsData.iBatchSize:iE,...] = tL.to("cpu")

        with open(strLogitPath, "wb") as f:
            torch.save(tLogits, f)

        return

    def LoadLogits(self) -> None:
        if self.kdTeacher is None:
            printf("Attempted to load logits but teacher model has not been set!", ERROR)
            return
        
        printf("Loading Logits", INFO)
        strLogitPath = self.kdTeacher.GenLogitPath()

        if not os.path.exists(strLogitPath):
            self.kdTeacher.GenSoftenedLogits()

        with open(strLogitPath, "rb") as f:
            self.logits = torch.load(f).to(device)
            
        return
    
    def SetupFeatureKD(self) -> None:
        if self.kdTeacher is None:
            printf("Attempted to load latent targets but teacher model has not been set!", ERROR)
            return
        
        printf("Setting up FKD Loss", INFO)
        
        with torch.no_grad():
            #Get sample features so FKD knows about dimensions
            test = torch.zeros(([1] + self.dsData.Shape())).to(device)
            _, vecSF = self.tModel(test, vecFeatureLayers = self.vecSLidx)
            #_, vecSF = self.tModel(torch.zeros((1, 3, 224, 224)).to(device), vecFeatureLayers = self.vecSLidx)
            vecSF = [tF.detach() for tF in vecSF]
            #swap hidden dim and #tokens 
            if self.IsTransformer():
                for i in range(len(vecSF)):
                    if len(vecSF[i].shape) == 3: vecSF[i] = torch.permute(vecSF[i], (0, 2, 1))
            
            _, vecTFSt = self.kdTeacher.tModel(test, vecFeatureLayers = self.vecTLidx)
            #_, vecTFSt = self.kdTeacher.tModel(torch.zeros((1, 3, 224, 224)).to(device), vecFeatureLayers = self.vecTLidx)
            if self.kdTeacher.IsTransformer():
                for i in range(len(vecTFSt)):
                    if len(vecTFSt[i].shape) == 3: vecTFSt[i] = torch.permute(vecTFSt[i], (0, 2, 1))

            vecTFS = [list(tF.detach().shape) for tF in vecTFSt]

            for i in range(len(vecTFS)):
                vecTFS[i][0] = self.dsData.Size()
                if self.GetValue("Dataset") == "TinyImagenet" and not self.kdTeacher.IsTransformer() and 0: #TODO HACK WARNING WARNING!!!1!
                    vecTFS[i][0] //= 2
        
        #Get teacher features for the full dataset. TODO: add switching logic for subsets of larger datasets
        #TODO: disable this section for some Proj. methods and non-store features mode
        vecTF = [None for _ in range(len(self.GetValue("TeacherLayers")))]
        
        if self.GetValue("ProjectionMethod") == "PCA" or self.GetValue("StoreFeatures"):
            self.kdTeacher.GenPerLayerFeatures(self.vecTLidx) #generate the features
        
            #load the features
            for i in range(len(self.vecTLidx)):
                iTL = self.vecTLidx[i]
                strFeaturePath = self.kdTeacher.GetCurrentFolder() + "Features/" + self.GenFeatureString(iTL) + ".pkl"
                if os.path.exists(strFeaturePath):
                    with open(strFeaturePath, "rb") as f:
                        vecTF[i] = torch.load(f)
                    printf("Loaded features from layer {}".format(iTL))
                else:
                    print("Feature Generation Failed!")
                    return   
                
            #reshape the teacher's features to the correct size
            vecTF = [vecTF[i].view(vecTFS[i]) for i in range(len(self.vecTLidx))]
        else:
            vecTF = vecTFSt
        
        #setup the FKDLoss class
        self.fkd = FKDLoss(self.vecSLidx, self.vecTLidx, vecSF, vecTF, self.GetValue("LayerMappingMethod"), self.GetValue("ProjectionMethod"), 
                           self.GetValue("BatchSize"), self.GetValue("StudentPartitions"), self.GetValue("TeacherPartitions"),
                           self.GetValue("LayerMap"), self.GetValue("LayerMapWeights"), bStoreFeatures = self.GetValue("StoreFeatures"), 
                           vecPowers = self.GetValue("TeacherPowers"))
        
        #PCA init
        if self.GetValue("ProjectionMethod") == "PCA":
            if self.GetValue("Dataset") != "TinyImagenet":
                self.fkd.LoadPCA(self.kdTeacher.GetCurrentFolder(), self.dlTrain[1], self.GetValue("UseGlobalCenter"))
            else:
                #GetRandomSubset returns a subset organized in class order
                cIdx = torch.zeros((50000, 200))
                for ic in range(200):
                    cIdx[250*ic:250*(ic+1),ic] = 1
                self.fkd.bGlobalOnly = True
                self.fkd.LoadPCA(self.kdTeacher.GetCurrentFolder(), cIdx, self.GetValue("UseGlobalCenter"))
        
        self.SetupFKDModules() #setup modules
        
        return
    
    def SetupFKDModules(self) -> None:
        #logic for attention mode
        if self.GetValue("LayerMappingMethod") == "LearnedAttn":
            #setup the other modules we need
            self.fkd.SetAttnModule(NLayerMLP, self.GetValue("AttnArchitecture"))
            
            if self.dModules is None:
                self.dModules = torch.nn.ModuleDict()
            self.dModules["LayerAttn"] = self.fkd.GetAttnModule() #hook this into the main optimizer
        
        #logic for relation mode
        if self.GetValue("ProjectionMethod") == "RelationFunction":
            if self.GetValue("RelationFunction") == "DotProd":
                self.fkd.SetRelation(ComputeNormalizedDPMatrix) #hello!
            elif self.GetValue("RelationFunction") == "Distance":
                self.fkd.SetRelation(ComputeDistanceMatrix)
            else:
                printf("ERROR! Invalid RelationFunction {}".format(self.GetValue("RelationFunction")), ERROR)

        #logic for projectors
        if self.GetValue("ProjectionMethod") == "LearnedProjector":
            iDim = 2
            if self.IsTransformer() or self.kdTeacher.IsTransformer(): iDim = 1

            if self.GetValue("ProjectorArchitecture") == "ThreeLayerConv":
                fM = ThreeLayerConv2D if iDim == 2 else ThreeLayerConv1D
            elif self.GetValue("ProjectorArchitecture") == "SingleLayerConv":
                fM = SingleLayerConv2D if iDim == 2 else SingleLayerConv1D
            #Add other projectors here
            else:
                printf("Error! Invalid projector {}".format(self.GetValue("ProjectorArchitecture")), ERROR)
            
            self.fkd.SetProjector(fM, iDim = iDim, bPoolMode = self.GetValue("PoolMode"), bLearned = self.GetValue("LearnProjectors")) #fkd makes copies under the hood
            
            if self.dModules is None:
                self.dModules = torch.nn.ModuleDict()
            self.dModules["Projector"] = self.fkd.GetProjector() #hook things into the main optimizer

            if self.GetValue("UseTeacherClassifier"):
                self.dModules["TeacherClassifier"] = ISeq(self.kdTeacher.tModel.vecLayers[self.kdTeacher.tModel.mapGenIdxToLiteralIdx[self.vecTLidx[-1]]:])
            
        return

    def Eval(self, dsData: IDataset = None) -> dict:
        if dsData is None:
            if not self.dsData.Loaded(): self.ILoadDataset()
            dsData = self.dsData #this is here for easy eval on OOD datasets

        if self.tLossFcn is None: self.ILoadLossFcn()
        
        #put model in test mode
        self.tModel.eval()

        with torch.no_grad():
            acc = 0
            avg_loss = 0
            iN = dsData.Size("test") // dsData.iBatchSize

            for i in tqdm.tqdm(range(iN)):
                idx = torch.arange(i*dsData.iBatchSize, (i+1)*dsData.iBatchSize, 1)
                x, y = dsData.GetSamples(idx, "test")
                x = x.to(self.device)
                y = y.to(self.device)

                if self.GetValue("FeatureKD") and self.GetValue("UseTeacherClassifier"):
                    xp, _ = self.tModel(x, vecFeatureLayers = self.vecSLidx, bEarlyQuit = True, bNaNCheck = False)
                    if self.kdTeacher.IsTransformer(): xp = torch.permute(xp, (0, 2, 1))
                    xp = self.fkd.tProj.inference(xp)
                    if self.kdTeacher.IsTransformer(): xp = torch.permute(xp, (0, 2, 1))
                    y_hat = self.dModules["TeacherClassifier"](xp)
                else:
                    y_hat = self.tModel(x)
                avg_loss += self.tLossFcn(y_hat, y).item()# * self.iTestBatchSize
                acc += torch.sum(torch.where(torch.argmax(y_hat, dim = 1) == y, 1, 0)).item()

        return {
                "Test Loss": avg_loss / iN,
                "Test Acc": acc / (iN * dsData.iBatchSize)
                }
    
    #main train function impl
    def Train(self) -> tuple[float, float, float, float, float]:
        self.tModel.train()
        if self.kdTeacher is not None and not self.GetValue("StoreFeatures"): self.kdTeacher.tModel.eval()
        ce_loss = 0
        lg_loss = 0
        ls_loss = 0
        ls_Mloss = 0
        ls_GradDP = 0
        if self.GetValue("FeatureKD") and self.GetValue("LayerMappingMethod") == "LearnedAttn":
            tMap = torch.zeros((self.fkd.iSL, self.fkd.iTL))

        iN = self.dsData.Size() // self.dsData.iBatchSize
        vecIdx = torch.randperm(self.dsData.Size())

        for i in tqdm.tqdm(range(iN)):
            idx = vecIdx[i*self.dsData.iBatchSize:(i+1)*self.dsData.iBatchSize]
            x, y = self.dsData.GetSamples(idx, "train")
            
            x = x.to(self.device)
            y = y.to(self.device)

            if self.GetValue("FeatureKD"):
                xp, vecF = self.tModel(x, vecFeatureLayers = self.vecSLidx, bEarlyQuit = True, bNaNCheck = False)
                
                if self.GetValue("SeperateLinear"):
                    xp = xp.detach()
                
                if self.GetValue("UseTeacherClassifier"):
                    if self.kdTeacher.IsTransformer(): xp = torch.permute(xp, (0, 2, 1))
                    xp = self.fkd.tProj.inference(xp)
                    if self.kdTeacher.IsTransformer(): xp = torch.permute(xp, (0, 2, 1))
                    if self.GetValue("LearnTeacherClassifier"):
                        y_hat = self.dModules["TeacherClassifier"](xp)
                    else:
                        with torch.no_grad():
                            y_hat = self.dModules["TeacherClassifier"](xp)
                else:
                    y_hat = self.tModel(xp, iStartIdxGen = self.iDetachIdx)
            else:
                xp = self.tModel.features(x, bNaNCheck = False)
                if self.GetValue("SeperateLinear"):
                    xp = xp.detach()
                y_hat = self.tModel.classify(xp, bNaNCheck = False)

            if self.GetValue("UseCELoss"):
                #print(y_hat.shape, y.shape)
                l = self.tLossFcn(y_hat, y)
                ce_loss += l.item() * self.GetValue("CELossWeight")
            
            if self.GetValue("VanillaKD"):
                if self.GetValue("Dataset") == "Imagenet":
                    with torch.no_grad():
                        logits = self.kdTeacher.tModel(x, bNaNCheck = False).detach()
                else:
                    logits = self.logits[idx,...]
                l2 = self.logit_crit(y_hat, logits, self.GetValue("Temperature"), bNormalize = self.GetValue("NormalizeLogits"))
                lg_loss += l2.item() * self.GetValue("LGLossWeight")
            
            if self.GetValue("FeatureKD"):
                vecTF = None
                if not self.GetValue("StoreFeatures"):
                    #gradient-less forward pass through teacher
                    with torch.no_grad():
                        _, vecTF = self.kdTeacher.tModel(x, 
                                                         vecFeatureLayers = self.vecTLidx, bEarlyQuit = True)
                        vecTF = [tf.detach() for tf in vecTF]
                    if self.kdTeacher.IsTransformer():
                        for fi in range(len(vecTF)):
                            if len(vecTF[fi].shape) == 3:
                                vecTF[fi] = torch.permute(vecTF[fi], (0, 2, 1))
                #swap dims around
                if self.IsTransformer():
                    for fi in range(len(vecF)):
                        if len(vecF[fi].shape) == 3:
                            vecF[fi] = torch.permute(vecF[fi], (0, 2, 1))
                vecL3 = self.fkd.ComputeLoss(vecF, idx, y, self.dCfg, vecTF = vecTF)
                if self.GetValue("ProjectionMethod") == "PCA":
                    ls_Mloss += max([torch.max(torch.mean(l3, dim = 1)).item() * self.GetValue("LSLossWeight") for l3 in vecL3])
                else:
                    ls_Mloss += max([torch.max(l3).item() * self.GetValue("LSLossWeight") for l3 in vecL3])
                l3r = 0
                for l3 in vecL3:
                    l3r += torch.mean(l3) / len(vecL3)
                ls_loss += l3r.item() * self.GetValue("LSLossWeight")
                
                if self.GetValue("LayerMappingMethod") == "LearnedAttn":
                    tMap += self.fkd.GetLayerWeightMap()
                
            total_loss = 0
            if self.GetValue("UseCELoss"): total_loss += self.GetValue("CELossWeight") * l
            if self.GetValue("VanillaKD"): total_loss += self.GetValue("LGLossWeight") * l2
            if self.GetValue("FeatureKD"): total_loss += self.GetValue("LSLossWeight") * l3r

            if total_loss != total_loss:
                print("NaNs detected!", total_loss, y_hat, y)
                input()

            self.tOpt.zero_grad()
            total_loss.backward()
            
            #grab the gradients of the linear classification head for analysis. Only done for student models
            if self.IsFKDStudent() and self.GetValue("FeatureKD") and self.GetValue("ProjectionMethod") == "PCA":
                g_hat = self.tModel.GetHeadGrad()
                ls_GradDP += torch.dot(g_hat[0,:], torch.mean(l3, dim = 0)).item()
            
            self.tOpt.step()
            if self.GetValue("LRScheduler") == "OneCycle": self.tSch.step()

        if self.GetValue("LRScheduler") in ["Exp", "MultiStep"]: self.tSch.step()

        dResult = {
            "CE Loss": ce_loss / iN,
            "LG Loss": lg_loss / iN,
            "Latent Loss": ls_loss / iN,
            "Max Latent Loss": ls_Mloss * self.GetValue("BatchSize") / iN
            }
        
        if self.GetValue("FeatureKD"):
            if self.GetValue("ProjectionMethod") == "PCA":
                dResult["GradientAlignment"] = ls_GradDP * self.GetValue("BatchSize") / iN
            if self.GetValue("LayerMappingMethod") == "LearnedAttn":
                dResult["LayerAttnMap"] = [[t2.item() for t2 in list(t)] for t in list(tMap * (self.GetValue("BatchSize") / iN))]
                
        return dResult