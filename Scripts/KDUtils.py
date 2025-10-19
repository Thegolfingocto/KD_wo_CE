from Trainer import *

def LookupModelMaxLR(strModel: str, strData: str, strProj: str = "") -> float:
    if strData in ["CIFAR10", "EuroSAT", "PatternNet", "CIFAR100C", "CIFAR100F"]:
        if "ViT" in strModel:
            if "_ET" in strModel: return 0.001
            elif "_T" in strModel: return 0.001
            elif "_S" in strModel: return 0.0005
            elif "_B" in strModel: return 0.0001
                
        if "ResNet" in strModel or "MobileNet" in strModel or "ShuffleNet" in strModel:
            return 0.0075
        elif "VGG" in strModel:
            return 0.005
        
    if strData in ["TinyImagenet", "Imagenet"]:
        if "ViT" in strModel:
            if "_ET" in strModel: return 0.001
            elif "_T" in strModel: return 0.001
            elif "_S" in strModel: return 0.0005
            elif "_B" in strModel: return 0.0001
                
        if "ResNet" in strModel or "MobileNet" in strModel or "ShuffleNet" in strModel:
            return 0.0075
        elif "VGG" in strModel:
            return 0.005
        
    if strData in ["StanfordCars", "StanfordCarsSmall", "CUB_200"]:
        if "ViT" in strModel:
            if "_ET" in strModel: return 0.001
            elif "_T" in strModel: return 0.001
            elif "_S" in strModel: return 0.0005
            elif "_B" in strModel: return 0.0005
                
        if "ResNet" in strModel or "MobileNet" in strModel or "ShuffleNet" in strModel:
            return 0.005
        elif "VGG" in strModel:
            return 0.005
        
    if strData in ["10News", "20News",]:
        if "TcT" in strModel:
            return 0.001
        
    if strData in ["DBpedia",]:
        if "TcT" in strModel:
            return 0.0001
    
    printf("Double Check strModel", INFO)
    
    return None

def LookupModelSGDLR(strModel: str):
    if "VGG" in strModel: return 0.01
    elif "ResNet" in strModel or "MobileNet" in strModel: return 0.05
    elif "ViT" in strModel: return 0.01

    return None

def ConstructTeacher(strTeacher: str, strData, bAvgTokens: bool = True, bY: bool = False, bDataAug: bool = False) -> KDTrainer:
    #bPT = strData in ["TinyImagenet", "StanfordCars"] or "ViT" in strTeacher
    tConfig = {
        "Dataset": strData,
        "Subset": 7,
        "EmbedDimension": 100,
        "Model": strTeacher,
        "AvgTokens": bAvgTokens,
        "BatchNorm": True,
        "PreTrained": True,
        "BackboneLRMultiplier": -1,
        "UseCELoss": True,
        "FeatureKD": False,
        "UseCosineLoss": False,
        "SeperateLinear": False,
        "VanillaKD": False,
        "Logits": "",
        "Temperature": -1,
        "CELossWeight": 1,
        "LGLossWeight": 1,
        "LSLossWeight": 1,
        "LearningRate": 0.1,
        "Optimizer": "AdamW",
        "LRScheduler": "OneCycle",
        "MaxLR": 0.005,
        "BatchSize": 128,
        "NumEpochs": 50,
        "NumRuns": 1,
        "EvalInterval": 1,
        "CheckpointInterval": -1,
        "SaveModel": True, #always save teacher models
        "SaveResult": True,
        "DataAugmentation": bDataAug,
    }
    
    tConfig["MaxLR"] = LookupModelMaxLR(strTeacher, strData)
    #if tConfig["PreTrained"]: tConfig["MaxLR"] *= 1e-2
    if tConfig["PreTrained"]: tConfig["BackboneLRMultiplier"] = 1e-2
    if "ViT" in tConfig["Model"]: tConfig["NumRuns"] = 1

    #if tConfig["Dataset"] == "CIFAR10": tConfig["NumEpochs"] = 25
    
    kdT = KDTrainer("../KDTrainer/", tConfig, bStartLog = False)

    #print("Debug:", kdT.IGenHash())
    
    kdT.LoadTrainedModel(bY = bY)
    kdT.LoadResult(bY = bY)

    printf("Found teacher model w/ accuracy {}".format(kdT.dResult["AvgTestAcc"]))
    printf(kdT.IGenHash())
    
    return kdT

def TrainTeacher(strTeacher: str, bAvgTokens: bool = False) -> KDTrainer:
    T = ConstructTeacher(strTeacher, bAvgTokens)
    T.LoadTrainedModel()
    return T
