'''
Code written by Nicholas J. Cooper.
Released under the MIT license, see the GitHub page for the full legal boilerplate.
tldr: you freely can do whatever you like with this code, but please cite the GitHub at: https://github.com/Thegolfingocto/KDwoCE.git
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
'''






import os
import json
import copy

from Trainer import KDTrainer
from KDUtils import *

from Arch.Logger import *
from Arch.Utils import *

def GetResultMaxTestAcc(strConfigPath: str, bPrintResults: bool = False,) -> dict:
    with open(strConfigPath, "r") as f:
        dCfg = json.load(f)

    return TrainConfig(dCfg, bPrintResults)["MaxTestAcc"]

def TrainConfig(dCfg: dict, bPrintResults: bool = False,) -> dict:

    
    dCfg["NumRuns"] = 1 #REMEMBER ME!


    T = KDTrainer("../KDTrainer", dCfg, bStartLog = False)
            
    if dCfg["FeatureKD"] or dCfg["VanillaKD"]:
        T.SetTeacher(ConstructTeacher(dCfg["Teacher"], dCfg["Dataset"], bDataAug = dCfg["DataAugmentation"], bY = True))
    

    T.LoadResult(bY = True)
    #if not bPrintResults: T.LoadTrainedModel(bY = True) #hello!
    if bPrintResults: T.IDisplayResult()
    
    dResult = copy.deepcopy(T.dResult)
    del T

    return dResult

def TrainConfigs(strConfigDir: str, bPrintResults: bool = False, dFilter: dict = None, dNFilter: dict = None) -> None:
    if not os.path.isdir(strConfigDir):
        printf("Error! Invalid directory {}".format(strConfigDir))
        return
    
    for strDir, _, vecF in os.walk(strConfigDir):
        for strF in vecF:
            if ".json" not in strF: continue
            print(strDir, strF)
            with open(strDir + "/" + strF, "r") as f:
                dCfg = json.load(f)
                
            if dFilter is not None:
                if not IsDictSubset(dFilter, dCfg): continue

            if dNFilter is not None:
                if IsDictSubset(dNFilter, dCfg): continue

            TrainConfig(dCfg, bPrintResults)
            try:
                TrainConfig(dCfg, bPrintResults)
            except Exception as e:
                print("ERROR!")
                print(e)
                print("----------------------------------")
                continue
        
    return

def ModifyExperiment(strConfigDir: str, dModifications: dict, strNewConfigDir: str) -> None:
    if not os.path.isdir(strConfigDir):
        printf("Error! Invalid directory {}".format(strConfigDir))
        return
    
    if not os.path.isdir(strNewConfigDir):
        os.mkdir(strNewConfigDir)
    
    for f in os.listdir(strConfigDir):
        strF = os.fsdecode(f)
        if ".json" not in strF: continue
        
        with open(strConfigDir + strF, "r") as f:
            dCfg = json.load(f)
            
        for key in dModifications.keys():
            '''if key not in dCfg.keys():
                printf("Error! Missing key {} in file {}".format(key, strConfigDir + strF))
                return'''
            if type(dModifications[key]) == dict:
                vecMaps = dModifications[key]["Maps"]
                for map in vecMaps:
                    if map[0] == dCfg[key]:
                        dCfg[key] = map[1]
                        break
            else:
                dCfg[key] = dModifications[key]
            
        with open(strNewConfigDir + strF, "w") as f:
            json.dump(dCfg, f)
            
    return

def GenerateBaseConfig(bDataAug: bool = False) -> dict:
    Config = {}

    Config["Dataset"] = ""
    Config["EmbedDimension"] = -1
    Config["Subset"] = -1

    Config["Model"] = ""
    Config["PreTrained"] = False
    Config["BackboneLRMultiplier"] = 1e-2
    Config["AvgTokens"] = True
    Config["BatchNorm"] = True
    
    Config["UseCELoss"] = True
    Config["SeperateLinear"] = False
    
    Config["Teacher"] = ""

    Config["VanillaKD"] = False
    Config["Temperature"] = 4
    Config["NormalizeLogits"] = False
    
    Config["FeatureKD"] = False
    Config["StoreFeatures"] = False
    
    Config["UseCosineLoss"] = False
    Config["UseHuberLoss"] = False

    Config["TeacherLayers"] = []
    Config["StudentLayers"] = []
    Config["LayerMappingMethod"] = "One2One"
    Config["TeacherPowers"] = []

    Config["LayerMap"] = [[]]
    Config["LayerMapWeights"] = [[]]

    Config["StudentPartitions"] = []
    Config["TeacherPartitions"] = []
    Config["AttnArchitecture"] = [256, 128]
    Config["AttnInput"] = "DPS"

    Config["ProjectionMethod"] = "LearnedProjector"

    Config["UseGlobalCenter"] = True
    Config["LandmarkMethod"] = "CC"
    Config["LandmarkPostScaleFactor"] = 1

    Config["ProjectorArchitecture"] = "SingleLayerConv"
    Config["RelationFunction"] = "DotProd"

    Config["CELossWeight"] = 1
    Config["LGLossWeight"] = Config["Temperature"]**2
    Config["LSLossWeight"] = 1
    
    Config["LearningRate"] = 0.01
    Config["WeightDecay"] = -1
    Config["Optimizer"] = "AdamW"
    Config["LRScheduler"] = "OneCycle"
    Config["PctStart"] = 0.3

    Config["LRSteps"] = []
    Config["LRStepMult"] = -1
    
    Config["BatchSize"] = 128
    Config["NumEpochs"] = 50
    Config["NumRuns"] = 1
    Config["EvalInterval"] = 1
    Config["CheckpointInterval"] = -1
    
    Config["SaveResult"] = True
    Config["SaveModel"] = True

    Config["DataAugmentation"] = bDataAug
    Config["AugmentTeacher"] = bDataAug

    Config["UseTeacherClassifier"] = False
    Config["LearnTeacherClassifier"] = False

    return Config

def GenerateMainExperimentParams() -> tuple[list[str], list[dict], list[str], dict, list[dict]]:
    vecDatasets = ["CIFAR10", "CIFAR100F", "TinyImagenet"]
    #{"Teacher": "VGG19", "Student": "ResNet9"},
    #{"Teacher": "ResNet34", "Student": "VGG11"},
    vecModelPairs = [
        {"Teacher": "VGG19", "Student": "VGG11"},
        {"Teacher": "ResNet34", "Student": "ResNet9"},
        {"Teacher": "ResNet34", "Student": "MobileNetV2"},
        {"Teacher": "ViT_B", "Student": "ViT_ETT"},
    ]

    vecDimensionalTranslations = ["SingleLayerConv", "ThreeLayerConv"]  #, "DotProd"]
    vecSeperateLinear = [False, True]

    dLayerConfigurations = {
        "Baseline": {
            "VGG19": [3, 7, 11, 15],
            "VGG11": [1, 3, 5, 7],
            "ResNet34": [3, 7, 13, 16],
            "ResNet9": [1, 2, 3, 4],
            "MobileNetV2": [3, 6, 13, 17],
            "ViT_B": [2, 4, 8, 10],
            "ViT_ETT": [1, 3, 5, 7],
        },
        "OursMulti": {
            "VGG19": {"CIFAR10": [11, 12, 13, 15], "CIFAR100F": [11, 12, 13, 15], "TinyImagenet": [11, 12, 13, 16], "StanfordCars": [11, 12, 13, 16]},
            "ResNet34": {"CIFAR10": [12, 13, 14, 15], "CIFAR100F": [12, 13, 14, 15], "TinyImagenet": [12, 13, 14, 15], "StanfordCars": [11, 12, 13, 14]},
            "ViT_B": {"CIFAR10": [9, 10, 11, 12], "CIFAR100F": [7, 10, 11, 12], "TinyImagenet": [7, 8, 11, 12]},
            "VGG11": [3, 5, 7, 8],
            "ResNet9": [2, 3, 4, 5],
            "ViT_ETT": [3, 5, 7, 8],
            "MobileNetV2": [3, 6, 13, 17],
        },
        "OursSingle": {
            "VGG19": {"CIFAR10": [7], "CIFAR100F": [11], "TinyImagenet": [11], "StanfordCars": [11]},
            "ResNet34": {"CIFAR10": [14], "CIFAR100F": [14], "TinyImagenet": [14], "StanfordCars": [14]},
            "ViT_B": [11],
        },
        "SepOnly": {
            "VGG19": [15, 16, 17, 18],
            "ResNet34": [14, 15, 16, 17],
            "ViT_B": [9, 10, 11, 12],
        },
        "InfOnly": {
            "VGG19": {"CIFAR10": [8, 9, 10, 11], "CIFAR100F": [8, 9, 10, 11], "TinyImagenet": [9, 10, 11, 12], "StanfordCars": [11, 12, 13, 16]},
            "ResNet34": {"CIFAR10": [11, 12, 13, 14], "CIFAR100F": [11, 12, 13, 14], "TinyImagenet": [11, 12, 13, 14], "StanfordCars": [11, 12, 13, 14]},
            "ViT_B": {"CIFAR10": [9, 10, 11, 12], "CIFAR100F": [7, 10, 11, 12], "TinyImagenet": [7, 8, 10, 11]},
        },
        "EffOnly": {
            "VGG19": {"CIFAR10": [7, 8, 9, 10], "CIFAR100F": [7, 8, 9, 11], "TinyImagenet": [8, 9, 10, 11], "StanfordCars": [11, 12, 13, 16]},
            "ResNet34": {"CIFAR10": [11, 12, 13, 14], "CIFAR100F": [12, 13, 14, 15], "TinyImagenet": [11, 12, 13, 14], "StanfordCars": [11, 12, 13, 14]},
            "ViT_B": {"CIFAR10": [7, 8, 10, 11], "CIFAR100F": [7, 10, 11, 12], "TinyImagenet": [6, 7, 8, 11]},
        },
        "SqrtIE": {
            "VGG19": {"CIFAR10": [7, 8, 9, 11], "CIFAR100F": [7, 8, 9, 11], "TinyImagenet": [8, 9, 10, 11], "StanfordCars": [11, 12, 13, 16]},
            "ResNet34": {"CIFAR10": [11, 12, 13, 14], "CIFAR100F": [11, 12, 13, 14], "TinyImagenet": [12, 13, 14, 15], "StanfordCars": [11, 12, 13, 14]},
            "ViT_B": {"CIFAR10": [9, 10, 11, 12], "CIFAR100F": [7, 10, 11, 12], "TinyImagenet": [7, 8, 9, 11]},
        },
        }
    
    vecConfigs = []

    vecConfigs.append( {
        "Name": "VKD",
        "Params": {
            "VanillaKD": True,
            "FeatureKD": False,
            "Temperature": 4,
            "LGLossWeight": 16,
        },
    } )

    vecConfigs.append( {
        "Name": "VKDNormalized",
        "Params": {
            "VanillaKD": True,
            "NormalizeLogits": True,
            "FeatureKD": False,
            "Temperature": 4,
            "LGLossWeight": 16,
        },
    } )

    vecConfigs.append( {
        "Name": "VKDNormalized+240E",
        "Params": {
            "VanillaKD": True,
            "NormalizeLogits": True,
            "FeatureKD": False,
            "Temperature": 4,
            "LGLossWeight": 16,

            "NumEpochs": 240,
            "NumRuns": 1,
            "Optimizer": "SGD",
            "LRScheduler": "MultiStep",
            "LRSteps": [150, 180, 210],
            "LRStepMult": 0.1,
            "LearningRate": 0.05,
            "WeightDecay": 5e-4
        },
    } )

    vecConfigs.append( {
        "Name": "VKD+240E",
        "Params": {
            "VanillaKD": True,
            "FeatureKD": False,
            "Temperature": 4,
            "LGLossWeight": 16,

            "NumEpochs": 240,
            "NumRuns": 1,
            "Optimizer": "SGD",
            "LRScheduler": "MultiStep",
            "LRSteps": [150, 180, 210],
            "LRStepMult": 0.1,
            "LearningRate": 0.05,
            "WeightDecay": 5e-4
        },
    } )
    
    vecConfigs.append( {
        "Name": "VKD-T2",
        "Params": {
            "VanillaKD": True,
            "FeatureKD": False,
            "Temperature": 2,
            "LGLossWeight": 4,
        },
    } )

    vecConfigs.append( {
        "Name": "Baseline1LC+VKD",
        "Params": {
            "VanillaKD": True,
            "Temperature": 4,
            "LGLossWeight": 16,
            "FeatureKD": True,
            "SeperateLinear": False,
            "TeacherLayers": "Baseline",
            "StudentLayers": "Baseline",
            "LayerMappingMethod": "One2One",
            "ProjectionMethod": "LearnedProjector",
            "ProjectorArchitecture": "SingleLayerConv",
        },
    } )

    vecConfigs.append( {
        "Name": "Baseline1LC+VKD+240E",
        "Params": {
            "VanillaKD": True,
            "Temperature": 4,
            "LGLossWeight": 16,
            "FeatureKD": True,
            "SeperateLinear": False,
            "TeacherLayers": "Baseline",
            "StudentLayers": "Baseline",
            "LayerMappingMethod": "One2One",
            "ProjectionMethod": "LearnedProjector",
            "ProjectorArchitecture": "SingleLayerConv",

            "NumEpochs": 240,
            "NumRuns": 1,
            "Optimizer": "SGD",
            "LRScheduler": "MultiStep",
            "LRSteps": [150, 180, 210],
            "LRStepMult": 0.1,
            "LearningRate": 0.05,
            "WeightDecay": 5e-4
        },
    } )

    vecConfigs.append( {
        "Name": "Baseline1LCFC+VKD",
        "Params": {
            "VanillaKD": True,
            "Temperature": 4,
            "LGLossWeight": 16,
            "FeatureKD": True,
            "SeperateLinear": False,
            "TeacherLayers": "Baseline",
            "StudentLayers": "Baseline",
            "LayerMappingMethod": "FullyConnected",
            "ProjectionMethod": "LearnedProjector",
            "ProjectorArchitecture": "SingleLayerConv",
        },
    } )

    vecConfigs.append( {
        "Name": "Baseline1LCFC+VKD+240E",
        "Params": {
            "VanillaKD": True,
            "Temperature": 4,
            "LGLossWeight": 16,
            "FeatureKD": True,
            "SeperateLinear": False,
            "TeacherLayers": "Baseline",
            "StudentLayers": "Baseline",
            "LayerMappingMethod": "FullyConnected",
            "ProjectionMethod": "LearnedProjector",
            "ProjectorArchitecture": "SingleLayerConv",

            "NumEpochs": 240,
            "NumRuns": 1,
            "Optimizer": "SGD",
            "LRScheduler": "MultiStep",
            "LRSteps": [150, 180, 210],
            "LRStepMult": 0.1,
            "LearningRate": 0.05,
            "WeightDecay": 5e-4
        },
    } )

    # vecConfigs.append( {
    #     "Name": "Baseline1LC",
    #     "Params": {
    #         "VanillaKD": False,
    #         "Temperature": 4,
    #         "LGLossWeight": 16,
    #         "FeatureKD": True,
    #         "SeperateLinear": False,
    #         "TeacherLayers": "Baseline",
    #         "StudentLayers": "Baseline",
    #         "LayerMappingMethod": "One2One",
    #         "ProjectionMethod": "LearnedProjector",
    #         "ProjectorArchitecture": "SingleLayerConv",
    #     },
    # } )

    vecConfigs.append( {
        "Name": "SemCKD",
        "Params": {
            "VanillaKD": True,
            "Temperature": 4,
            "LGLossWeight": 16,
            "FeatureKD": True,
            "SeperateLinear": False,
            "TeacherLayers": "Baseline",
            "StudentLayers": "Baseline",
            "LayerMappingMethod": "LearnedAttn",
            "AttnArchitecture": [256, 128],
            "AttnInput": "DPS",
            "ProjectionMethod": "LearnedProjector",
            "ProjectorArchitecture": "ThreeLayerConv",
        },
    } )

    vecConfigs.append( {
        "Name": "SimKD",
        "Params": {
            "VanillaKD": True,
            "Temperature": 4,
            "LGLossWeight": 16,
            "FeatureKD": True,
            "SeperateLinear": False,
            "TeacherLayers": "Baseline",
            "StudentLayers": "Baseline",
            "LayerMappingMethod": "One2One",
            "ProjectionMethod": "LearnedProjector",
            "ProjectorArchitecture": "ThreeLayerConv",
            "UseTeacherClassifier": True,
            "LearnTeacherClassifier": True,
        },
    } )

    vecConfigs.append( {
        "Name": "SimKD+240E",
        "Params": {
            "VanillaKD": True,
            "Temperature": 4,
            "LGLossWeight": 16,
            "FeatureKD": True,
            "SeperateLinear": False,
            "TeacherLayers": "Baseline",
            "StudentLayers": "Baseline",
            "LayerMappingMethod": "One2One",
            "ProjectionMethod": "LearnedProjector",
            "ProjectorArchitecture": "ThreeLayerConv",
            "UseTeacherClassifier": True,
            "LearnTeacherClassifier": True,

            "NumEpochs": 240,
            "NumRuns": 1,
            "Optimizer": "SGD",
            "LRScheduler": "MultiStep",
            "LRSteps": [150, 180, 210],
            "LRStepMult": 0.1,
            "LearningRate": 0.05,
            "WeightDecay": 5e-4
        },
    } )

    vecConfigs.append( {
        "Name": "SemCKD+240E",
        "Params": {
            "VanillaKD": True,
            "Temperature": 4,
            "LGLossWeight": 16,
            "FeatureKD": True,
            "SeperateLinear": False,
            "TeacherLayers": "Baseline",
            "StudentLayers": "Baseline",
            "LayerMappingMethod": "LearnedAttn",
            "AttnArchitecture": [256, 128],
            "AttnInput": "DPS",
            "ProjectionMethod": "LearnedProjector",
            "ProjectorArchitecture": "ThreeLayerConv",

            "NumEpochs": 240,
            "NumRuns": 1,
            "Optimizer": "SGD",
            "LRScheduler": "MultiStep",
            "LRSteps": [150, 180, 210],
            "LRStepMult": 0.1,
            "LearningRate": 0.05,
            "WeightDecay": 5e-4
        },
    } )

    vecConfigs.append( {
        "Name": "BaselineSP+VKD",
        "Params": {
            "VanillaKD": True,
            "Temperature": 4,
            "LGLossWeight": 16,
            "FeatureKD": True,
            "SeperateLinear": False,
            "TeacherLayers": "Baseline",
            "StudentLayers": "Baseline",
            "LayerMappingMethod": "One2One",
            "ProjectionMethod": "RelationFunction",
            "RelationFunction": "DotProd",
        },
    } )

    vecConfigs.append( {
        "Name": "BaselineSP+VKD+240E",
        "Params": {
            "VanillaKD": True,
            "Temperature": 4,
            "LGLossWeight": 16,
            "FeatureKD": True,
            "SeperateLinear": False,
            "TeacherLayers": "Baseline",
            "StudentLayers": "Baseline",
            "LayerMappingMethod": "One2One",
            "ProjectionMethod": "RelationFunction",
            "RelationFunction": "DotProd",

            "NumEpochs": 240,
            "NumRuns": 1,
            "Optimizer": "SGD",
            "LRScheduler": "MultiStep",
            "LRSteps": [150, 180, 210],
            "LRStepMult": 0.1,
            "LearningRate": 0.05,
            "WeightDecay": 5e-4
        },
    } )

    # vecConfigs.append( {
    #     "Name": "STL+SepLin",
    #     "Params": {
    #         "VanillaKD": False,
    #         "FeatureKD": True,
    #         "SeperateLinear": True,
    #         "TeacherLayers": "Baseline",
    #         "StudentLayers": "Baseline",
    #         "LayerMappingMethod": "One2One",
    #         "ProjectionMethod": "LearnedProjector",
    #         "ProjectorArchitecture": "SingleLayerConv",
    #     },
    # } )

    vecConfigs.append( {
        "Name": "OursMulti",
        "Params": {
            "VanillaKD": False,
            "FeatureKD": True,
            "SeperateLinear": True,
            "TeacherLayers": "OursMulti",
            "StudentLayers": "Baseline",
            "LayerMappingMethod": "One2One",
            "ProjectionMethod": "LearnedProjector",
            "ProjectorArchitecture": "SingleLayerConv",
        },
    } )

    # vecConfigs.append( {
    #     "Name": "OursMulti+CE",
    #     "Params": {
    #         "VanillaKD": False,
    #         "FeatureKD": True,
    #         "SeperateLinear": False,
    #         "TeacherLayers": "OursMulti",
    #         "StudentLayers": "Baseline",
    #         "LayerMappingMethod": "One2One",
    #         "ProjectionMethod": "LearnedProjector",
    #         "ProjectorArchitecture": "SingleLayerConv",
    #     },
    # } )

    # vecConfigs.append( {
    #     "Name": "OursMulti+CE+VKD",
    #     "Params": {
    #         "VanillaKD": True,
    #         "Temperature": 4,
    #         "LGLossWeight": 16,
    #         "FeatureKD": True,
    #         "SeperateLinear": False,
    #         "TeacherLayers": "OursMulti",
    #         "StudentLayers": "Baseline",
    #         "LayerMappingMethod": "One2One",
    #         "ProjectionMethod": "LearnedProjector",
    #         "ProjectorArchitecture": "SingleLayerConv",
    #     },
    # } )

    # vecConfigs.append( {
    #     "Name": "Ours-S",
    #     "Params": {
    #         "VanillaKD": False,
    #         "FeatureKD": True,
    #         "SeperateLinear": True,
    #         "TeacherLayers": "SepOnly",
    #         "StudentLayers": "Baseline",
    #         "LayerMappingMethod": "One2One",
    #         "ProjectionMethod": "LearnedProjector",
    #         "ProjectorArchitecture": "SingleLayerConv",
    #     },
    # } )

    # vecConfigs.append( {
    #     "Name": "Ours-I",
    #     "Params": {
    #         "VanillaKD": False,
    #         "FeatureKD": True,
    #         "SeperateLinear": True,
    #         "TeacherLayers": "InfOnly",
    #         "StudentLayers": "Baseline",
    #         "LayerMappingMethod": "One2One",
    #         "ProjectionMethod": "LearnedProjector",
    #         "ProjectorArchitecture": "SingleLayerConv",
    #     },
    # } )

    # vecConfigs.append( {
    #     "Name": "Ours-E",
    #     "Params": {
    #         "VanillaKD": False,
    #         "FeatureKD": True,
    #         "SeperateLinear": True,
    #         "TeacherLayers": "EffOnly",
    #         "StudentLayers": "Baseline",
    #         "LayerMappingMethod": "One2One",
    #         "ProjectionMethod": "LearnedProjector",
    #         "ProjectorArchitecture": "SingleLayerConv",
    #     },
    # } )

    # vecConfigs.append( {
    #     "Name": "Ours-SqrtIE",
    #     "Params": {
    #         "VanillaKD": False,
    #         "FeatureKD": True,
    #         "SeperateLinear": True,
    #         "TeacherLayers": "SqrtIE",
    #         "StudentLayers": "Baseline",
    #         "LayerMappingMethod": "One2One",
    #         "ProjectionMethod": "LearnedProjector",
    #         "ProjectorArchitecture": "SingleLayerConv",
    #     },
    # } )
    
    return vecDatasets, vecModelPairs, vecDimensionalTranslations, dLayerConfigurations, vecConfigs

def Get4ParamConfigString(dTempCfg) -> str:
    _, _, _, dLC, _ = GenerateMainExperimentParams()
    dLSNames = {
        "Baseline": "Std",
        "OursMulti": "Ours",
        "EndLayers": "End",
    }
    dLMNames = {
        "One2One": "O2O",
        "FullyConnected": "FC",
        "LearnedAttn": "SemCKD"
    }
    dDTNames = {
        "SingleLayerConv": "1LC",
        "ThreeLayerConv": "3LC",
        "DotProd": "SP"
    }

    strRet = "["
    strM = dTempCfg["Teacher"]

    for strLS in dLC.keys():
        if type(dLC[strLS][strM]) == dict:
            vecL = dLC[strLS][strM][dTempCfg["Dataset"]]
        else: 
            vecL = dLC[strLS][strM]
        if vecL == dTempCfg["TeacherLayers"]:
            strRet += dLSNames[strLS]
            break

    strRet += ", " + dLMNames[dTempCfg["LayerMappingMethod"]]
    strRet += ", " + dDTNames[dTempCfg["ProjectorArchitecture"] if dTempCfg["ProjectionMethod"] == "LearnedProjector" else dTempCfg["RelationFunction"]]
    strRet += ", "
    strRet += "Ours" if dTempCfg["SeperateLinear"] else "Std"
    return strRet + "]"

def GenMainExperimentBaselinePath(strData: str, strModel: str, bDataAug: bool = False) -> str:
    strRet = "../ExperimentConfigs/MainExperiments/"
    if bDataAug: strRet += "DataAug/"
    strRet += strData + "/Baselines/"
    strRet += strModel + "_Baseline.json"

    return strRet

def GenMainExperimentPath(strData: str, dModelPair: dict, strDimTrans: str, strName: str, bDataAug: bool = False) -> str:
    strRet = "../ExperimentConfigs/MainExperiments/"
    if bDataAug: strRet += "DataAug/"
    strRet += strData + "/"
    strRet += dModelPair["Teacher"] + "->" + dModelPair["Student"]  + "/"
    if strDimTrans != "": strRet += strDimTrans  + "/"
    strRet += strName
    strRet += ".json"
    return strRet

def GenerateMainExperimentConfigs(bDataAug: bool = False) -> None:
    vecDatasets, vecModelPairs, vecProjectors, dLayerConfigurations, vecConfigs = GenerateMainExperimentParams()
    
    strTop = "../ExperimentConfigs/MainExperiments/"
    if bDataAug: strTop += "DataAug/"
    if not os.path.isdir(strTop): os.mkdir(strTop)

    for strDataset in vecDatasets:
        strDSPath = strTop + strDataset
        if not os.path.isdir(strDSPath): os.mkdir(strDSPath)
        strBLPath = strDSPath + "/Baselines/"
        if not os.path.isdir(strBLPath): os.mkdir(strBLPath)
        for dModelPair in vecModelPairs:
            strMPPath = strDSPath + "/" + dModelPair["Teacher"] + "->" + dModelPair["Student"]

            #baseline teacher/student
            dCfg = GenerateBaseConfig(bDataAug)
            dCfg["Model"] = dModelPair["Teacher"]
            dCfg["Dataset"] = strDataset
            if dCfg["Dataset"] in ["StanfordCars", "CUB_200"]: dCfg["NumEpochs"] = 100 #need to train for longer on these datasets b/c there are far fewer samples
            #if dCfg["Dataset"] == "CIFAR10": dCfg["NumEpochs"] = 25 #50 epochs on C10 actually causes overfitting in many models
            dCfg["PreTrained"] = True
            dCfg["MaxLR"] = LookupModelMaxLR(dCfg["Model"], dCfg["Dataset"])
            dCfg["NumRuns"] = 1

            with open(strBLPath + dModelPair["Teacher"] + "_Baseline.json", "w") as f:
                json.dump(dCfg, f, indent = 2)

            dCfg = GenerateBaseConfig(bDataAug)
            dCfg["Model"] = dModelPair["Student"]
            dCfg["Dataset"] = strDataset
            if dCfg["Dataset"] in ["StanfordCars", "CUB_200"]: dCfg["NumEpochs"] = 100
            #if dCfg["Dataset"] == "CIFAR10": dCfg["NumEpochs"] = 25
            dCfg["MaxLR"] = LookupModelMaxLR(dCfg["Model"], dCfg["Dataset"])
            dCfg["NumRuns"] = 3

            with open(strBLPath + dModelPair["Student"] + "_Baseline.json", "w") as f:
                json.dump(dCfg, f, indent = 2)


            # dCfg["NumEpochs"] = 240
            # dCfg["NumRuns"] = 1
            # dCfg["Optimizer"] = "SGD"
            # dCfg["LRScheduler"] = "MultiStep"
            # dCfg["LRSteps"] = [150, 180, 210]
            # dCfg["LRStepMult"] = 0.1
            # dCfg["LearningRate"] = LookupModelSGDLR(dCfg["Model"])
            # dCfg["WeightDecay"] = 5e-4

            # with open(strBLPath + dModelPair["Student"] + "_Baseline+240E.json", "w") as f:
            #     json.dump(dCfg, f, indent = 2)


            #KD configs
            if not os.path.isdir(strMPPath): os.mkdir(strMPPath)

            for Config in vecConfigs:
                dCfg = GenerateBaseConfig(bDataAug)
                dCfg["Teacher"] = dModelPair["Teacher"]
                dCfg["Model"] = dModelPair["Student"]
                dCfg["Dataset"] = strDataset
                if dCfg["Dataset"] in ["StanfordCars", "CUB_200"]: dCfg["NumEpochs"] = 100
                #if dCfg["Dataset"] == "CIFAR10": dCfg["NumEpochs"] = 25
                dCfg["NumRuns"] = 3
                
                for key in Config["Params"].keys():
                    if key not in dCfg.keys():
                        print("Error! Missing or invalid key: ", key)
                        return
                    dCfg[key] = Config["Params"][key]

                dCfg["MaxLR"] = LookupModelMaxLR(dCfg["Model"], dCfg["Dataset"]) 
                dCfg["LearningRate"] = LookupModelSGDLR(dCfg["Model"])

                if dCfg["FeatureKD"]:
                    if type(dLayerConfigurations[Config["Params"]["TeacherLayers"]][dModelPair["Teacher"]]) == dict:
                        dCfg["TeacherLayers"] = dLayerConfigurations[Config["Params"]["TeacherLayers"]][dModelPair["Teacher"]][strDataset]
                    else:
                        dCfg["TeacherLayers"] = dLayerConfigurations[Config["Params"]["TeacherLayers"]][dModelPair["Teacher"]]

                    if type(dLayerConfigurations[Config["Params"]["StudentLayers"]][dModelPair["Student"]]) == dict:
                        dCfg["StudentLayers"] = dLayerConfigurations[Config["Params"]["StudentLayers"]][dModelPair["Student"]][strDataset]
                    else:
                        dCfg["StudentLayers"] = dLayerConfigurations[Config["Params"]["StudentLayers"]][dModelPair["Student"]]

                    dCfg["StoreFeatures"] = "CIFAR" in strDataset or "ViT" in dModelPair["Teacher"]
                    if dCfg["AugmentTeacher"]: dCfg["StoreFeatures"] = False
                    
                    if dCfg["ProjectorArchitecture"] == "ALL":
                        for strProj in vecProjectors:
                            dCfg["ProjectorArchitecture"] = strProj
                            dCfg["MaxLR"] = LookupModelMaxLR(dCfg["Model"], dCfg["Dataset"], strProj)

                            strDir = strMPPath + "/" + strProj + "/"
                            if not os.path.exists(strDir): os.mkdir(strDir)
                            with open(strDir + Config["Name"] + ".json", "w") as f:
                                json.dump(dCfg, f, indent = 2)

                    else:
                        strP = dCfg["ProjectorArchitecture"] if dCfg["ProjectionMethod"] == "LearnedProjector" else dCfg["RelationFunction"] 
                        dCfg["MaxLR"] = LookupModelMaxLR(dCfg["Model"], dCfg["Dataset"], strP)

                        strDir = strMPPath + "/" + strP + "/"
                        if not os.path.exists(strDir): os.mkdir(strDir)
                        with open(strMPPath + "/" + strP + "/" + Config["Name"] + ".json", "w") as f:
                            json.dump(dCfg, f, indent = 2)

                else:
                    with open(strMPPath + "/" + Config["Name"] + ".json", "w") as f:
                        json.dump(dCfg, f, indent = 2)
            
    return

if __name__ == "__main__":
    GenerateMainExperimentConfigs(bDataAug = False)