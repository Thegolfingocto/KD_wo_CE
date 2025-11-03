'''
Code written by Nicholas J. Cooper.
Released under the MIT license, see the GitHub page for the full legal boilerplate.
tldr: you freely can do whatever you like with this code, but please cite the GitHub at: https://github.com/Thegolfingocto/KDwoCE.git
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
'''




import torch
import argparse

from Arch.Models.ModelUtils import *

from FeatureUtils import *
from TrainerUtils import *
from KDUtils import *
from ConfigUtils import *
from Arch.Analysis.Metrics import *

from Arch.Utils import *
from Arch.Logger import *

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#print(f"Using device '{device}'")

def FromConfig(strPath: str) -> None:
    with open(strPath, "r") as f:
        dCfg = json.load(f)

    T = KDTrainer("../KDTrainer", dCfg, bStartLog = True)
    
    if dCfg["FeatureKD"] or dCfg["VanillaKD"]: T.SetTeacher(ConstructTeacher(dCfg["Teacher"], dCfg["Dataset"], bDataAug = dCfg["DataAugmentation"]))

    return T.IDisplayResult()

def RunCfgs() -> None:
    #Just a place for coding up larger experiment runs
    #TrainConfigs("../ExperimentConfigs/MainExperiments/CIFAR10/", bPrintResults = False, dNFilter = {"Model": "ViT_ETT"})
    #TrainConfigs("../ExperimentConfigs/MainExperiments/CIFAR100F/", bPrintResults = False, dNFilter = {"Model": "ViT_ETT"})
    TrainConfigs("../ExperimentConfigs/MainExperiments/TinyImagenet/", bPrintResults = False, dNFilter = {"Model": "ViT_ETT"})

    return

def RunConfigs(strDir: str = None) -> None:
    if strDir is not None: TrainConfigs(strDir, bPrintResults = False)
    return

def TestBench():
    Config = {}
    
    #Dataset stuff
    Config["Dataset"] = "CIFAR10" #CIFAR10, CIFAR100C, CIFAR100F, CIFAR100S, 10News, 20News, TinyImagenet, EuroSAT, StanfordCars, DBpedia
    Config["DownSample"] = -1
    Config["Normalization"] = "MeanVar"
    Config["EmbedDimension"] = 100 #only used for text datasets
    Config["Subset"] = 7 #only used for CIFAR100S
    
    #Model stuff
    Config["Model"] = "ResNet9" #VGG19, ResNet56, ResNet50, ResNet9, MiniResNet, MiniResNetX2, ViT_(E)T(L), ViT_S, ViT_B, TcT_(E)T(L), TcT_S, TcT_B
    Config["PreTrained"] = False #only ResNet{18, 34, 50, 101, 152}, VGG{11, 13, 16, 19}, MobileNetV2, and ViT_B support this
    Config["BackboneLRMultiplier"] = 1e-2 #modification factor for layers prior to the head. Set to -1 to disable backbone fine-tuning
    Config["AvgTokens"] = True #Only used for ViTs
    Config["BatchNorm"] = True #TODO: consider removing this as there does not appear to exist any use cases for running without BN

    #ReLU stuff
    Config["UseCustomReLU"] = False
    Config["Kn"] = 0.236
    Config["Kp"] = 1
    
    Config["UseCELoss"] = True
    Config["SeperateLinear"] = False #when true, disconnects layers past the final distillation target from the rest of the graph
    
    Config["Teacher"] = "ResNet34"
    
    #Logit-KD stuff
    Config["VanillaKD"] = False
    Config["Temperature"] = 4
    Config["NormalizeLogits"] = False
    
    #Feature/Relation-KD stuff
    Config["FeatureKD"] = False #main flag that enables/disables this section
    Config["StoreFeatures"] = False #When True, loads teacher's features into VRAM. This speeds up training but may impose high 
                                   #VRAM requirements. Set to False if you get OOM errors

    Config["UseTeacherClassifier"] = False
    Config["LearnTeacherClassifier"] = False
    
    #Config["StoreFeatures"] = Config["Dataset"] != "TinyImagenet"
    
    #Loss options. These are only relevant for ProjectionMethods which compute target points in student space
    Config["UseCosineLoss"] = False #use direction only for computing loss in student space
    Config["UseHuberLoss"] = False #use Huber vs. MSE
    
    #These fields define the distillation layer scheme independantly from the distillation technique

    #-------------Important Note--------------#
    #All Major Model "standard" layer selections
    #VGG19 := [1, 3, 7, 11, 15, 16]
    #VGG11 := [0, 1, 3, 5, 7, 8]
    #ResNet34 := [0, 3, 7, 13, 16, 17]
    #ResNet9 := [0, 1, 2, 3, 4, 5]
    #MobileNetV2 := [0, 3, 6, 13, 17, 19]
    #ViT_B := [0, 2, 4, 8, 10, 12]
    #ViT_ETT := [0, 1, 3, 5, 7, 8]
    #Additional Note: Most existing approaches only use [1:-1] from above lists!

    Config["TeacherLayers"] = [12, 14, 15, 16] #[12, 13, 14, 15, 16] #teacher layers to allow student to learn from
    #Config["TeacherLayers"] = [2, 4, 8, 12, 16] -> [2, 4, 8, 10, 12] -> [8, 9, 10, 11, 12]
    #[2, 4, 8, 12, 16]: good test for VGG19, [0, 4, 8, 14, 17] for RN34
    Config["StudentLayers"] = [3, 7, 13, 16] #student layers to distill knowledge into, layers are in generalized layer format. TODO: consider
                                   #adding support for non-gen. layers if need be.
    Config["LayerMappingMethod"] = "One2One" #One2One, PreDefined, FullyConnected, LearnedAttn
    Config["TeacherPowers"] = [] #optional list of power transform exponents to reduce teacher feature ID
    
    #These fields are only used in PreDefined mode
    Config["LayerMap"] = [[]] #set up a list[list[int]] to define which teacher layers distill to each student layer
    Config["LayerMapWeights"] = [[]] #optional list[list[float]] of same shape as LayerMap
    
    #These fields are only used in LearnedAttn mode
    Config["StudentPartitions"] = [] #optional list[int] of breakpoints of student model
    Config["TeacherPartitions"] = [] #optional list[int] of breakpoints of teacher model. Only layers within the same partitions
                                       #are allowed to attend to each other
    Config["AttnArchitecture"] = [256, 128] #list of dims defines an MLP. Do NOT include input size, this is batch_size**2.
    Config["AttnInput"] = "DPS" #DPS, ED, dot-product sim. euclidean dist.
    
    #These fields define the distillation technique, independant from the distillation layer scheme
    Config["ProjectionMethod"] = "LearnedProjector" #PCA, LearnedProjector, RelationFunction
    
    #These fields are only used for PCA mode
    Config["UseGlobalCenter"] = True #When True, subtracts translated global center from features
    Config["LandmarkMethod"] = "CC" #method for computing the class centers
    Config["LandmarkPostScaleFactor"] = 1 #option for inflating/deflating intra-class variance
    
    #These fields are only used for LearnedProjector mode
    Config["ProjectorArchitecture"] = "SingleLayerConv" #ThreeLayerConv, SingleLayerConv, TODO more?
    Config["PoolMode"] = True
    Config["LearnProjectors"] = True
    
    #These fields are only used for RelationFunction mode
    Config["RelationFunction"] = "DotProd" #DotProd, Distance, TODO: others?

    #Loss weights
    Config["CELossWeight"] = 1
    Config["LGLossWeight"] = Config["Temperature"]**2
    Config["LSLossWeight"] = 1
    
    Config["LearningRate"] = 0.01
    Config["WeightDecay"] = -1 #1e-4
    Config["DataAugmentation"] = False

    Config["ExpPowersetErasing"] = False

    Config["AugmentTeacher"] = True

    #Config["Optimizer"] = "SGD"
    Config["Optimizer"] = "AdamW" #shorthand for Adam w/ weight decay
    
    #Config["LRScheduler"] = "None"

    #Config["LRScheduler"] = "MultiStep"
    Config["LRSteps"] = [30, 60, 90]
    Config["LRStepMult"] = 0.1

    #Config["LRScheduler"] = "Exp"
    Config["ExpPower"] = 0.9
    
    Config["LRScheduler"] = "OneCycle"
    Config["PctStart"] = 0.3 #Don't touch this! aparently...
    #Config["MaxLR"] = 0.001

    Config["MaxLR"] = LookupModelMaxLR(Config["Model"], Config["Dataset"], Config["ProjectorArchitecture"] if Config["FeatureKD"] else "")
    if Config["LRScheduler"] == "OneCycle":
        Config["LearningRate"] = Config["MaxLR"]
    #if Config["PreTrained"] and Config["FeatureKD"]: Config["BackboneLRMultiplier"] = 1 #allow normal training when using FKD
    
    Config["BatchSize"] = 128
    Config["NumEpochs"] = 50
    #Config["TwoPhaseMode"] = True

    Config["NumRuns"] = 1
    Config["EvalInterval"] = 1
    Config["CheckpointInterval"] = -1 #save model every N epochs. Set to -1 to disable
    
    Config["SaveResult"] = True
    Config["SaveModel"] = True

    T = KDTrainer("../KDTrainer", Config, bStartLog = True)
    
    if Config["FeatureKD"] or Config["VanillaKD"]: T.SetTeacher(ConstructTeacher(Config["Teacher"], Config["Dataset"], bDataAug = Config["DataAugmentation"]))

    T.IDisplayResult() #will train if config not found

    TPlotClassificationEfficiency(T)

    quit()
    
    return

if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-c", "--config")
    argParser.add_argument("-e", "--experiment")
    argParser.add_argument("-etb", "--experimentTB")
    args = argParser.parse_args()

    if args.config is not None: FromConfig(args.config)
    elif args.experiment is not None: RunConfigs(args.experiment)
    elif args.experimentTB is not None: RunCfgs()
    else: TestBench()