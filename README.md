# Setup
* Clone this repository somewhere.
* Make sure you have the necessary python packages
    + Run `python -m pip install torch torchvision numpy` to get started.
    + Either fix missing packages as you need, or install everything in the `requirements.txt` file.
    + Depending on which part(s) of the code you wish to use, many packages may not be necessary.
* Test the setup
    + `cd` into the repo's Scripts folder, and try running `python ./main.py`
    + If sucessful, this should attempt to run a basic train/display loop (Resnet on CIFAR10)


## Important Notes
* This software was developed for use on Linux-based operating systems **only**. I have absolutely no idea how well it will work on Windows/Mac, if at all. Furthermore, the existence of a CUDA capable GPU is assumed.
* By default, the code will attempt to download and save datasets to `/home/{username}/Datasets/`. If this is undesirable behavior, go modify the file `Scripts/Arch/Datasets/IDataset.py`, line 33.
* If you want to do things with pre-trained weights for some teacher models, you will need to go download them yourself. Web links have been left as comments in the applicable places, e.g., if you want to use the same ResNet34 weights, you can find the link in `Scripts/Arch/Models/Resnet.py`.
    + In analogy with the datasets, the code will assume you've downloaded pre-trained weights to `/home/{username}/ImagenetPretrainedModels/`.
* I do understand that some the above points will be annoying to deal with, and for this, I apolgize. If enough people harass me, I will spend some time to clean things up.

# Usage
I imagine there are two primary things you'll want to do with this code:
* Run experiments one-at-a-time
* Run a whole bunch of experiments all-at-once

Therefore, there are two primary use cases of the `main.py` script. If no arguments are passed, the config defined in the `TestBench()` function on line 52 will be run. You can mess with configuration parameters there, and run the corresponding experiment with `python ./main.py`. Alternatively, you can pass `-c {path_to_config_file}` to point the code at a config on the disk. Use `-e {path_to_folder}` to run all config files in the directory. Config files for the major experiments reported on in the paper have been provided in the `ExperimentConfigs/MainExperiments/` folder. Experiment results will be automatically assigned a hash ID and stored in the `KDTrainer` folder.

If you wish to compute and plot the geometric summaries of a model's latent features, uncomment the call to `TPlotClassificationEfficiency()` on line 197 of `main.py`. **WARNING**: computing such metrics requires significant disk space and computational resources.

# Code Overview

## ITrainers and the Arch "Submodule"
Much of this project is based on some "experiment-to-disk hashmap" utilities I've been developing for the past few years. This is everything in the `Scripts/Arch` folder. I plan to release this code as a standalone repo in the future, when it is closer to its final form. Currently, it is still being actively developed, which is why it is directly copied here instead of linked in via submodules.

The key point of the `Arch` code is to make keeping track of complex KD experiments easier. Such experiments may, and often do, depend on a lot of inter-related parameters. This goal is accomplished via config hashing. The `KDTrainer/BaselineConfig.cjson` file contains the rules for deciding how hashes are computed based on the config parameters. This establishes the logic for handling sub-parameters, e.g., the "Teacher" field is only relevant when one is training a model with distillation.

The ITrainer class (and inheritees thereof) connects the PyTorch based training code into the experiment hashing code, and is the main point of interaction for all the code in this repository. Specifically, the KDTrainer class manages running and saving KD experiments.

## The IModel Interface
As mentioned in the paper, the ill-defined nature of the term "layer" can pose a big problem for controlled FKD experimentation. Therefore, all models used in this repository are wrapped in an interface class which extracts their `.children()` and stores all "generalized layers" in an easy linear data structure. The key idea here is to partition the `.children()` based on the non-linearities. The benefit of this interface is that one can easily access any subset of the model which makes working with complex FKD layer mapping schemes a lot simpler.

## Configuration Parameters
Many of the basic config parameters are rather obvious, so for these I refer you to the comments in `main.py`. Some of the more interesting ones are described below:

| Parameter | Description | Use-Cases/Values |
|-----------|-------------|------------------|
|VanillaKD | Enables/Disables training w/ classical KD | True/False |
|NormalizeLogits| When True, upgrades VanillaKD to the technique described in "Logit Standardization in Knowledge Distillation", by Sun et. al.| True/False |
|UseTeacherClassifier| Enables/Disables reusing the teacher's classification head based on the technique of "Knowledge Distillation With the Reused Teacher Classifier", by Chen et. al.| True/False|
|LearnTeacherClassifier| Enables/Disables gradient backpropagation through the teacher classifier. Only relevant if UseTeacherClassifier is True| True/False|
|Teacher/StudentLayers| Generalized layer indices between which FKD loss is computed| See code comment for "normal" values |
|ProjectionMethod| Controls how teacher features are translated to student space.| LearnedProjector creates learnable modules, PCA uses the classical dimension reduction algorith, and RelationFunction is used for RKD-type approaches|
|ProjectorMethod| Specifies the type of projectors to use for LearnedProjector mode.| SingleLayerConv, ThreeLayerConv|
|RelationFunction| Specifies the relation function used to compute dimensional translation loss| Use DotProd for the technique from "Similarity-Preserving Knowledge Distillation", by Tung et. al.|

By mixing and matching different config parameters, one can recreate many popular FKD and RKD methods. See the provided ExperimentConfigs for some examples of how this works.

# Remarks
Thank you for checking out this repository! I hope this readme has provided you with sufficient informaiton to accomplish your goals. If it has not, reach out to me via email (preferred) or create an issue. I'm always happy to help.