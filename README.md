# Overview
This is the official code repository for the NeurReps paper "Logit-Based Losses Limit the Effectiveness of Feature Knowledge Distillation". arXiv link: https://arxiv.org/abs/2511.14981

# Setup
* Clone this repository somewhere.
* Create a new virtual environment with `python3 -m venv {desired_venv_path}`.
* Activate the new environment with `source {desired_venv_path}/bin/activate`.
* Install the necessary python packages
    + Run `python -m pip install torch torchvision numpy matplotlib scipy tqdm scikit-learn pandas` to get started.
    + The above should be everything you need. If for some reason it doesn't work, you can try installing everything in the `requirements.txt` file.
    + On some linux distros, you may need to run `python -m pip install PyQt6` to get the matplotlib GUI to work.
* Test the setup
    + `cd` into the repo's Scripts folder, and try running `python ./main.py`
    + If the setup was sucessful, this will go download CIFAR10 then run a basic train/display loop with a small ResNet.


## Important Notes

### Environment
* This software was developed on Ubuntu 24.04 and is suitable for use on Linux-based operating systems **only**. I have absolutely no idea how well it will work on Windows/Mac, if at all. Some modifications will likely be necessary. 
* The existence of a CUDA capable GPU is assumed.
* Developed using python 3.12.

### Code Behavior
* By default, the IDataset interface class will look for a path to store dataset files in /Scripts/Datasets/DatasetsCfg.json. When you first run the main.py script, it will prompt you to specify a dataset storage location, and use `/home/{username}/Datasets/` as a fallback.
* If you want to do things with pre-trained weights for some teacher models, you will need to go download them yourself. Web links have been left as comments in the applicable places, e.g., if you want to use the same ResNet34 weights, you can find the link in `Scripts/Arch/Models/Resnet.py`.
    + The code assumes all downloaded pre-trained weights are saved to `/home/{username}/ImagenetPretrainedModels/`.


# Usage

## Basics
I imagine there are two primary things you'll want to do with this code:
* Run experiments one-at-a-time
* Run a whole bunch of experiments all-at-once

Therefore, there are two primary use cases of the `main.py` script. If no arguments are passed, the config defined in the `TestBench()` function on line 52 will be run. You can mess with configuration parameters there, and run the corresponding experiment with `python ./main.py`. Alternatively, you can pass `-c {path_to_config_file}` to point the code at a config on the disk. A blank config file has been provided for use as a starting point. You can also use `-e {path_to_folder}` to run all config files in a directory. Config files for the experiments reported in the paper have been provided in the `ExperimentConfigs/MainExperiments/` folder. Experiment results will be automatically assigned a hash ID and stored in the `KDTrainer` folder.

## Reproducing the Paper's Results
As mentioned above, all config files used for experiments in the paper are included in the `ExperimentConfigs/MainExperiments/` folder. I highly recommend to **NOT** simply try to run every experiment all-at-once, as this can take several days (depending on your GPU). If you have access to a multi-GPU machine with sufficient RAM and PCIe lanes, I recommend leveraging the multi-level organization of the experiment configs to parallelize things across workers. For example, if I wanted to re-run all the CIFAR100 experiments on a system with 2 GPUs, I would proceed as follows:
* Launch a new terminal. Call this terminal A.
    + Optional: launch a `tmux` or `screen` if running things on a remote system to prevent unnecessary crashes.
* Run `export CUDA_VISIBLE_DEVICES=0`. This will ensure Pytorch executes everything on the first GPU.
* Run `./main.py -e ../ExperimentConfigs/MainExperiments/CIFAR100F/ResNet34->ResNet9` to launch a round of experiments on GPU 0.
* Launch a second terminal (B).
* Run `export CUDA_VISIBLE_DEVICES=1` to switch things to GPU 1.
* Run `./main.py -e ../ExperimentConfigs/MainExperiments/CIFAR100F/ResNet34->MobileNetV2`.
* Make yourself a snack (optional) while you wait for terminals A and B to complete (not optional).
* Launch experiments for the remaining model pairs, rinse and repeat until complete.

The above procedure will ensure that only managable sections of the total experiment pool will be run at once. You can of course modify this workflow (or set up a bash script) to leverage however many GPUs you have available. The experiment caching system allows for incremental passes while safely storing away all the results for future visualization. 

## Feature Analysis
If you wish to compute and plot the geometric summaries of a model's latent features, uncomment the call to `TPlotClassificationEfficiency()` on line 196 of `main.py`. **WARNING**: computing such metrics can require significant disk space and computational resources depending on the dataset. The compute cost of running the analysis is highly dependent on the *dataset* for CNNs, and less so for ViTs. It requires around 32gb RAM and 16gb VRAM for small datasets (e.g., CIFAR10/100), whereas 96gb RAM and 24gb VRAM are recommended for larger datasets (e.g., TinyImagenet).

The analysis computes the geometric quantities discussed in the paper. *All features and numeric data will be saved to the disk in the corresponding experiment's cache folder.* It is easy to run out of disk space when analyzing many different models, so you may need to periodically delete some of the features.

## Plotting Results
Various plotting scripts are provided in `Plotfigs.py`. However, they will be largely useless until all the configs in the `ExperimentConfigs/MainExperiments` folder have been run. Basic training curves (losses and accuracies vs. epochs) can be easily displayed for any individual experiment by pointing `main.py` to the corresponding config.

# Code Overview

## ITrainers and the Arch "Submodule"
Much of this project is based on some "experiment-to-disk hashmap" utilities I've been developing for the past few years. This is everything in the `Scripts/Arch` folder. I plan to release this code as a standalone repo in the future, when it is closer to its final form. As it is still being actively developed, it has been directly copied here instead of linked in via submodules.

The key point of the `Arch` code is to make keeping track of complex KD experiments easier; as such experiments may, and often do, depend on a lot of inter-related parameters. This goal is accomplished via config hashing. The `KDTrainer/BaselineConfig.cjson` file contains the rules for deciding how hashes are computed based on the config parameters. This establishes the logic for handling sub-parameters, e.g., the "Teacher" field is only relevant when one is training a model with some form of distillation.

The ITrainer class (and inheritees thereof) connects the PyTorch based training code into the experiment hashing code, and is the main point of interaction for all the code in this repository. Specifically, the KDTrainer class manages running and saving KD experiments.

## The IModel Interface
As mentioned in the paper, the ill-defined nature of the term "layer" can pose a big problem for controlled FKD experimentation. Therefore, all models used in this repository are wrapped in an interface class which extracts their `.children()` and stores all "generalized layers" in an easy to work with linear data structure. The key idea here is to partition the `.children()` based on the non-linearities. The benefit of this interface is that one can easily access any subset of the model which makes working with complex FKD layer mapping schemes a lot simpler.

## Configuration Parameters
Many of the basic config parameters are rather obvious, so for these I refer you to the comments in `main.py`. Some of the more interesting ones are described in further detail below:

| Parameter | Description | Use-Cases/Values |
|-----------|-------------|------------------|
|VanillaKD | Enables/Disables training w/ classical KD | True/False |
|NormalizeLogits| When True, upgrades VanillaKD to the technique described in "Logit Standardization in Knowledge Distillation", by Sun et. al.| True/False |
|UseTeacherClassifier| Enables/Disables reusing the teacher's classification head based on the technique of "Knowledge Distillation With the Reused Teacher Classifier", by Chen et. al.| True/False|
|LearnTeacherClassifier| Enables/Disables gradient-based parameter updates in the teacher classifier. Only relevant if UseTeacherClassifier is True| True/False|
|Teacher/StudentLayers| Generalized layer indices between which FKD loss is computed| See code comment for "normal" values |
|ProjectionMethod| Controls how teacher features are translated to student space.| LearnedProjector creates learnable modules, PCA uses the classical dimension reduction algorith, and RelationFunction is used for RKD-type approaches|
|ProjectorMethod| Specifies the type of projectors to use for LearnedProjector mode.| SingleLayerConv, ThreeLayerConv|
|RelationFunction| Specifies the relation function used to compute dimensional translation loss| Use DotProd for the technique from "Similarity-Preserving Knowledge Distillation", by Tung et. al.|

By mixing and matching different config parameters, one can recreate many popular FKD and RKD methods. See the provided ExperimentConfigs for some examples of how this works.

# Remarks
Thank you for checking out this repository! I hope this readme has provided you with sufficient information to accomplish your goals. If it has not, reach out to me via email at ```nick.cooper@colorado.edu``` (preferred) or create an issue. I'm always happy to help.

# Citation
Please use the following citation if you find the paper and/or code useful for your research:
```
@inproceedings{KDwoCE,
    title = {Logit-Based Losses Limit the Effectivenss of Feature Knowledge     Distillation},
    author = {Nicholas Cooper and Lijun Chen and Sailesh Dwivedy and Danna Gurari},
    booktitle = {NeurIPS Workshop on Symmetry and Geometry in Neural Representations},
    year = {2025},

}
```