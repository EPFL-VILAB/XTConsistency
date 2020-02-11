![](./assets/intro.png)

# [Repo under construction!] Robust Learning Through Cross-Task Consistency

This repository shares the pretrained models from several vision tasks that have been trained to give consistent predictions given a query (RGB) image. You can find the download links to these networks and demo code for visualizing the results on a single image.

For further details about consistency (the what, why and how) or for more technical details, refer to the [paper]() or [website]().


Table of contents
=================

   * [Introduction](#introduction)
   * [Installation](#install-requirements)
   * [Running Single-Image Tasks](#run-demo-script)
   * [Training](#training)
	   * [Code structure](#the-code-is-structured-as-follows)
	   * [Steps](#steps)
   * [Citing](#citation)


## Introduction 


#### Dataset

The following domains from the [Taskonomy dataset](https://github.com/StanfordVL/taskonomy/tree/master/data) were used to train the model. Tasks with (\*) are used as a target domain with all other tasks being used as percep losses ie. a `depth` target would have `curvature`, `edge2d`, `edge3d`, `keypoint2d`, `keypoint3d`, `reshading`, `normal` as percep losses.

```
Curvature         Depth*                Edge-3D        
Edge-2D           Keypoint-2D           Keypoint-3D     
Reshading*        Surface-Normal*       RGB
```

#### Network Atchitecture

The networks are based on the [UNet](https://arxiv.org/pdf/1505.04597.pdf) architecture. They take in an input size of 256x256, upsampling is done via bilinear interpolations instead of deconvolutions and trained with the L1 loss. See the table below for more information.

| Task Name | Output Dimension | Downsample Blocks |
|-----------|------------------|-------------------|
| depth     | 256x256x1        | 6                 |
| reshading | 256x256x1        | 5                 |
| normal    | 256x256x3        | 6                 |

## Install requirements
See `requirements.txt` for complete list of packages. We recommend doing a clean installation of requirements using virtualenv:

```
conda create -n testenv python=3.6
source activate testenv
pip install -r requirements.txt
```

## Run demo script

#### Clone the code from github

```
git clone https://github.com/amir32002/scaling.git
cd scaling
git checkout ch_release
```

#### Download pretrained networks
The pretrained models for the demo can be downloaded with the following command.

```
sh ./tools/download_models.sh
```

This downloads the `baseline`, `consistency` trained models for `depth`, `normal` and `reshading` target (1.27GB). They will be saved to a folder called `models`.

Individial models can be downloaded [here](https://drive.switch.ch/index.php/s/QPvImzbbdjBKI5P).

#### Running single image tasks

To run the trained model of a task on a specific image:

```
python demo.py --task $TASK --img_path $PATH_TO_IMAGE_OR_FOLDER --output_path $PATH_TO_SAVE_OUTPUT
```

The `--task` flag specifies the target task for the input image, which should be either `normal`, `depth` or `reshading`.

To run the script for a `normal` target on the [example image](./assets/abbasi-hotel-safavid-suite.png):

```
python demo.py --task normal --img_path assets/test.png --output_path assets/
```

It returns the output prediction from the baseline (`test_normal_baseline.png`) and consistency models (`test_normal_consistency.png`).

Test image                 |  Baseline			|  Consistency
:-------------------------:|:-------------------------: |:-------------------------:
<img src="./assets/test.png" width="256" height="256" />|  ![](./assets/test_normal_baseline.png) |  ![](./assets/test_normal_consistency.png)


Similarly, running for target tasks `reshading` and `depth` gives the following.

  Baseline (reshading)      |  Consistency (reshading)   |  Baseline (depth)	       |  Consistency (depth)
:-------------------------: |:-------------------------: | :-------------------------: |:-------------------------:
![](./assets/test_reshading_baseline.png) |  ![](./assets/test_reshading_consistency.png) | ![](./assets/test_depth_baseline.png) |  ![](./assets/test_depth_consistency.png)



## Training

#### Download pretrained networks
```
[add command]
```
The models should be placed in the file path defined by `MODELS_DIR` in `utils.py`.

#### The code is structured as follows

```python
config/  
    split.txt             	# Train, val split
    jobinfo.txt			# Defines job name, base_dir
modules/          		# Network definitions
train.py			# Training script
dataset.py			# Creates dataloader
energy.py			# Defines path config, computes total loss, logging
models.py			# Implements forward backward pass
graph.py			# Computes path defined in energy.py
task_configs.py			# Defines task specific preprocessing, masks, loss fn
transfers.py			# Loads models
utils.py			# Defines file paths (described below) 
demo.py             		# Demo script
```

#### Default folder structure
```python
base_dir/  		            # The following paths are defined in utils.py (BASE_DIR)
    shared/			    # with the corresponding variable names in brackets
        models/			    # Pretrained models (MODELS_DIR)
        results_[jobname]/	    # Checkpoint of model being trained (RESULTS_DIR)
        ood_standard_set/	    # OOD data for visualization (OOD_DIR)
data_dir/			    # taskonomy data (DATA_DIRS)
```

## Steps

1) Create a `jobinfo.txt` file and define the name of the job and root folder where data, models results would be stored. An example config would be,

```
normaltarget_allperceps, /scratch
```

2) Train the task-specific network with the command

```
python -m train multiperceptual_{depth,normal,reshading}
```
More options can be found in the `train.py` file.

3) Losses and some visualizations are logged in Visdom. This can be accessed via `[server name]/env/[job name]`

**Note**: this folder provides the full code and additional resources for archival and information purposes only. We dont maintain the code here.  For more details and the full methodology, please see the [paper]() and [website]().

## Citation
If you find the code, models, or data useful, please cite this paper:

```
[add ref]
```


----
### TODOs

- <del> Create demo code that produces target output given rgb image
- <del> Table of contents
- Save trained models somewhere
	- Rename them to something reasonable
	- Change/remove the file links in transfer.py

----
