# Mesh Generation for Semantic Scene Completion
This repository contains the complete architecture extending inference for the [EdgeNet360](https://gitlab.com/UnBVision/edgenet360) network to generate an additional mesh output from the voxel model.

## Requirements
1. Hardware Requirements (minimum):

   * Ubuntu 18.04 LTS or newer
   * 4GB GPU

2. Software Requirements:

All software listed here is open-source.
 
   * GPU Drivers
   * Python 3
   * Anaconda 3: It is recommended to use an [Anaconda](https://www.anaconda.com/distribution/) virtual environment 
     for simplicity.

Packages:

   * Tensorflow 2 with GPU/CUDA support. It is recommended to create an Anaconda environment that automatically installs Tensorflow within:

``` shell
    conda create -n tf2 tensorflow-gpu
```

  * After creating this environment, activate it with:

``` shell
    conda activate tf2
```

   * [OpenCV 3 for Python](https://anaconda.org/anaconda/py-opencv).
   * [pandas](https://anaconda.org/anaconda/pandas).
   * [matplotlib](https://anaconda.org/conda-forge/matplotlib).
   * [sklearn](https://anaconda.org/anaconda/scikit-learn).
   * [skimage](https://scikit-image.org/docs/dev/install.html).
   * [openmesh](https://anaconda.org/conda-forge/openmesh-python).

## Inferring Over Stereoscopic Images

### Depth Map Enhancement

Example:

``` shell
    python enhance360.py DWRC shifted-disparity.png shifted_t.png new_shifted-disparity.png
```

### Prediction

Example:

``` shell
    python infer360.py DWRC new_shifted-disparity.png shifted_t.png DWRC
```

## Inferring Over Stanford 2D-3D-Semantics Dataset

### Download Stanford 2D-3D-S Dataset

To download and prepare the Stanford dataset images for inference, first copy the dataset structure using the following commands:

``` shell
    cd Data/stanford
    git clone https://github.com/alexsax/2D-3D-Semantics .
```

Next, follow the instructions on their [GitHub](https://github.com/alexsax/2D-3D-Semantics) to download areas of the dataset, and insert into the relevant folders.

### Extract Room Ground Truth

Ground truth extraction is performed per area. Find the script `read_pointcloud_mat.py` and edit the lines below according to your environment:

``` shell
    area_dir='area_3' #desired area
    base_dir = './Data/stanford' #2D-3D Semantics dataset root
    output_dir = './Data/stanford_processed' #Folder to put extracted ground truth
```

Save and run the script:
``` shell
    python read_pointcloud_mat.py
```

### Preprocess Images

To prepare the dataset for inference, find the `preproc360_stanford_batch.py` script, and adjust the lines below, according to your environment. 

``` shell
    in_path = './Data/stanford'
    processed_path = './Data/stanford_processed'
```
Save and run the script:
``` shell
    python preproc360_stanford_batch.py
```

### Inference

Run:  

``` shell
    python infer360_stanford.py area_3  office_7 0e30c45ea0604ddeb7467fd384362503 --base_path=./Data/stanford_processed
```

The output will be located in `./Output` and will include the original EdgeNet360 voxel output, as well as an additional OBJ and corresponding MAT file for the generated mesh.

### Evaluation 

To prepare the entire dataset for inference, find the `eval_stanford.py` script, and adjust the lines below, according to your environment. 

``` shell
    processed_path = './Data/stanford_processed'
```

Save and run the script:
``` shell
    python eval_stanford.py
```


