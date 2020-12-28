# Canonical Capsules: Unsupervised Capsules in Canonical Pose

![teaser](docs/teaser.gif)

## Introduction

> :warning: the source code is subject to changes up to June 1st, 2021 (official release date)

This is the official repository for the PyTorch implementation of "Canonical Capsules: Unsupervised Capsules in Canonical Pose" by [Weiwei Sun*](https://weiweisun2018.github.io/), [Andrea Tagliasacchi*](https://taiya.github.io/), [Boyang Deng](https://boyangdeng.com/), [Sara Sabour](https://scholar.google.ca/citations?user=l8wQ39EAAAAJ&hl=en), [Soroosh Yazdani](https://scholar.google.com/citations?user=u6IqTfoAAAAJ&hl=en), [Geoffrey Hinton](https://www.cs.toronto.edu/~hinton), [Kwang Moo Yi](https://www.cs.ubc.ca/~kmyi).

### Download links

- [Project Website](https://canonical-capsules.github.io)
- [PDF](https://arxiv.org/abs/2012.04718) (arXiv)
- [PDF](https://canonical-capsules.github.io/pdf/caca.pdf) (github copy)

### Citation

> :warning:    If you use this source core or data in your research (in any shape or format), we require you to cite our paper as:

```
@conference{sun2020canonical,
   title={Canonical Capsules: Unsupervised Capsules in Canonical Pose},
   author={Weiwei Sun and Andrea Tagliasacchi and Boyang Deng and 
           Sara Sabour and Soroosh Yazdani and Geoffrey Hinton and
           Kwang Moo Yi},
   booktitle={arXiv preprint},
   publisher_page={https://arxiv.org/abs/2012.04718},
   year={2020}
}
```

## Requirements

Please install dependencies with the provided `environment.yml`: 
```
conda env create -f environment.yml
```

## Datasets

- We use the ShapeNet dataset as in AtlasNetV2: download the data from AtlasNetV2's [official repo](https://github.com/TheoDEPRELLE/AtlasNetV2) and convert the downloaded data into h5 files with the provided script (i.e., `data_utils/ShapeNetLoader.py`).  

- For faster experimentation, please use our [2D planes dataset](https://drive.google.com/file/d/1YUa1aDGTyacu_84QCmMlAsZ857l4yfG5/view?usp=sharing), which we generated from ShapeNet (please cite both our paper, as well as ShapeNet if you use this dataset).

## Training/testing (2D) 

To train the model on 2D planes (training of network takes only 50 epochs, and one epoch takes approximately 2.5 minutes on an NVIDIA GTX 1080 Ti):
```
./main.py --log_dir=plane_dim2 --indim=2 --scheduler=5
```

To visualize the decompostion and reconstruction:
```
./main.py --save_dir=gifs_plane2d --indim=2 --scheduler=5 --mode=vis --pt_file=logs/plane_dim2/checkpoint.pth
``` 

## Training/testing (3D)

To train the model on the 3D dataset:
```
./main.py --log_dir=plane_dim3 --indim=3 --cat_id=-1
```

We test the model with:
```
./main.py --log_dir=plane_dim3 --indim=3 --cat_id=-1 --mode=test
``` 

Note that the option `cat_id` indicates the category id to be used to load the corresponding h5 files ([this look-up table](https://drive.google.com/file/d/1XOcahFL0FPYHn475GW-xMbiHYOHVv7_v/view?usp=sharing)):

| id | category |
|----|------------|
| -1 | all        |
| 0  | bench      |
| 1  | cabinet    |
| 2  | car        |
| 3  | cellphone  |
| 4  | chair      |
| 5  | couch      |
| 6  | firearm    |
| 7  | lamp       |
| 8  | monitor    |
| 9  | plane      |
| 10 | speaker    |
| 11 | table      |
| 12 | watercraft |

## Pre-trained models (3D)
We release the 3D [pretrained models](https://drive.google.com/file/d/1Hv3Xo7e2vec-PPQeptYsHKYWCfQ0cQ5V/view?usp=sharing)
for both single categy (airplanes), as well as multi-category (all 13 classes).


## Classification
We plan to release our code for the classification experiments in the paper as well (**ETA: by the end of Janurary**).
