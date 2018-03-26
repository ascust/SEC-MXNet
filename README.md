# MXNet Implementation of SEC
This is a reimplementation of the paper "Seed, Expand and Constrain: Three Principles for Weakly-Supervised Image 
Segmentation"([Original Github](https://github.com/kolesman/SEC)). 

## Features

1. Compared with the original Caffe version, this version includes all the training codes such as training code for 
foreground cues and background cues. Therefore new dataset can be used.

2. This version supports multi-gpu training, which is much faster than the original Caffe version.

3. Apart from VGG16 base network, Resnet50 is also provided as a backbone.

4. For performance (VOC12 validation), VGG16 version is a bit lower than the score reported in the paper (IOU: 50.2 vs 50.7),
due to randomness. 
The Resnet50 version is much higher than the VGG16 version (IOU: 55.3). 

## Dependencies

The code is implemented in MXNet. Please go to the official website ([HERE](https://mxnet.apache.org)) for installation.
Please make sure the MXNet is compiled with OpenCV support. 

The other python dependences can be found in "dependencies.txt" and can be installed:

```pip install -r dependencies.txt```

## Dataset

There are two datasets, PASCAL VOC12([HERE](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar)) and
 SBD([HERE](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz)).
Extract them and put them into folder "dataset", and then run:

```python create_dataset.py```

## Training

Download models pretrained on Image-net ([HERE](https://1drv.ms/u/s!ArsE1Wwv6I6dgQGqn_nDGobaSSSf)), extract the files and 
put them into folder "models". 

In "cores.config.py", the base network can be changed by editing "conf.BASE_NET". The other parameters can also be tweaked.

The training process involves three steps: training fg cues, training bg cues and training SEC model, which are:

```
python train_bg_cues.py --gpus 0,1,2,3
python train_fg_cues.py --gpus 0,1,2,3
python train_SEC.py --gpus 0,1,2,3
```

## Evaluation

The snapshots will be saved in folder "snapshots". To evaluate a snapshot, simply use (for example epoch 8):

```python eval.py --gpu 0 --epoch 8```

There are other flags:

```
--savemask          save output masks
--crf               use CRF as postprocessing
--flip              also use flipped images in inference
```

Trained models can be found at ([vgg16](https://1drv.ms/u/s!ArsE1Wwv6I6dgQJzWjofCuKSjj__) and 
[resnet50](https://1drv.ms/u/s!ArsE1Wwv6I6dgQNwHjkgirojG_zW)). 