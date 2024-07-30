# An End-to-end Facial Image Inpainting with Skip Connection

This repository contains the code for the course 5AUA0 given at Eindhoven University of Technology in Q4, 2021-2022 which I did together with Liyuan Jiang.

## Requirements
This code has been tested with the following versions:
- python == 3.8
- pytorch == 1.11
- torchvision == 0.12
- numpy == 1.21
- pillow == 9.0
- cudatoolkit == 11.3 (Only required for using GPU & CUDA)

## Dataset
We use the CelebA dataset for training and testing. To download and prepare the dataset, follow these steps:
- Download the [img_align_celeba](https://www.cs.toronto.edu/~kriz/cifar.html) from [this URL](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz).
- Unpack the file and store the `img_align_celeba` directory in the `data` directory of this repository.

## Training and testing the network
- random mask for training, center mask for testing


## Config and hyperparameters
- see config.py and models.py

## Results
![exp4.jpg](..%2F..%2F..%2Ffinal%2Fexp4.jpg)
Our report `5aua0-group23-report.pdf` describes the architecture of the model and the training procedure.
## Reference
[CE code](https://github.com/eriklindernoren/PyTorch-GAN)
[Context Encoders: Feature Learning by Inpainting](https://openaccess.thecvf.com/content_cvpr_2016/papers/Pathak_Context_Encoders_Feature_CVPR_2016_paper.pdf)

