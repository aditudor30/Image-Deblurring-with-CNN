# Image Deblurring with CNN
This Computer Vision project shows how powerful are Convolutional Neural Networks when it comes to deblurring images, even on a small dataset (350 images). This repository contains instructions on how to run the code on your local machine and the main ideas behind it.
## Table of Contents
  - [Overview](https://github.com/aditudor30/Image-Deblurring-with-CNN/blob/main/README.md#overview)
  - [Requirements](https://github.com/aditudor30/Image-Deblurring-with-CNN/blob/main/README.md#requirements)
  - [Description](https://github.com/aditudor30/Image-Deblurring-with-CNN/blob/main/README.md#Description)
  - [Project Structure](https://github.com/aditudor30/Image-Deblurring-with-CNN/blob/main/README.md#project-structure)
  - [Conclusion](https://github.com/aditudor30/Image-Deblurring-with-CNN/blob/main/README.md#conclusion)
## Overview

This project tackles the problem of *image deblurring* using a deep learning approach. Motion blur, defocus, or other distortions can cause significant degradation in image quality. The goal of this project is to train a neural network that can take a blurred image as input and output a *sharpened reconstruction* that closely matches the original.

The model is trained in a *supervised learning* setting, using paired examples of blurred and sharp images. It learns to map blurry inputs to their corresponding high-quality versions. The training process involves minimizing the difference (e.g., L1 or MSE loss) between the predicted output and the ground truth sharp image.

Key features of this project:
- Applying gaussian blur to sharp images so that we obtain our training and validation dataset
- Training loop with periodic checkpoint saving and result visualization
- Output images saved for qualitative comparison across training epochs
- Visual inspection using matplotlib to assess improvements over time

The architecture and loss function are designed to recover structural details and textures that are lost due to blur, with the goal of producing visually pleasing and accurate reconstructions.

## Requirements

To run this project, you'll need the following dependencies:

- Python 3.8+
- [PyTorch](https://pytorch.org/) (compatible with your GPU or CPU)
- torchvision
- OpenCV (opencv-python)
- matplotlib
- numpy
- tqdm
- Jupyter Notebook (optional, for exploration and visualization)

### Install with pip:

```
pip install torch torchvision opencv-python matplotlib numpy tqdm notebook
```
Make sure your CUDA drivers are installed correctly if you plan to train on GPU.

### Dataset

The following project was made using the Blur dataset from kaggle which you can find [here](https://www.kaggle.com/datasets/kwentar/blur-dataset)!

## Description

This project applies *deep learning* techniques to solve the problem of *image deblurring*. The core approach is based on training a *convolutional neural network (CNN)* that learns to reconstruct sharp images from their blurred counterparts in a supervised learning setting. The dataset consists of pairs of blurred and sharp images, and the network is trained to minimize the pixel-wise difference between the predicted and ground truth images, typically using *L1 loss* or *Mean Squared Error (MSE)* loss.

### Techniques used
- *Convolutional Neural Networks (CNNs):* Used for learning spatial hierarchies of features in images to map blurry inputs to sharp outputs.
- *Supervised Learning with Paired Data:* Training is done using one-to-one mappings between blurred and sharp images.
- *Data Preprocessing and Augmentation:* Images are preprocessed and optionally augmented to improve generalization.

### Evaluation

After making the model and using it, I had to evaluate it and I did so with the following graph:

<br/>

![loss](https://github.com/user-attachments/assets/fa2b5a2f-6e4a-4577-b151-038edc69fb34)

<br/>

We can clearly see how the model is performing better and better with each epoch and isn't overtrained.
Following this, I wanted to actually see how it does on an image in different stages of training, and with the last block of code I represented an image in the first epoch of training, then in the middle of training, and then the final product of training:

<br/>

![Progress](https://github.com/user-attachments/assets/cd9957df-375e-4b23-8499-981f4fb899b4)

<br/>

The result is speaking for itself, the model succesfully found the mathematical connection between the blurred images and the sharp ones.
   
## Project Structure
This project has the following structure:

  ```
├───inputs
│   ├───gaussian_blurred
│   └───sharp
├───outputs
│   └───saved_images
└───src
│   first_deblur.ipynb
  ```
## Conclusion
Thanks for checking out my Image Deblurring repository! As an intermediate practitioner, I've found that working on these projects has deepened my understanding of various machine learning concepts and techniques. I hope you find these examples helpful in your own journey, whether you're looking to enhance your skills or explore new ideas.

Feel free to experiment with the code, adapt the projects to your own datasets, or contribute with your insights. If you have any questions or suggestions, don’t hesitate to reach out.


