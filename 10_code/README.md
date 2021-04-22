# Notebooks

Two models are trained with variation in whether using openCV cropped faces as test data.

## Model A: Simple CNN

![CNN architecture](/01_images/CNN.jpg)

A convolutional neural network (CNN) is a class of deep neural networks, most commonly applied to analyzing visual imagery. 

A simple CNN architecture includes a covolutional layer that converts input features, a pooling layer that selects sample features, and a fully-connected layer that connect output from previous layers and map it into specific range. 

CNN is applied in image and video recognition, recommender systems, image classification, image segmentation, medical image analysis, natural language processing, brain-computer interfaces, and financial time series.

## Model B: VGG16 & Resnet50

VGG16 architecture:

![VGG16 architecture](/01_images/VGG16.png)

VGG16 is a convolutional neural network model proposed by K. Simonyan and A. Zisserman from the University of Oxford in

“Very Deep Convolutional Networks for Large-Scale Image Recognition” 

The model achieves 92.7% top-5 test accuracy in ImageNet, which is a dataset of over 14 million images belonging to 1000 classes.

Resnet50 architecture:

![Resnet50 architecture](/01_images/resnet50.png)

ResNet, short for Residual Networks is a classic neural network used as a backbone for many computer vision tasks. 

This model was the winner of ImageNet challenge in 2015. 

The fundamental breakthrough with ResNet was it allowed training extremely deep neural networks with 150+ layers successfully.