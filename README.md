# Final Project
## Title - Object Classification/Recognition using Using CNN Networks and Transfer-learning with EfficientNet-B0 



## Team Members:-
<ul>
<li>Akanksha Rawat</li>
<li>Karishma Kuria</li>
<li>Nisha Mohan Devadiga</li>
</ul>

# Project Abstract

Convolutional neural network (CNN) is a class of deep neural network commonly used to analyze images. The objective of this project is to build a convolutional neural network model that can correctly recognize and classify colored images of objects into one of the 100 available classes for CIFAR-100 dataset. The recognition of images in this project has been done using transfer learning approach.

The network built in this project uses the state-of-the-art EfficientNet-B0 which was trained on the popular, challenging and large ImageNet dataset.

Transfer learning and the idea of intelligently scaling the network (carefully balancing the network's width, depth and resolution) helped in getting a good performance on this dataset. By just training the model for 15 epochs, the model managed to achieve an accuracy of 82 percent. This is definitely a much better performance than the one achieved using a 9-layer convolutional neural network model trained for 100 epochs. The training of the model has been done on a GPU and the model has also been tested on some new random images to visualize the top 5 category predictions along with their probabilities.

# Motivation

The motivation behind working on this dataset using transfer learning is the challenge of achieving a good accuracy score (more than 59 % as achieved using a 9-layer convolutional neural network built earlier). The dataset has 100 classes but just 600 images in each class (500 for training and 100 for testing). The most interesting part of this dataset is the image quality. Each of the image in the dataset is of 32 × 32 pixels which makes recognition a challenging task for machine. So in order to train the machine to correctly recognize and classify the images better than earlier, transfer learning approach has been used. However, the main limitation for building a deep neural network for CIFAR-100 with millions of parameters is memory. But, I felt that dealing with all these challenges would be a great learning and I decided to proceed with this dataset.

# CIFAR-100 Dataset

<img width="1002" alt="image" src="https://user-images.githubusercontent.com/89233753/206949501-5d0e29bd-7b1d-4dab-b062-ad8cf33a4dff.png">

CIFAR-100 dataset which has numerous images of all gender and age groups. The dataset has 100 classes but just 600 images in each class (500 for training and 100 for testing). The most interesting part of this dataset is the image quality. Each of the images in the dataset is of 32 × 32 pixels which makes recognition a challenging task for machines.

# EfficientNetB0 Architecture
<img width="1002" alt="image" src="https://user-images.githubusercontent.com/89233753/206949064-d5ec72d8-ce55-4e6e-98f2-f9f7f24ffcc4.png">


# MLOPS Experimentation using Weights and Biases

W&B is a platform for enabling a collaborative MLOps culture.
Designed to support and automate key steps in the MLOps life cycle - experiment tracking, dataset versioning and model management.

<img width="1101" alt="image" src="https://user-images.githubusercontent.com/89233753/206991106-00dac64e-67db-4afd-b2d6-76d934e5b3a9.png">


# Accuracy and Loss

<img width="1002" alt="image" src="https://user-images.githubusercontent.com/89233753/206950671-3c550929-b131-44fc-aac1-3fa0406cb406.png">
The model completes with the validation accuracy of 80.89 % and test accuracy of 80.55 % which is better as compared to the basic CNN model built in prior step. 



# Model Prediction Using Transfer Learning on EfficientNetB0 model

The result for true and predicted image after model validation can be seen here -

<img width="400" height="500" alt="image" src="https://user-images.githubusercontent.com/89233753/206950272-b9ad71e7-587a-4d0f-bacf-a0511a9aee97.png">

### Using Streamlit

![image](https://user-images.githubusercontent.com/77387431/206968695-6fd648dd-3ef6-49e4-8142-106c8d553532.png)

![image](https://user-images.githubusercontent.com/77387431/206968772-93818c12-6515-4b04-9f96-e10c4adcfa53.png)



# Links
#### Github Link: https://github.com/akanksha0911/AdvDL_FinalProject
----
#### Data set Link: https://drive.google.com/drive/folders/1D_78eEMNs_gbuhOGd-fji29bTLd5zzp7?usp=sharing
----
#### Heroku deployed Streamlit link: 
---
#### MLOps Experimentation Link: https://wandb.ai/akanksha09/FinalProject_TL_IC?workspace=user-nishamdev
---
#### Presentation Slides Link : https://docs.google.com/presentation/d/1oTOgroUCFDchAuXpOk2DwywzgVVRWMy5z4LyBKmL5z0/edit?usp=sharing
---
#### Presentation Video Link: https://drive.google.com/drive/folders/1WhH1X2MJKQw_S4MOnh74HSUTF0OwCjd7?usp=sharing
---
#### Project Report Link: https://docs.google.com/document/d/1w742029Bo8WR4Wt6phQQ0hGevDQ_xig73Dv7DBJKIgo/edit?usp=sharing
---
#### Project colab Link: https://colab.research.google.com/drive/1dnnV4oSz_EoH_4q3O-kyD9EnVHcFQqQX?authuser=1#scrollTo=BEFrM-eMWaGo


Referenes :
<ul>
<li>https://towardsdatascience.com/using-convolutional-neural-network-for-image-classification-5997bfd0ede4</li>
<li>https://www.irjet.net/archives/V7/i11/IRJET-V7I11204.pdf</li>
<li>https://journalofbigdata.springeropen.com/articles/10.1186/s40537-021-00444-8</li>
<li>https://www.analyticsvidhya.com/blog/2020/02/learn-image-classification-cnn-convolutional-neural-networks-3-datasets/</li>
<li>https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/</li>
<li>https://hackernoon.com/how-to-use-streamlit-and-python-to-build-a-data-science-app</li>
<li>https://coderwall.com/p/ssxp5q/heroku-deployment-without-the-app-being-at-the-repo-root-in-a-subfolder</li>
<li>https://github.com/timanovsky/subdir-heroku-buildpack</li>
<li>https://help.heroku.com/CKVOUPSY/how-to-switch-deployment-method-from-github-to-heroku-git-with-all-the-changes-app-code-available-in-a-github-repo</li>
