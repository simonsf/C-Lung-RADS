# C-Lung-RADS
The is the code repository for the paper:

Data-driven risk stratification and precision management of pulmonary nodules detected on chest computed tomography. Chengdi Wang, Jun Shao, Yichu He, Jiaojiao Wu, Xingting Liu, Liuqing Yang, Ying Wei, Xiang Sean Zhou, Yiqiang Zhan, Feng Shi, Dinggang Shen, and Weimin Li

***
## Introduction
This repository is for the proposed C-Lung-RADS framework used for Large-scale Pulmonary Nodule Screening on Low-dose Chest Computed Tomography.

The repository consists of three machine learning components:

### Part 1: Decision Tree-Based Thresholding Scheme
Given a table including nodule diameter and density type, this part is to build a tree structure and assign each nodule a specific risk grade. The grading thresholds are generated based on decision trees and grid search strategies. Please refer to `Part1-DecisionTreeBasedThresholding` directory in this repository for detailed usage.

### Part 2: CNN-based malignancy prediction
Given an image patch of a nodule, this part is to use CNN with attention to obtain the probablity of its malignancy. The input is the cropped nodule patch and the associated nodule mask. Please refer to `Part2-CNNBasedMalignancyPrediction` directory in this repository for detailed usage.

### Part 3: Gradient boosting-based multidimensional benign-malignant discrimination
Given a table including different groups (imaging and/or clinical and/or follow-up) of patient features, this part is to build a linear model for multidimensional identification of malignant risk of nodules. Optional clinical and follow-up features are involved in model construction by means of gradient boosting. Please refer to `Part3-GradientBoostingDiscrimination` directory in this repository for detailed usage.
***

