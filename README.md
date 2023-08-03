# C-Lung-RADS
A triage-driven Chinese Lung Nodules Reporting and Data System (C-Lung-RADS) for the estimatation of malignant risk of lung cancer.

***
## Introduction
This repository is for the proposed multi-phase framework used for Large-scale Pulmonary Nodule Screening on Low-dose Chest Computed Tomography

The repository consists of three machine learning components:

### Part 1: Decision Tree-Based Thresholding Scheme
Given a table including nodule diameter and density type, this part is to build a tree structure and assign each nodule a specific risk grade. The grading thresholds are generated based on decision trees and grid search strategies. Please refer to `DecisionTreeBasedThresholding` directory in this repository for detailed usage.

### Part 2: CNN-based malignancy prediction
Given an image patch of a nodule, this part is to use CNN with attention to obtain the probablity of its malignancy. Here we used our previous repository [Hierarchical Attention Mining](https://github.com/oyxhust/HAM) for dataset preparation, model training and inference, where 2D was changed to 3D for CT images.

### Part 3: Gradient boosting-based multidimensional benign-malignant discrimination
Given a table including different groups (imaging and/or clinical and/or follow-up) of patient features, this part is to build a linear model for multidimensional identification of malignant risk of nodules. Optional clinical and follow-up features are involved in model construction by means of gradient boosting. Please refer to `GradientBoostingDiscrimination` directory in this repository for detailed usage.
***

