### Part 2: CNN-based malignancy prediction
Given an image patch of a nodule, this part is to use CNN with attention to obtain the probablity of its malignancy. 

The input is the cropped nodule patch and the associated nodule mask. 

-------------------------------
## Preparation

- Training/Testing data csv: the full path of training data, including intensity image and nodule mask image (.nii.gz), class, bounding box information (x, y, z, weight, height, depth). Refer to 'train_info.csv' file in this directory.

- Config file: refer to `config.py` file in this directory.

- Create 'lung_malignant' folder for checkpoints.
-------------------------------
## Run
```
python train.py -i config.py
```
## Inference
```
python test.py -i test_info.csv -m lung_malignant -o test_info_res.csv
```
-------------------------------
## Requirements
torch
