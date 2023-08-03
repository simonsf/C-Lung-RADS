# Gradient boosting-based multidimensional benign-malignant discrimination

Given a table including different groups (imaging and/or clinical and/or follow-up) of patient features, this part is to build a linear model for multidimensional identification of malignant risk of nodules. 

-------------------------------
## Preparation

- Input Data：table contains different groups (imaging and/or clinical and/or follow-up) of patient features and pathological label
- Config file: refer to `config.py` file in this directory

-------------------------------
## Run
```
python multidimension_classification.py -i path_to_config_file 
```
or
```
from multidimension_classification import GBClassification
gb = GBClassification(`path_to_config_file`)
gb.run()
model, data_result = gb.get_results()

# for predicting
import pandas as pd
test_data = pd.read_csv(`path_to_test_file`)
test_result = gb.data_inference(test_data)
```
-------------------------------
## Requirements
Scipy, Scikit-learn, numpy, pandas