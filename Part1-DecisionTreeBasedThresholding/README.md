# DecisionTreeBasedThresholding

Given a table including nodule diameter and density type, this part is to build a tree structure and assign each nodule a specific risk grade.

-------------------------------
## Preparation

- Input Data：table contains nodule size (average diameter of nodule or its solid component), density type, pathological label and / or manual label given by expert. 
- Config file: refer to `config.py` file in this directory

-------------------------------
## Run
```
python thresholding.py -i path_to_config_file
```
or
```
from thresholding import Thresholding
thres = Thresholding(`path_to_config_file`)
thres.run()
thres.get_results()
```

-------------------------------
## Requirements
Scikit-learn, numpy, pandas