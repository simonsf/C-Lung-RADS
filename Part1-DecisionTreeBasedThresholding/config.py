from easydict import EasyDict as edict

__C = edict()
cfg = __C

# Str, input data file path, should be in csv or excel format
# refer to `utils/data_format.yaml` for requirement of inpit file
__C.data_list = "path_to_read"

# Str or None. If provided, result data will be saved in the directory
__C.save_dir = 'path_to_save'

# List of length of 2. Names of columns in input file
# First item is the name of total mean diameter, second item the name of mean diameter of solid component of mGGNs
__C.axis_colums = ['Axis', 'SolidAxis']

# Str, name of pathological label column
__C.pathology_label_column = 'Cancer'

# Str or None, name of manual label column
__C.manual_label_column = 'Level'

# Str or None, name of image malignancy column
# If provided, nodule with high malignancy will get higher grade
__C.malignancy_prob_column = 'MalignantProb'
# Float between [0, 1] or None, should be provided if malignancy_prob_column is not None
# threshold for define high-maligancy nodule
__C.malignancy_prob_thres = 0.5


# Configures for define decision tree
__C.tree = {}
# Same as `max_depth` and `max_leaf_nodes` and `class_weight` in scikit-learn DecisionTreeClassifier
__C.tree.max_depth = 5
__C.tree.max_nodes = 32
__C.tree.class_weight = 'balanced'
# Int, lower and upper bound of selected thresholds. 
# Threshold candidates below lower bound or above upper bound will be discarded
__C.tree.axis_lower_bound = 4
__C.tree.axis_upper_bound = 20

# Configures for grid search
__C.grading = {}
# Int, num of best threshold combination of each density type to be kept
__C.grading.num_topK = 10
# Dict, min population proportion of each grade
__C.grading.min_proportion = {
    'solid': [0.6,0.05,0.01,0.005],
    'pGGN': [0.5, 0.1, 0.05],
    'mGGN': [0.2,0.05,0.01,0.005],
    'total': [0.6,0.05,0.01,0.005]
}

