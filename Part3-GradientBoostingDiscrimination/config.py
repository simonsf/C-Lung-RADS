from easydict import EasyDict as edict

__C = edict()
cfg = __C

# Str, input data file path, should be in csv or excel format
# refer to `utils/data_format.yaml` for requirement of inpit file
__C.data_list = "path_to_read"

# Str or None. If provided, result data will be saved in the directory
__C.save_dir = 'path_to_save'

# Str, name of pathological label column
__C.pathology_label_column = 'Cancer'
# Str, name of AI-malignancy column (result from CNN-based malignancy prediction module)
__C.image_column = 'MalignantPro'
# Str, name of sex column
__C.sex_column = 'Sex'
# List of str, names of all clinical feature columns except sex
# need to be normalized in advance
__C.clinical_columns = ['Age', 'Smoking', 'Tumor', 'FamLungCancer', 'FamTumor',]
# List of Str, names of all follow-up feature columns
# If use VDT and SGR together, they should be normalized to same scale in advance
__C.followup_columns = ['SGR+', 'SGR-']
# List of str, subset of all feature columns.
# Names of all descrete features 
__C.descrete_columns = ['Sex', 'Smoking', 'Tumor', 'FamLungCancer', 'FamTumor']


__C.model = {}
#Int, use bagging to decorrelate Sex and other features, fit on clinic_bagging_num of sex-balanced subsets
__C.model.clinic_bagging_num = 50
# Float, each subset should contain clinic_bagging_frac of total samples
__C.model.clinic_bagging_frac = 0.8

# Str, one of ['classification', 'regression'].
# Use classification or regression model in gradient boosting
__C.model.type = 'classification'
# Str, type of loss function
# Should be one of ['logistic', 'modified_huber'] for classification model
# Should be one of ['square', 'huber'] for regression model
__C.model.loss_func = 'modified_huber'

# Boolean, whether to fill null feature during training. If fillna is False, drop samples with null feature
__C.model.fillna = False
# Numeric, fill null features with fillval 
__C.model.fillval = 0

# 'balanced' or numeric, weight for positive (malignant) samples when training with clinical features. 
# If set to 'balanced', automatically adjust weights inversely proportional to class frequencies
__C.model.weight_pos = 'balanced'
# 'balanced' or numeric, weight for positive (malignant) samples when training with follow-up features. 
# If set to 'balanced', automatically adjust weights inversely proportional to class frequencies
__C.model.weight_pos_follow = 100

# Boolean, whether to use dummy feature for descrete columns
__C.model.dummy = True

#Boolean, whether to calculate the intercept for LasoCV model.
__C.model.use_intercept = False
