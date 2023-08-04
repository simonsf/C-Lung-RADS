from easydict import EasyDict as edict
from md_segmentation3d.utils.vseg_helpers import AdaptiveNormalizer, FixedNormalizer
import numpy as np

__C = edict()
cfg = __C


##################################
# general parameters
##################################

__C.general = {}

# image-mask-label pair list
# training csv file, head format
# single modality [image_path, mask_path, class, x, y, z, width, height, depth]
# multi modality [image_path, image_path1, ..... , mask_path, class, x, y, z, width, height, depth]
__C.general.im_classification_list = "train_info.csv"
__C.general.im_test_list = "test_info_DL.csv"

# the output of training models and logs
__C.general.save_dir = 'lung_malignant'

# continue training from certain epoch, -1 to train from scratch
__C.general.resume_epoch = -1

# when finetune from certain model, can choose clear start epoch idx
__C.general.clear_start_epoch = False

# the number of GPUs used in training
__C.general.num_gpus = 1

# random seed used in training (debugging purpose)
__C.general.seed = 1


##################################
# data set parameters
##################################

__C.dataset = {}

# the number of input channels
__C.dataset.input_channel = 1

# the number of classes
__C.dataset.num_classes = num_classes = 1

# index for label and label name
__C.dataset.label_name_index = {'0': 0, '1': 1}

# the resolution on which segmentation is performed
__C.dataset.spacing = [1, 1, 1]

# the sampling crop size, e.g., determine the context information
__C.dataset.crop_size = [96, 96, 96]


# the re-sample padding type (0 for zero-padding, 1 for edge-padding)
__C.dataset.pad_t = 0

# the default padding value list
__C.dataset.default_values = [-1024]

# translation augmentation (unit: mm)
__C.dataset.random_translation = [5, 5, 5]

# interpolation method:
# 1) NN: nearest neighbor interpolation
# 2) LINEAR: linear interpolation
__C.dataset.interpolation = 'NN'

# crop intensity normalizers (to [-1,1])
# one normalizer corresponds to one input modality
# 1) FixedNormalizer: use fixed mean and standard deviation to normalize intensity
# 2) AdaptiveNormalizer: use minimum and maximum intensity of crop to normalize intensity
#__C.dataset.crop_normalizers = [{'modality':'MR','min_p':0.01,'max_p':0.9999}]
__C.dataset.crop_normalizers = [{'modality':'CT', 'mean':-600, 'stddev':1500, 'clip':True}]


####################################
# training loss
####################################

__C.loss = {}

# the weight for cam loss
__C.loss.cam_weight = 0.1


#####################################
# net
#####################################

__C.net = {}

# the network name
# resnet18 resnet34 resnet50 resnet101 resnet152
__C.net.name = 'resnet18'


######################################
# training parameters
######################################

__C.train = {}

# the number of training epochs
__C.train.epochs = 501

# the number of samples in a batch
__C.train.batchsize = 8

# the number of threads for IO
__C.train.num_threads = 0

# the learning rate
__C.train.lr = 1e-4

##### ���� CosineAnnealing ���� T_max,eta_min,last_epoch
##### ���� Step            ���� step_size, gamma, last_epoch
##### ���� MultiStep       ���� milestones, gamma, last_epoch
##### ���� Exponential     ���� gamma, last_epoch
##### last_epoch���û�����û��������?1��last_epoch��������Ϊ__C.general.resume_epoch
##### �������кܶ࣬�Լ�pytorch��ѯ
__C.train.lr_scheduler = {}
__C.train.lr_scheduler.name = "Step"
__C.train.lr_scheduler.params = {"step_size": 500, "gamma": 0.1, "last_epoch": -1}

##### ���� Adam           ���� betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False
##### ���� SGD            ���� momentum=0, dampening=0, weight_decay=0, nesterov=False
##### �������кܶ࣬�Լ�pytorch��ѯ
__C.train.optimizer = {}
__C.train.optimizer.name = "Adam"
__C.train.optimizer.params = {"betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.01, "amsgrad": False}

# the number of batches to update loss curve
__C.train.plot_snapshot = 10

# the number of batches to save model
__C.train.save_epochs = 1




