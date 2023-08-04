import os
import glob
import torch.nn as nn
from utils.tools import *
from utils.Normalizer import Fix_normalizer, Adaptive_normalizer
from network import model

def last_checkpoint(chk_root):
    """
    find the directory of last check point
    :param chk_root: the check point root directory, which may contain multiple checkpoints
    :return: the last check point directory
    """

    last_epoch = -1
    chk_folders = os.path.join(chk_root, 'chk_*')
    for folder in glob.glob(chk_folders):
        folder_name = os.path.basename(folder)
        tokens = folder_name.split('_')
        epoch = int(tokens[-1])
        if epoch > last_epoch:
            last_epoch = epoch

    if last_epoch == -1:
        raise OSError('No checkpoint folder found!')

    return os.path.join(chk_root, 'chk_{}'.format(last_epoch))

def load_model(model_dir):
    """ load classification model
    :param model_dir: model directory
    :return: a loaded dictionary with all model info
    """

    checkpoint_dir = last_checkpoint(os.path.join(model_dir, 'checkpoints'))
    param_file = os.path.join(checkpoint_dir, 'params.pth')

    if not os.path.isfile(param_file):
        print('{} param file not found'.format(checkpoint_dir))
        return None

    # load network parameters
    state = load_pytorch_model(param_file)

    # load network structure
    net_name = state['net']
    net = model.__dict__[net_name](num_classes=state['num_classes'], input_channels=state['input_channel'])
    net = nn.parallel.DataParallel(net)
    net = net.cuda()
    net.load_state_dict(state['state_dict'])
    net.eval()

    crop_normalizers = []
    for crop_normalizer in state['crop_normalizers']:
        if crop_normalizer['type'] == 0:
            crop_normalizers.append(Fix_normalizer(crop_normalizer['mean'],
                                                    crop_normalizer['stddev'],
                                                    crop_normalizer['clip']))
        elif crop_normalizer['type'] == 1:
            crop_normalizers.append(Adaptive_normalizer(crop_normalizer['min_p'],
                                                       crop_normalizer['max_p'],
                                                       crop_normalizer['clip']))
        else:
            raise ValueError('unknown normalizer type')

    model_dict = {'net': net,
                  'spacing': state['spacing'],
                  'crop_size':  state['crop_size'],
                  'crop_normalizers': crop_normalizers,
                  'interpolation': state['interpolation'],
                  'pad_t': state['pad_t'],
                  'default_values': state['default_values'],
                  'input_channel': state['input_channel']}

    return model_dict