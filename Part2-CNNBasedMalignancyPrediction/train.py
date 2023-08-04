from __future__ import print_function
from builtins import input
from imp import reload
import argparse
import importlib
import os
import sys
import time
import shutil
import numpy as np
import easydict
import torch
import torch.nn as nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import matplotlib
import matplotlib.pyplot as plt
from utils.tools import  *
from utils.dataset import ClassificationDataset
from network import model
from loss.DiceLoss import DiceLoss


def worker_init(worker_idx):
    """
    The worker initialization function takes the worker id (an int in "[0,
    num_workers - 1]") as input and does random seed initialization for each
    worker.
    :param worker_idx: The worker index.
    :return: None.
    """
    MAX_INT = sys.maxsize
    worker_seed = np.random.randint(int(np.sqrt(MAX_INT))) + worker_idx
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    torch.cuda.manual_seed(worker_seed)

def load_module_from_disk(pyfile):
    """
    load python module from disk dynamically
    :param pyfile     python file
    :return a loaded network module
    """
    dirname = os.path.dirname(pyfile)
    basename = os.path.basename(pyfile)
    modulename, _ = os.path.splitext(basename)

    need_reload = modulename in sys.modules

    # To avoid duplicate module name with existing modules, add the specified path first.
    os.sys.path.insert(0, dirname)
    lib = importlib.import_module(modulename)
    if need_reload:
        reload(lib)
    os.sys.path.pop(0)

    return lib


def save_checkpoint(net, opt, epoch_idx, batch_idx, cfg, config_file, num_modality):
    """ save model and parameters into a checkpoint file (.pth)

    :param net: the network object
    :param opt: the optimizer object
    :param epoch_idx: the epoch index
    :param batch_idx: the batch index
    :param cfg: the configuration object
    :param config_file: the configuration file path
    :param max_stride: the maximum stride of network
    :param num_modality: the number of image modalities
    :return: None
    """
    chk_folder = os.path.join(cfg.general.save_dir, 'checkpoints', 'chk_{}'.format(epoch_idx))
    if not os.path.isdir(chk_folder):
        os.makedirs(chk_folder)

    filename = os.path.join(chk_folder, 'params.pth')
    opt_filename = os.path.join(chk_folder, 'optimizer.pth')

    state = {'epoch':             epoch_idx,
             'batch':             batch_idx,
             'net':               cfg.net.name,
             'state_dict':        net.state_dict(),
             'spacing':           cfg.dataset.spacing,
             'crop_size':         cfg.dataset.crop_size,
             'interpolation':     cfg.dataset.interpolation,
             'pad_t':             cfg.dataset.pad_t,
             'default_values':    cfg.dataset.default_values,
             'input_channel':     num_modality,
             'num_classes':       cfg.dataset.num_classes,
             'crop_normalizers':  [normalization_to_dict(normalizer) for normalizer in cfg.dataset.crop_normalizers]}

    # save python check point
    save_pytorch_model(state, filename)

    # save python optimizer state
    save_pytorch_model(opt.state_dict(), opt_filename)

    # copy config file
    shutil.copy(config_file, os.path.join(chk_folder, 'config.py'))


def load_checkpoint(epoch_idx, net, opt, save_dir):
    """ load network parameters from directory

    :param epoch_idx: the epoch idx of model to load
    :param net: the network object
    :param opt: the optimizer object
    :param save_dir: the save directory
    :return: loaded epoch index, loaded batch index
    """
    # load network parameters
    chk_file = os.path.join(save_dir, 'checkpoints', 'chk_{}'.format(epoch_idx), 'params.pth')
    assert os.path.isfile(chk_file), 'checkpoint file not found: {}'.format(chk_file)

    state = load_pytorch_model(chk_file)
    net.load_state_dict(state['state_dict'])

    # load optimizer state
    opt_file = os.path.join(save_dir, 'checkpoints', 'chk_{}'.format(epoch_idx), 'optimizer.pth')
    assert os.path.isfile(opt_file), 'optimizer file not found: {}'.format(chk_file)

    opt_state = load_pytorch_model(opt_file)
    opt.load_state_dict(opt_state)

    return state['epoch'], state['batch']
    

def plot_progress(cfg, batch_idx, all_tr_losses):
    """
    Should probably by improved
    :return:
    """
    try:
        font = {'weight': 'normal',
                'size': 18}

        matplotlib.rc('font', **font)

        fig = plt.figure(figsize=(30, 24))
        ax = fig.add_subplot(111)

        ax.plot(batch_idx, all_tr_losses, color='b', ls='-', label="loss_tr")

        ax.set_xlabel("batch")
        ax.set_ylabel("loss")
        ax.legend()
        if not os.path.exists(cfg.general.save_dir):
            os.makedirs(cfg.general.save_dir)
        fig.savefig(os.path.join(cfg.general.save_dir, "train_loss.png"))
        plt.close()
    except IOError:
        print("failed to plot: ", sys.exc_info())


def train(config_file, msg_queue=None):
    """ volumetric segmentation training engine

    :param config_file: the input configuration file
    :param msg_queue: message queue to export runtime message to integrated system
    :return: None
    """
    assert torch.cuda.is_available(), 'CUDA is not available! Please check nvidia driver!'
    assert os.path.isfile(config_file), 'Config not found: {}'.format(config_file)

    # load config file
    cfg = load_module_from_disk(config_file)
    cfg = cfg.cfg

    # convert to absolute path since cfg uses relative path
    root_dir = os.path.dirname(config_file)
    cfg.general.im_classification_list = os.path.join(root_dir, cfg.general.im_classification_list)
    cfg.general.save_dir = os.path.join(root_dir, cfg.general.save_dir)
    print(cfg.general.save_dir)

    # control randomness during training
    np.random.seed(cfg.general.seed)
    torch.manual_seed(cfg.general.seed)
    torch.cuda.manual_seed(cfg.general.seed)

    # clean the existing folder if not continue training
    if cfg.general.resume_epoch < 0 and os.path.isdir(cfg.general.save_dir):
        sys.stdout.write("Found non-empty save dir.\n"
                         "Type 'yes' to delete, 'no' to continue: ")
        choice = input().lower()
        if choice == 'yes':
            shutil.rmtree(cfg.general.save_dir)
        elif choice == 'no':
            pass
        else:
            raise ValueError("Please type either 'yes' or 'no'!")

    # enable CUDNN
    cudnn.benchmark = True

    # dataset
    dataset = ClassificationDataset(
        im_classification_list=cfg.general.im_classification_list,
        input_channels=cfg.dataset.input_channel,
        label_name_index=cfg.dataset.label_name_index,
        crop_size=cfg.dataset.crop_size,
        crop_normalizers=cfg.dataset.crop_normalizers,
        spacing=cfg.dataset.spacing,
        default_values=cfg.dataset.default_values,
        random_translation=cfg.dataset.random_translation,
        interpolation=cfg.dataset.interpolation,
        )

    sampler = EpochConcateSampler(dataset, cfg.train.epochs)

    data_loader = DataLoader(dataset, sampler=sampler, batch_size=cfg.train.batchsize,
                             num_workers=cfg.train.num_threads, pin_memory=True, worker_init_fn=worker_init)

    # define network
    gpu_ids = list(range(cfg.general.num_gpus))

    net = model.__dict__[cfg.net.name](num_classes=cfg.dataset.num_classes, input_channels=cfg.dataset.input_channel)
    net = nn.parallel.DataParallel(net, device_ids=gpu_ids)
    net = net.cuda()

    # training optimizer
    opt = getattr(torch.optim, cfg.train.optimizer.name)(
        [{'params': net.parameters(), 'initial_lr': cfg.train.lr}],
        lr=cfg.train.lr, **cfg.train.optimizer.params
    )

    # load checkpoint if resume epoch > 0
    if cfg.general.resume_epoch >= 0:
        net = torch.load(os.path.join(cfg.general.save_dir, 'checkpoints', 'chk_{}'.cfg.general.resume_epoch))
        last_save_epoch, batch_start = cfg.general.resume_epoch, 0
    else:
        last_save_epoch, batch_start = 0, 0

    scheduler = getattr(torch.optim.lr_scheduler, cfg.train.lr_scheduler.name+"LR")(
        optimizer=opt, **cfg.train.lr_scheduler.params)

    batch_idx = batch_start
    if cfg.general.clear_start_epoch:
        batch_idx = 0
    data_iter = iter(data_loader)

    all_tr_losses = []
    batch_losses = []
    batches = []
    # loop over batches
    for i in range(len(data_loader)):

        begin_t = time.time()

        crops, masks, labels, filenames = data_iter.next()

        crops, masks, labels = crops.cuda(), masks.cuda(), labels.cuda()

        # clear previous gradients
        opt.zero_grad()

        # network forward
        outputs, cams = net(crops)

        # the epoch idx of model
        epoch_idx = batch_idx * cfg.train.batchsize // len(dataset)

        ex_criterion = DiceLoss()
        exloss = ex_criterion(cams, masks)

        bce_loss_func = nn.BCELoss()
        bce_loss = bce_loss_func(nn.Sigmoid()(outputs), labels.float())

        train_loss = bce_loss + exloss * cfg.loss.cam_weight

        # backward propagation
        train_loss.backward()

        # update weights
        opt.step()

        if epoch_idx != scheduler.last_epoch:
            scheduler.step(epoch=epoch_idx)
        
        batch_idx += 1
        batch_duration = time.time() - begin_t
        sample_duration = batch_duration * 1.0 / cfg.train.batchsize
        batch_losses.append(train_loss.item())

        # print training loss per batch       
        if (batch_idx + 1) % cfg.train.plot_snapshot == 0:
            all_tr_losses.append(sum(batch_losses)/len(batch_losses))
            batches.append(batch_idx)
            batch_losses=[]
            plot_progress(cfg, batches, all_tr_losses)

        if epoch_idx != 0 and (epoch_idx % cfg.train.save_epochs == 0):
            if last_save_epoch != epoch_idx:
                save_checkpoint(net, opt, epoch_idx, batch_idx, cfg, config_file, dataset.num_modality())
                last_save_epoch = epoch_idx


def main():

    long_description = "UII RTP-Net Train Engine"

    parser = argparse.ArgumentParser(description=long_description)
    parser.add_argument('-i', '--input', nargs='?', default="./code_config.py",
                        help='volumetric segmentation3d train config file')
    args = parser.parse_args()
    train(args.input)


if __name__ == '__main__':
    main()
