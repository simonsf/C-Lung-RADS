from __future__ import print_function
import os
import numpy as np
import time
import argparse
import os
import numpy as np
import importlib
import codecs
from easydict import EasyDict as edict
import csv
import torch
import torch.nn as nn
from utils.tools import *
from utils.ClsModel import load_model
from utils.dataset import fix_normalizers, adaptive_normalizers, resize_image_itk, center_crop
import SimpleITK as sitk
import copy
import subprocess


class use_gpu(object):
    """ switch to a gpu for computation """
    def __init__(self, gpu_id):
        self.gpu_id = gpu_id

    def __enter__(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(self.gpu_id)

    def __exit__(self, exc_type, exc_val, exc_tb):
        del os.environ['CUDA_VISIBLE_DEVICES']


def get_gpu_memory(gpu_id):
    """Get the gpu memory usage.

    :param gpu_id the gpu id
    :return the gpu memory used
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ])

    # convert lines into a dictionary
    gpu_memory = [int(x) for x in result.decode().strip().split('\n')]
    gpu_memory_dict = dict(list(zip(list(range(len(gpu_memory))), gpu_memory)))

    return gpu_memory_dict[gpu_id]


def classify_load_model(model_folder, gpu_id=0):
    """
    load classification model from folder
    :param model_folder:          the folder that contains classification model
    :param gpu_id:          which gpu to run classification model
    :return: a classification model
    """
    assert isinstance(gpu_id, int)

    # switch to specific gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(gpu_id)
    assert torch.cuda.is_available(), 'CUDA is not available! Please check nvidia driver!'

    model = dict()
    model['classification'] = load_model(model_folder)
    model['gpu_id'] = gpu_id

    # switch back to default
    del os.environ['CUDA_VISIBLE_DEVICES']

    return model


def prepare_image_fixed_spacing(images, model, center):
    ori_spacing = images[0].GetSpacing()
    spacing = model['spacing']
    box_size = (np.array(images[0].GetSize()) * ori_spacing / spacing + 0.5).astype(np.int32)

    method = model['interpolation']

    assert method in ('NN', 'LINEAR')

    resample_images = []
    iso_images = []
    for idx, image in enumerate(images):
        ret = model['crop_normalizers'][idx]
        data = sitk.GetArrayFromImage(image)
        norm_data = ret(data)

        image_origin = image.GetOrigin()
        image_spacing = image.GetSpacing()
        image_direction = image.GetDirection()

        image = sitk.GetImageFromArray(norm_data)
        image.SetOrigin(image_origin)
        image.SetSpacing(image_spacing)
        image.SetDirection(image_direction)

        if method == 'NN':
            resample_image = resize_image_itk(image, box_size.tolist(), spacing)
        elif method == 'LINEAR':
            resample_image = resize_image_itk(image, box_size.tolist(), spacing, resamplemethod=sitk.sitkLinear)
        resample_images.append(resample_image)

        crop_data = center_crop(resample_image, center, model['crop_size'], padvalue=model['default_values'][idx])
        iso_images.append(crop_data)      
  
    iso_image_tensor = torch.from_numpy(np.array(iso_images)).unsqueeze(0)

    return iso_image_tensor


def network_output(iso_batch, model):
    net = model['net']
    with torch.no_grad():
        outs, cam = net(iso_batch.cuda())
        probability = nn.Sigmoid()(outs)
        probability = probability.data.cpu().item()
        prediction = int(probability >= 0.5)

    return probability, prediction


def test(input_path, model_path, output_path, gpu_id=0, save_image=True, save_single_prob=True):
    model = classify_load_model(model_path, gpu_id=gpu_id)

    input_fieldname = get_input_fieldname_from_csv(input_path)
    input_channels = get_input_channels_from_fieldname(input_fieldname)

    output_file = open(output_path, "w")
    input_fieldname.extend(['label', 'prob'])
    writer = csv.DictWriter(output_file, fieldnames=input_fieldname)
    writer.writeheader()  

    predictions = dict()

    case_imnames, case_boxs_list = load_case_csv(input_path, input_channels)

    num_cases = 0
    num_boxs = 0

    for imname, case_boxs in zip(case_imnames, case_boxs_list):
        num_cases += 1
        print('{}/{}: {}'.format(num_cases, len(case_imnames), imname))

        images = []
        for j in range(len(imname)):
            images.append(sitk.ReadImage(imname[j], outputPixelType=sitk.sitkFloat32))
        try:
            begin = time.time()
            for box in case_boxs:
                num_boxs += 1
                bbox_info = []
                bbox_info.extend([float(box['x']), float(box['y']), float(box['z']), \
                                  float(box['width']), float(box['height']), float(box['depth'])])
                iso_batch = prepare_image_fixed_spacing(images, model['classification'], bbox_info[0:3])

                probability, prediction = network_output(iso_batch, model['classification'])

                label = box['class']
                box['label'] = box['class']
                box['class'] = prediction
                box['prob'] = probability

                writer.writerow(box)

                # Statistical results
                if label is not None:
                    if label not in predictions.keys():
                        predictions[label] = dict()
                        predictions[label][prediction] = 1
                    else:
                        if prediction not in predictions[label].keys():
                            predictions[label][prediction] = 1
                        else:
                            predictions[label][prediction] += 1
            process_time = time.time() - begin
            print("Process time:{:.3f}(box:{})".format(process_time, len(case_boxs)))

        except Exception as e:
            print('fails to classify volume: ', imname, ', {}'.format(e))
            continue

    right = 0.0
    for key in predictions.keys():
        try:
            right += predictions[key][key]
        except Exception as e:
            print("Warning: {} were not predicted!".format(e))
    print("accuracy:", right / num_boxs)
    print("prediction results:", predictions)

    
def main():

    from argparse import RawTextHelpFormatter

    long_description = 'UII RTP-Net Testing Engine\n\n' \
                       'It supports multiple kinds of input:\n' \
                       '1. Image list txt file\n' \
                       '2. Single image file\n' \
                       '3. A folder that contains all testing images\n'

    parser = argparse.ArgumentParser(description=long_description,
                                     formatter_class=RawTextHelpFormatter)

    parser.add_argument('-i', '--input', type=str, help='input csv', default="/data/qingzhou/Cls_CAM-main/test_info.csv")
    parser.add_argument('-m', '--model', type=str, help='model dir', default="/data/qingzhou/Cls_CAM-main/lung_malignant/")
    parser.add_argument('-o', '--output', type=str, help='out csv', default="/data/qingzhou/Cls_CAM-main/test_info_res.csv")
    parser.add_argument('-g', '--gpu_id', default='5', help='the gpu id to run model')
    args = parser.parse_args()
    test(args.input, args.model, args.output, int(args.gpu_id))

if __name__ == '__main__':
    main()


