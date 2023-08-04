# -*- coding:utf-8 -*-
import os
import csv
import numpy as np
import torch
from collections import namedtuple
from torch.utils.data import Dataset
import random
import codecs
import SimpleITK as sitk
from utils.tools import *
from utils.Normalizer import *

def readlines(file):
    """
    read lines by removing '\n' in the end of line
    :param file: a text file
    :return: a list of line strings
    """
    fp = codecs.open(file, 'r', encoding='utf-8')
    linelist = fp.readlines()
    fp.close()
    for i in range(len(linelist)):
        linelist[i] = linelist[i].rstrip('\n')
    return linelist

def read_classification_csv_mask(csv_file, input_channels):
    """
    :param csv_file:        csv file path
    :param input_channels:  the number of input image in one case
    :return: return image path list, label name and bounding box information
    """
    ims_list, labels, bbox_info_list, masks_list = [], [], [], []
    with open(csv_file, 'r') as fp:
        reader = csv.reader(fp)
        head = next(reader)
        for i in range(len(head)):
            if head[i] == "class":
                head[i] = 'type'
        Row = namedtuple('Row', head)

        for line in reader:
            row = Row(*line)
            im_list = list()
            for i in range(input_channels):
                if i == 0:
                    im_list.append(row.__getattribute__("image_path"))
                else:
                    im_list.append(row.__getattribute__("image_path"+str(i)))
            ims_list.append(im_list)
            masks_list.append(row.mask_path)
            bbox_info_list.append([float(row.x), float(row.y), float(row.z),
                                   float(row.width), float(row.height), float(row.depth)])
            try:
                labels.append(str(row.type))
            except Exception as e:
                print(e)
                labels.append(None)

    return ims_list, labels, bbox_info_list, masks_list


def resize_image_itk(ori_img, target_Size, target_Spacing, resamplemethod=sitk.sitkNearestNeighbor, pixel_type=sitk.sitkFloat32):
    # target_Size = target_img.GetSize()      # 目标图像大小  [x,y,z]
    # target_Spacing = target_img.GetSpacing()   # 目标的体素块尺寸    [x,y,z]
    target_origin = ori_img.GetOrigin()      # 目标的起�?[x,y,z]
    target_direction = ori_img.GetDirection()  # 目标的方�?[�?�?横]=[z,y,x]

    # itk的方法进行resample
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ori_img)  # 需要重新采样的目标图像
    # 设置目标图像的信�?    
    resampler.SetSize(target_Size)		# 目标图像大小
    resampler.SetOutputOrigin(target_origin)
    resampler.SetOutputDirection(target_direction)
    resampler.SetOutputSpacing(target_Spacing)
    # 根据需要重采样图像的情况设置不同的dype
    if resamplemethod == sitk.sitkNearestNeighbor:
        # resampler.SetOutputPixelType(sitk.sitkUInt16)   # 近邻插值用于mask的，保存uint16
        resampler.SetOutputPixelType(pixel_type)
    else:
        resampler.SetOutputPixelType(sitk.sitkFloat32)  # 线性插值用于PET/CT/MRI之类的，保存float32
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))    
    resampler.SetInterpolator(resamplemethod)
    itk_img_resampled = resampler.Execute(ori_img)  # 得到重新采样后的图像

    return itk_img_resampled

def center_crop(image, coord, size, padtype=0, padvalue=0):
    """
    crop a sub-volume centered at voxel.
    :param image:       an image3d object
    :param coord:       the coordinate of center voxel
    :param spacing:     spacing of output volume
    :param size:        size of output volume
    :param padtype:     padding type, 0 for value padding and 1 for edge padding
    :param padvalue:    the default padding value for value padding
    :return: a sub-volume image3d object
    """

    assert padtype in [0, 1], 'padtype not support'
    coord = np.array(coord, dtype=np.int32)
    data = sitk.GetArrayFromImage(image)
    #slice_data = data[coord[2],:,:]
    data_size = data.shape
    new_center = np.array([coord[2], coord[1], coord[0]], dtype=np.int32)
    if padtype == 0:
        pad_mode = 'constant'
    else:
        pad_mode = 'edge'
        
    pad_data = np.pad(data, 
                      ((max(0, int(size[0]//2-new_center[0])), max(int(size[0]//2+size[0]%2+new_center[0]-data_size[0]), 0)),
                       (max(0, int(size[1]//2-new_center[1])), max(int(size[1]//2+size[1]%2+new_center[1]-data_size[1]), 0)),
                       (max(0, int(size[2]//2-new_center[2])), max(int(size[2]//2+size[2]%2+new_center[2]-data_size[2]), 0))
                       
                      ),
                      mode = pad_mode,
                      constant_values = (padvalue, padvalue)
                      )
    new_center_pad = [new_center[0]+max(0, int(size[0]//2-new_center[0])), new_center[1]+max(0, int(size[1]//2-new_center[1])), new_center[2]+max(0, int(size[2]//2-new_center[2]))]
    
    crop_data = pad_data[-size[0]//2+new_center_pad[0]:size[0]//2+size[0]%2+new_center_pad[0],
                         -size[1]//2+new_center_pad[1]:size[1]//2+size[1]%2+new_center_pad[1],
                         -size[2]//2+new_center_pad[2]:size[2]//2+size[2]%2+new_center_pad[2]]

    return crop_data


class ClassificationDataset(Dataset):
    """ training data set for volumetric classification """

    def __init__(self, im_classification_list, input_channels, label_name_index,
                 crop_size, crop_normalizers,
                 spacing, default_values,
                 random_translation, interpolation
                ):
        """ constructor
        :param imlist_file: image-mask list file
        :param num_classes: the number of classes
        :param spacing: the resolution, e.g., [1, 1, 1]
        :param crop_size: crop size, e.g., [96, 96, 96]
        :param default_values: default padding value list, e.g.,[0]
        :param random_translation: random translation
        :param interpolation: 'LINEAR' for linear interpolation, 'NN' for nearest neighbor
        :param crop_normalizers: used to normalize the image crops, one for one image modality
        """
        if im_classification_list.endswith('csv'):
            self.ims_list, self.labels, self.bbox_info_list, self.masks_list = read_classification_csv_mask(im_classification_list, input_channels)
        else:
            raise ValueError('im_classification_list must be a csv file')

        self.default_values = default_values

        self.spacing = np.array(spacing, dtype=np.double)

        self.crop_size = np.array(crop_size, dtype=np.int32)

        self.label_name_index = label_name_index

        self.random_translation = np.array(random_translation, dtype=np.double)
        assert self.random_translation.size == 3, 'Only 3-element of random translation is supported'

        assert interpolation in ('LINEAR', 'NN'), 'interpolation must either be a LINEAR or NN'
        self.interpolation = interpolation

        assert isinstance(crop_normalizers, list), 'crop normalizers must be a list'
        self.crop_normalizers = crop_normalizers


    def __len__(self):
        """ get the number of images in this data set """
        return len(self.ims_list)

    def num_modality(self):
        """ get the number of input image modalities """
        return len(self.ims_list[0])         

    def __getitem__(self, index):

        images_path, label, bbox_info, mask_path = self.ims_list[index], self.labels[index], self.bbox_info_list[index], self.masks_list[index]

        case_name = os.path.basename(os.path.dirname(images_path[0]))
        case_name += '_' + os.path.basename(images_path[0])
        images = []
        for image_path in images_path:
            image = sitk.ReadImage(image_path, outputPixelType=sitk.sitkFloat32)
            images.append(image)

        seg = sitk.ReadImage(mask_path, outputPixelType=sitk.sitkFloat32)
               
        ori_origin = images[0].GetOrigin()
        ori_spacing = images[0].GetSpacing()
        ori_direction = images[0].GetDirection()

        spacing = self.spacing
        output_shape = (np.array(seg.GetSize()) * ori_spacing / spacing + 0.5).astype(np.int32)

        for idx in range(len(images)):
            if self.crop_normalizers[idx] is not None:
                if self.crop_normalizers[idx]['modality'] == 'CT':
                    normalizer = Fix_normalizer(float(self.crop_normalizers[idx]['mean']), float(self.crop_normalizers[idx]['stddev']), self.crop_normalizers[idx]['clip'])
                    norm_data = normalizer(sitk.GetArrayFromImage(images[idx]))
                    image = sitk.GetImageFromArray(norm_data)
                    image.SetOrigin(ori_origin)
                    image.SetSpacing(ori_spacing)
                    image.SetDirection(ori_direction)
                elif self.crop_normalizers[idx]['modality'] == 'MR':
                    normalizer = Adaptive_normalizer(float(self.crop_normalizers[idx]['min_p']), float(self.crop_normalizers[idx]['max_p']), self.crop_normalizers[idx]['clip'])
                    norm_data = normalizer(sitk.GetArrayFromImage(images[idx]))         
                    image = sitk.GetImageFromArray(norm_data)
                    image.SetOrigin(ori_origin)
                    image.SetSpacing(ori_spacing)
                    image.SetDirection(ori_direction)
            images[idx] = image
                      
        for idx in range(len(images)):
            if self.interpolation == 'NN':
                image = resize_image_itk(images[idx], output_shape.tolist(), spacing.tolist())
            elif self.interpolation == 'LINEAR':
                image = resize_image_itk(images[idx], output_shape.tolist(), spacing.tolist(), resamplemethod=sitk.sitkLinear)
            images[idx] = image
        
        seg = resize_image_itk(seg, output_shape.tolist(), spacing.tolist())
        
        center = bbox_info[0:3]
        voxel_translation = self.random_translation / ori_spacing[:3]
        trans = np.random.uniform(-voxel_translation, voxel_translation, size=[3]).astype(np.int16)
        center += trans

        for idx in range(len(images)):
            images[idx] = center_crop(images[idx], center, self.crop_size, padvalue=self.default_values[idx])

        seg = center_crop(seg, center, self.crop_size, padvalue=0)   

        axis = random.choice([0, 1, 2, 3, 4, 5])
        if axis in [0, 1, 2]:
            for idx in range(len(images)):
                images[idx] = np.flip(images[idx], axis)

            seg = np.flip(seg, axis)

        # convert to tensors
        im = torch.from_numpy(np.array(images))
        seg = torch.from_numpy(np.array([seg]))

        label = torch.Tensor([self.label_name_index[label]])

        return im, seg, label, case_name




