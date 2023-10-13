import copy

import torch.utils.data as data
import os
import datetime
from os.path import join
import torch
import numpy as np
from PIL import Image
import random

class MyDataLoader(data.Dataset):
    def __init__(self, root_dir,split, resize=(224,224),  transform=None,  catId=2, shuffle=True, dist_map=True, label_num=None):
        super(MyDataLoader, self).__init__()
        path_list = open(join(root_dir, split+ '_pair_path_list.txt'),'r').readlines()
        if label_num is not None:
            path_list = path_list[:label_num]
        if shuffle and not split=='test':
            random.shuffle(path_list)
        if split=='test':
            self.istest = True
        else:
            self.istest=False
        self.resize=resize
        self.catId=catId
        self.dist_map=dist_map
        self.t2w_filenames = [x.strip().split(' ')[0] for x in path_list]
        self.tgt_filenames = [x.strip().split(' ')[1] for x in path_list]
        self.adc_filenames = [x.strip().split(' ')[2] for x in path_list]
        self.dwi_filenames = [x.strip().split(' ')[3] for x in path_list]
        self.zon_filenames = [x.strip().split(' ')[4] for x in path_list]
        self.prefix=[os.path.split(x)[-1] for x in self.t2w_filenames]

        assert len(self.t2w_filenames) == len(self.tgt_filenames)

        # report the number of images in the dataset
        print('Number of {0} images: {1} NIFTIs'.format(split, self.__len__()))

        # data augmentation
        self.transform = transform

    def __getitem__(self, index):
        # update the seed to avoid workers sample the same augmentation parameters
        np.random.seed(datetime.datetime.now().second + datetime.datetime.now().microsecond)

        # load the nifti images
        t2w = self.read_image(self.t2w_filenames[index], self.resize)
        tgt = self.read_image(self.tgt_filenames[index], self.resize, label=True)
        adc = self.read_image(self.adc_filenames[index], self.resize)
        dwi = self.read_image(self.dwi_filenames[index], self.resize)
        zon = self.read_image(self.zon_filenames[index], self.resize, label=True)
        #zon=(zon*2).round()
        semantic_masks={}
        # handle exceptions
        data = np.concatenate((t2w[ ..., np.newaxis], adc[ ..., np.newaxis], dwi[ ...,np.newaxis]), axis=-1)

        if self.transform:
            transformed = self.transform(image=data, mask=tgt, mask1=zon)
            data, tgt, zon = transformed['image'], transformed['mask'], transformed['mask1']

        tgt_zon = (zon * tgt).astype(np.float32)
        for ann in range(self.catId):
            semantic_mask = np.zeros_like(t2w, dtype='uint8')
            if self.catId == 3:
                tgt=tgt_zon
            semantic_mask[tgt == ann] = 1
            semantic_masks[ann] = semantic_mask

        prostate=copy.deepcopy(zon)#torch.zeros(size=zon.shape)
        #prostate[zon>0]=1.0 # for prostate roi only
        scribble_data = data*np.concatenate((prostate[...,None], prostate[...,None], prostate[..., None]),axis=-1).astype(np.float32) #np.zeros_like(semantic_mask, dtype='uint8')
        scribble_data = torch.from_numpy(scribble_data.transpose(2,0,1))
        data = torch.from_numpy(data.transpose(2, 0, 1))
        # import matplotlib.pyplot as plt
        # plt.subplot(2,1,1)
        # plt.imshow(semantic_mask[...],cmap='gray')
        # plt.axis('off')
        # plt.subplot(2, 1, 2)
        # plt.imshow(tgt[...],cmap='gray')
        # plt.axis('off')
        # plt.show()
        sample = {'image': scribble_data,#data
                  'label': semantic_masks,
                  'inst': tgt,#'prostate': scribble_data,
                  'zone' : zon,}
                  #'zone_lesion': zone_lesions}

        #image_t = torch.from_numpy(np.array(sample['image']).transpose(2, 0, 1))
        
        # Transform to tensor
        # if self.to_tensor is not None:
        #     sample = self.to_tensor(sample)

        sample['id'] = index
        #sample['image_t'] = image_t
        sample['prefix']='none'

        if self.istest:
            sample['prefix']=self.prefix[index]
            return sample
        else:
            return sample

    def __len__(self):
        return len(self.t2w_filenames)


    def is_image_file(filename):
        return any(filename.endswith(extension) for extension in [".png", ".jpg"])

    def read_image(self, path, inputsize,label=False):
        if label:
            inputsize = inputsize
            types = np.uint8
            interp = Image.NEAREST
        else:
            inputsize = inputsize
            types = np.float32
            interp = Image.BILINEAR
        img = Image.open(path).resize(inputsize, interp)
        img = (np.array(img)).astype(types)
        return img/255


