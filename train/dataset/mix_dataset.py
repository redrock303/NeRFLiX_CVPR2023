import cv2
import numpy as np
import os
import sys

import torch
import torch.utils.data as tData
import glob

import random
import json 
import sys
sys.path.append('/newdata/kunzhou/project/rendering/NeRFLiX_package')

from dataset.llff_t import LLFFT_Synthetic
from dataset.vimeo_7f import Vimeo7F_Synthetic


class MixDataset(tData.Dataset):
    def __init__(self, split='train', patch_width=196, patch_height=196, nframes=3,rank=0):

        self.syndatset = Vimeo7F_Synthetic(patch_width=patch_width, patch_height=patch_height,rank=rank,split=split,nframes=nframes)
        self.syndatsetv1 =LLFFT_Synthetic(patch_width=patch_width, patch_height=patch_height,rank=rank,split=split,nframes=nframes)

    def __len__(self):
        return 2000
    def __getitem__(self,index):
        if np.random.random()  >0.3:
            # print('syn')
            return self.syndatset[-1]
        else:
            # print('llff-t')
            return self.syndatsetv1[-1]
        

if __name__ == '__main__':
    import torchvision
    dataset = MixDataset(patch_width=128, patch_height=128)
    for i in range(100):
        nbr,hr,prior= dataset[i]

        print(hr.shape,nbr.shape,prior.shape)
        torchvision.utils.save_image(hr.unsqueeze(0),\
            '{}_ins.png'.format('test'))

        # torchvision.utils.save_image(lr,\
        #     '{}_lr.png'.format('test'))

        torchvision.utils.save_image(nbr,\
            '{}_nbr.png'.format('test'))
        input('check')