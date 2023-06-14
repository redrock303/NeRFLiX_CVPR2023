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

# size = 48
from dataset.nds_degradation import *
def aug_moving(stacked_img,x,y,ph,pw):
    n,h,w = stacked_img.shape[:3]
    # print('n,h,w',n,h,w,x,y)
    step_x = random.randint(-1,1)
    step_y = random.randint(-1,1)
    # print('-------------')
    # print('vol',step_x,step_y)
    img_moving = []
    for idx in range(n):
        if idx!= n //2:
            dis = np.abs(n //2 -idx)
            if idx<n//2:
                dx =step_x* dis
                dy =step_y* dis
            else:
                dx = -step_x* dis
                dy = -step_y* dis

            # print(idx,dx,dy)
            bx = x + dx
            by = y + dy 
            bx = max(0,bx)
            bx = min(bx,w-1)

            by = max(0,by)
            by = min(by,h-1)

            # print('0',bx,by)
            pad_x = w - (bx + pw)
            if pad_x<0:
                bx = bx + pad_x 
            pad_y = h - (by + ph)
            if pad_y<0:
                by = by + pad_y

            # print(pad_x,pad_y,x,y,bx,by)
            img_crop = stacked_img[idx,by:by+ph,bx:bx+pw]
        else:
            img_crop = stacked_img[idx,y:y+ph,x:x+pw]
        # print('img_crop',img_crop.shape)
        img_moving.append(img_crop)
    return np.stack(img_moving,0) 
class Vimeo7F_Synthetic(tData.Dataset):
    def __init__(self, dataPath = '/newdata/kunzhou/dataset/vimeo_septuplet',nframes=3,
                 split='train', patch_width=196, patch_height=196,rank=0):
        self.dataPath = dataPath 
        self.split = split

        self.nframes = nframes
        self.patch_width = patch_width
        self.patch_height = patch_height
        self.rank = rank

        self.coord_map_dict ={'{}-{}'.format(patch_height,patch_width):defineCoorMap(patch_height,patch_width)} 
        self.load_data()
    def load_data(self):
        if 'train' in self.split:
            sep_txt = os.path.join(self.dataPath,'sep_trainlist.txt')
        else:
            sep_txt = os.path.join(self.dataPath,'sep_testlist.txt')
        self.file_folder_list = []
        with open(sep_txt,'r') as f:
            while True:
                _lineStr = f.readline().strip('\n')
                if len(_lineStr) < 2:
                    break
                folder_path = os.path.join(self.dataPath,'sequences',_lineStr)
                # # print('folder_path',folder_path,os.path.exists(folder_path))
                # file_list = sorted(glob.glob(folder_path+'/*.*'))
                self.file_folder_list.append(folder_path)
        if 'train' in self.split and self.rank!=0:
            random.shuffle (self.file_folder_list)

    def __len__(self):
        return len(self.file_folder_list)
    def __getitem__(self,index):
        if index == -1:
            index = random.randint(0,self.__len__()-1)

        idx_list_full =[i for i in range(1,8)]
        random.shuffle(idx_list_full)

        idx_list = idx_list_full[:self.nframes]

        folder_path = self.file_folder_list[index]

        

        if 'train' in self.split:
            if np.random.random() > 0.5:
                idx_list.reverse()
        else:
            idx_list = [1,2,3]
        frame_hr = [cv2.imread(os.path.join(folder_path,'im{}.png'.format(idx))) for idx in idx_list] 

        try:
            frame_sta = np.stack(frame_hr,0)
        except Exception as e:
            print(e.args,index,idx_list,folder_path)
        

        h,w,c = frame_sta[0].shape
        if 'train' in self.split:
            x = random.randint(0,w-self.patch_width-1)
            y = random.randint(0,h-self.patch_height-1)

            lr_data = aug_moving(frame_sta,x,y,self.patch_height,self.patch_width) #n h w 3
            # lr_data = frame_sta[:,y:y+self.patch_height, x:x+self.patch_width]


            if np.random.random() > 0.5:
                lr_data = lr_data[:,:, ::-1, :]

            # vertical flip
            if np.random.random() > 0.5:
                lr_data = lr_data[:,::-1, :, ]

            # rotate
            if np.random.random() > 0.5:
                lr_data = lr_data.transpose(0,2,1,3)
        else:
            lr_data = frame_sta.copy()


        H,W = lr_data[0].shape[:2]
        key = '{}-{}'.format(H,W)
        if key in self.coord_map_dict:
            coord_map = self.coord_map_dict[key]
        else :
            self.coord_map_dict[key] = defineCoorMap(H,W)
            coord_map = self.coord_map_dict[key]
        mask = defineHighlightArea(H,W,coord_map.copy())
        
        # print('mask',mask.max(),mask.min())
        for j in range(3):
            tmp = color_jet(lr_data[j])
            lr_data[j] = lr_data[j] * (1-mask) + mask * tmp


        hr_data = lr_data[1].copy()
        lr_data[1],jpeg_quality,noise_level = process(lr_data[1],coord_map.copy())
        lr_data[1] = reposition(lr_data[1],ratio=0.3)

        # if 'train' not in self.split:
        #     cv2.imwrite('finalcolor_{}.png'.format(index),lr_data[j])
        # print('lr_data',lr_data.shape)
        lr_data = lr_data.transpose(0,3,1,2)
        hr_data = hr_data.transpose(2,0,1)

        lr_tensor = torch.from_numpy(lr_data.astype(np.float32) / 255.0).float()
        hr_tensor = torch.from_numpy(hr_data.astype(np.float32) / 255.0).float()

        # nbr_tensor = lr_tensor[[0,2]]
        # ref_lr_tensor = imresize(lr_tensor[[1]], scale= self.scale)[0]
        noise_prior = np.array([jpeg_quality,noise_level])
        noise_prior = torch.from_numpy(noise_prior).float()
        return lr_tensor,hr_tensor,noise_prior
if __name__ == '__main__':
    import torchvision
    dataset = Vimeo_Synthetic()
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
