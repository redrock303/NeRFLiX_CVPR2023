import argparse
import os
import os.path as osp
import logging

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tensorboardX import SummaryWriter
import sys
sys.path.append('/newdata/kunzhou/project/rendering/NeRFLiX_package')
sys.path.append('/newdata/kunzhou/project/package')
sys.path.append('/newdata/kunzhou/project/package_3090')

from config import config
from utils import model_opr



from model import IVM 

import torchvision

from utils.common import calculate_psnr, calculate_ssim
import json 

import numpy as np
import cv2
device = torch.device('cuda')
import glob

def imgtotensor(img):
    img = img.transpose(2,0,1)
    lr_tensor = torch.from_numpy(img.astype(np.float32) / 255.0).float().unsqueeze(0).to(device)
    return lr_tensor

def test(img_lq,prior, model, tile=784, window_size=8,tile_overlap=4):
    if tile is None:
        # test the image as a whole
        output = model(img_lq)
    else:
        # test the image tile by tile
        b, t,c, h, w = img_lq.size()
        # print(h,w,tile)
        tile = min(tile, h, w)
        assert tile % window_size == 0, "tile size should be a multiple of window_size"
        tile_overlap = tile_overlap
        sf = 1

        stride = tile - tile_overlap
        h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
        w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
        E = torch.zeros(b,c, h*sf, w*sf).type_as(img_lq)
        W = torch.zeros_like(E)

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = img_lq[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                
                out_patch = model(in_patch,prior)[-1]
                # print('out_patch',out_patch.shape,in_patch.shape)
                out_patch_mask = torch.ones_like(out_patch)

                E[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch)
                W[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch_mask)
        output = E.div_(W)

    return output

def forward(i0,i1,i2,prior,model):
    ins = torch.stack([i0,i1,i2],1)
    return test(ins,prior,model)
    return model( torch.stack([i0,i1,i2],1),prior)[-1]

def forward_x4(i0,i1,i2,prior,model):
    result_f = forward(i0,i1,i2,prior,model)

    result = forward(torch.flip(i0,(-1,)),torch.flip(i1,(-1,)),torch.flip(i2,(-1,)),prior,model)
    result_f =result_f+ torch.flip(result ,(-1,))

    result = forward(torch.flip(i0,(-2,)),torch.flip(i1,(-2,)),torch.flip(i2,(-2,)),prior,model)
    result_f =result_f+ torch.flip(result ,(-2,))

    result = forward(torch.flip(i0,(-2,-1)),torch.flip(i1,(-2,-1)),torch.flip(i2,(-2,-1)),prior,model)
    result_f =result_f+ torch.flip(result ,(-2,-1))

    
    return 0.25 * result_f

pretrained_weights = './nerflix_weights.pth'

root_result_path = './result_tank_tensorf_nerflix'
if os.path.exists(root_result_path) is False:
    os.mkdir(root_result_path)
tanks_data_folder_path = '/newdata/kunzhou/dataset/vfi_benchmarks/TanksAndTemple'

tensorf_tanks_result_path = '/newdata/kunzhou/dataset/vfi_benchmarks/tanksandtemples/results'

model = IVM(config).to(device)
print("model have {:.3f}M paramerters in total".format(sum(x.numel() for x in model.parameters())/1000000.0))
model_opr.load_model(model,  pretrained_weights, strict=False, cpu=True)



all_object_file = '{}/all.txt'.format(root_result_path)
if os.path.exists(all_object_file) is False:
    with open(all_object_file,'w') as f:
        f.write('-------------result_tensorf---------------- \n')
        

object_list = ['Caterpillar','Ignatius','Barn','Family','Truck']

metrics_dict = {}
for object_name in object_list:
    metrics_dict[object_name] = []
for object_name in object_list:
    root_dir = '{}/{}'.format(tanks_data_folder_path,object_name)
    result_path = '{}/{}'.format(root_result_path,object_name)
    if os.path.exists(result_path) is False:
        os.mkdir(result_path)

    json_file = '{}/matching_data_tankstemple_v1/{}.json'.format(tanks_data_folder_path,object_name)  # os.path.join(root_dir,'test_pairs.json')
    with open(json_file,'r') as f:
        img_pair_dict = json.load(f)
    # print('img_pair_dict',img_pair_dict)
    image_paths = sorted(glob.glob(os.path.join(root_dir, 'rgb/*')))

    test_image_files = sorted(glob.glob(os.path.join('{}/{}'.format(tensorf_tanks_result_path,object_name), '*.png')))
    print('test_image_files',len(test_image_files))
    # input('cc')
    all_rgbimgs = {}

    for i in range(len(image_paths)):
        image_path = image_paths[i]
        pose_name = image_path.split('/')[-1].replace('.png','.txt')
        # print(image_path,pose_name)
        # input('cc')
        img = cv2.imread(image_path)
        all_rgbimgs[pose_name] = [img,image_path]

    psnr_l = []
    ssim_l = []
    psnr_init_l = []
    ssim_init_l = []
    g_idx = 0
    for pose_name in img_pair_dict:
        _ref = test_image_files[g_idx]
        g_idx += 1
        print(_ref,pose_name)
        gt_img,gt_url = all_rgbimgs[pose_name]

        gt_mask = cv2.imread(gt_url,-1)[:,:,-1]
        # print(gt_url,_ref,gt_mask.shape)
        # input('cc')
        img_name = gt_url.split('/')[-1]
        if i == 0:
            i1,i2 = 1,2 
        elif i == len(image_paths) -1:
            i1,i2 = len(image_paths) -2,len(image_paths) -3
        else:
            i1 = i - 1
            i2 = i + 1

        i1,i2 = img_pair_dict[pose_name][:2]
        print('matched image',i1,i2)
        # if i not in [4*8,5*8,6*8]:
            # i1,i2 = img_pair_dict[str(i)]
        # print(i,i1,i2)
        # print(i,i1,i2,img_pair_dict[str(i)])

        

        # _ref = os.path.join('/newdata/kunzhou/project/rendering/svox2-master/opt/ckpt/{}/test_renders/'.format(object_name),\
        #         img_name)

        print('_ref',_ref)
        img1 = cv2.imread(_ref)
        H,W = img1.shape[:2]
        H,W = int(H//8)*8,(W//8)*8
        img1 = img1[:H,:W]

        img0 = all_rgbimgs[i1][0][:H,:W]
        img2=  all_rgbimgs[i2][0][:H,:W]

        gt_img = gt_img[:H,:W]
        gt_mask = gt_mask[:H,:W]

        idx = np.where(gt_mask == 0)
        gt_img[idx[0],idx[1]] = 255

        tensor1 = imgtotensor(img1)
        tensor0 = imgtotensor(img0)
        tensor2 = imgtotensor(img2)

        jpeg_quality = 50 *0.01
        noisy = 0.5 * 0.02  * 90
        prior = np.array([jpeg_quality,noisy])
        prior = torch.from_numpy(prior).to(tensor1.device).float()

        with torch.no_grad():
            # nbr_img = torch.stack([tensor0,tensor1,tensor2],1)[:,:,:-]
            # sr_norm,sr_vsr = model(nbr_img,prior)
            # sr_vsr = forward(tensor0,tensor1,tensor2,prior,model)
            sr_vsr = forward_x4(tensor0,tensor1,tensor2,prior,model)

            # sr_vsr_half1 = sr_vsr_half1.clamp(0,1)

            # sr_vsr_half2 = forward_x4(tensor2,tensor1,tensor0,prior,model)
            # sr_vsr_half2 = sr_vsr_half2.clamp(0,1)

            # sr_vsr = 0.5*sr_vsr_half1 + 0.5 * sr_vsr_half2

            output_new = sr_vsr.detach().cpu().numpy()[0].astype(np.float32)
            output_new = np.transpose(output_new,(1,2,0))

            output_old = tensor1.detach().cpu().numpy()[0].astype(np.float32)
            output_old = np.transpose(output_old,(1,2,0))

        psnr_init = calculate_psnr(output_old*255.0, gt_img)
        psnr_pred = calculate_psnr(output_new*255.0, gt_img)


        ssim_pred = calculate_ssim(output_new*255.0, gt_img)
        ssim_init = calculate_ssim(output_old*255.0, gt_img)

        ipath = os.path.join(result_path,img_name)


        psnr_init_l.append(psnr_init)
        ssim_init_l.append(ssim_init)

        psnr_l.append(psnr_pred)
        ssim_l.append(ssim_pred)

        print('ipath',ipath,psnr_init,ssim_init)
        print('ipath',ipath,psnr_pred,ssim_pred)
        cv2.imwrite(ipath,output_new*255 )

        # print('ipath',ipath,psnr_pred,ssim_pred)
        # input('cc')
        # break 
    avg_psnr_init = sum(psnr_init_l) / len(psnr_init_l)
    avg_ssim_init = sum(ssim_init_l) / len(ssim_init_l)

    avg_psnr = sum(psnr_l) / len(psnr_l)
    avg_ssim = sum(ssim_l) / len(ssim_l)

    print(object_name,'init',avg_psnr_init,avg_ssim_init)
    print(object_name,'refine',avg_psnr,avg_ssim)


    with open(os.path.join(result_path,'metrics.txt'),'a') as f:
        f.write('q:{}n{}\n'.format(jpeg_quality * 100,noisy/(0.02  * 90)))
        f.write('init {}/{} \n'.format(avg_psnr_init,avg_ssim_init))
        f.write('refine {}/{} \n '.format(avg_psnr,avg_ssim))
    metrics_dict[object_name].append([avg_psnr_init,avg_ssim_init,avg_psnr,avg_ssim])

avg_psnr_objects = []
avg_ssim_objects = []
for object_name in metrics_dict:
    print('---------------------------')
    print(object_name,metrics_dict[object_name])
    p0,s0,p1,s1 = metrics_dict[object_name][0]
    avg_psnr_objects.append([p0,p1])
    avg_ssim_objects.append([s0,s1])
    with open(all_object_file,'a') as f:
        f.write('object {} init metrics {}/{} refine metrics {}/{} \n'.format(object_name,p0,s0,p1,s1))

avg_psnr_objects = np.array(avg_psnr_objects).reshape(-1,2)
avg_ssim_objects = np.array(avg_ssim_objects).reshape(-1,2)
print('init',avg_psnr_objects[:,0].mean(),avg_ssim_objects[:,0].mean())
print('refine',avg_psnr_objects[:,1].mean(),avg_ssim_objects[:,1].mean())
with open(all_object_file,'a') as f:
    f.write(' init metrics {}/{} refine metrics {}/{} \n'.format(avg_psnr_objects[:,0].mean(),avg_ssim_objects[:,0].mean(),avg_psnr_objects[:,1].mean(),avg_ssim_objects[:,1].mean()))