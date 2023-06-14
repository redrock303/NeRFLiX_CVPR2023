import cv2
import os

import torch
import torchvision
from utils.common import tensor2img, calculate_psnr, calculate_ssim, bgr2ycbcr

import numpy as np
def validate(model, val_loader, device, iteration, sisr_net=None,down=4, to_y=True, save_path='.', save_img=False, max_num=5):
    # for batch=1
    viz_img = []
    psnr_l = []
    ssim_l = []
    print('val_loader',val_loader.__len__())
    for idx, batch_data in enumerate(val_loader):
        if idx >= max_num:
            break
        lr_img = batch_data[0].to(device)
        hr_img = batch_data[1].to(device)

        noise_prior = batch_data[2].to(device)
       
        # print('lr_img',lr_img.shape,hr_img.shape)

        # loss_dict = model(lr_img,noise_map,hr_img)

        with torch.no_grad():
            sr_norm,sr_vsr = model(lr_img,noise_prior)
            sr_vsr = sr_vsr.clamp(0,1)
            
        

        output_old = sr_norm.detach().cpu().numpy()[0].astype(np.float32)
        output_new = sr_vsr.detach().cpu().numpy()[0].astype(np.float32)

        # lr_img_data = lr_img.cpu().numpy()[0].astype(np.float32)

        output_old = np.transpose(output_old,(1,2,0))
        output_new = np.transpose(output_new,(1,2,0))


        gt = hr_img.cpu().numpy()[0].astype(np.float32)
        gt = np.transpose(gt,(1,2,0))

        # lr_img_data = np.transpose(lr_img_data,(1,2,0))

        if True:
            save_img = np.hstack([output_old[:,:]*255.0,output_new[:,:]*255.0,gt[:,:]*255.0])
            viz_img.append(save_img.copy())
            # wpath = os.path.join(save_path, '%d_%d_cat_check.png' % (iteration,idx))
            # cv2.imwrite(wpath, save_img)
        if True:
            output_new = bgr2ycbcr(output_new, only_y=True)
            gt = bgr2ycbcr(gt, only_y=True)
            

        output_new = output_new[2:-2,2:-2]
        gt = gt[2:-2,2:-2]
        psnr = calculate_psnr(output_new*255.0, gt*255.0)
        ssim = calculate_ssim(output_new*255.0, gt*255.0)
        psnr_l.append(psnr)
        ssim_l.append(ssim)

    avg_psnr = sum(psnr_l) / len(psnr_l)
    avg_ssim = sum(ssim_l) / len(ssim_l)

    print(avg_psnr)
    # 
    
    ipath = os.path.join(save_path, '%d_cat_check.png' % (iteration))
    saveImg = np.vstack(viz_img)
    cv2.imwrite(ipath, saveImg)

    # input('check')
    return avg_psnr,avg_ssim


