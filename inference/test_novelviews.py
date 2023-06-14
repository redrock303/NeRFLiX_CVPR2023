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



from model import IVM as IVM

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
def forward(i0,i1,i2,prior,model):
    return model( torch.stack([i0,i1,i2],1),prior)
def forward_x4(i0,i1,i2,prior,model):
    result_f = forward(i0,i1,i2,prior,model)

    result = forward(torch.flip(i0,(-1,)),torch.flip(i1,(-1,)),torch.flip(i2,(-1,)),prior,model)
    result_f =result_f+ torch.flip(result ,(-1,))

    result = forward(torch.flip(i0,(-2,)),torch.flip(i1,(-2,)),torch.flip(i2,(-2,)),prior,model)
    result_f =result_f+ torch.flip(result ,(-2,))

    result = forward(torch.flip(i0,(-2,-1)),torch.flip(i1,(-2,-1)),torch.flip(i2,(-2,-1)),prior,model)
    result_f =result_f+ torch.flip(result ,(-2,-1))

    
    return 0.25 * result_f

model = IVM(config).to(device)
print("model have {:.3f}M paramerters in total".format(sum(x.numel() for x in model.parameters())/1000000.0))
pretrained_weights = './nerflix_weights.pth'
model_opr.load_model(model, pretrained_weights , strict=False, cpu=True)

# step 1 run a nerf model and save the novel view camera poses(RT n 4 4) and the candidation input views (m,4,4) as well as the rendered views(images)
# step 2 perform view selection and get two reference views for enhancing a novel view using IVM model with the saved novel-view cameras and input cameras 

# I have given a pose matching script here for illustration 
result_path = './test_samples/llff_fern_novelviews/llff_fern_novelview_enc'
if not os.path.exists(result_path):
    os.mkdir(result_path)

# run novel view enhancement 
with open(os.path.join('./test_samples/test_idx_novelview.json'),'r') as f:
    data_dict = json.load(f)

for key in data_dict:
    _ref = data_dict[key]['url']
    img_name = _ref.split('/')[-1]

    _nbr_views = data_dict[key]['ref'][:2] # by default we only take the top-2 reference views 
    print(_ref,_nbr_views)
    img1 = cv2.imread(_ref)
    H,W = img1.shape[:2]
    H,W = int(H//8)*8,(W//8)*8
    img1 = img1[:H,:W]
    tensor1 = imgtotensor(img1)

    img0 = cv2.imread(_nbr_views[0][1]) [:H,:W]
    img2 = cv2.imread(_nbr_views[1][1]) [:H,:W]
    
    tensor0 = imgtotensor(img0)
    tensor2 = imgtotensor(img2)
    # print(tensor1.shape,tensor0.shape,tensor2.shape)
    # input('cc')
    
    jpeg_quality = 50 *0.01
    noisy = 0.5 #* 0.02  * 90

    prior = np.array([jpeg_quality,noisy])
    prior = torch.from_numpy(prior).to(tensor1.device).float()

    with torch.no_grad():
        enhanced_view = forward_x4(tensor0,tensor1,tensor2,prior,model)

    enhanced_view = enhanced_view.detach().cpu().numpy()[0].astype(np.float32)
    enhanced_view = np.transpose(enhanced_view,(1,2,0))
    ipath = os.path.join(result_path,img_name)
    cv2.imwrite(ipath,enhanced_view*255 )

