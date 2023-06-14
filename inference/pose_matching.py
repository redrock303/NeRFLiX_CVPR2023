import numpy as np 
import  glob 
import os
import torch 
import math
import json

def view_selection(train_poses,train_image_root,rendered_poses,rendered_image_root):

    pose = np.load(train_poses)
    query_pose = np.load(rendered_poses)


    reference_images = sorted(glob.glob(os.path.join(train_image_root,'*.png')))
    print(len(reference_images),pose.shape)


    data_dict = {}
    reference_images = sorted(glob.glob(os.path.join(train_image_root,'*.png')))
    print(len(reference_images),pose.shape)


    for i in range(query_pose.shape[0]):
        data_dict[str(i)] = {}

        data_dict[str(i)]['url'] = os.path.join(rendered_image_root,'{}.png'.format(str(i).zfill(5)))
        data_dict[str(i)]['ref'] = []
        c2w_mat = query_pose[i]
        n0  = np.matmul(c2w_mat[:3,:3],np.array([[0,0,1]]).T).T
        n0 = n0/np.linalg.norm(n0)

        t0 = c2w_mat[:3,3]
        print(n0,t0)

        c2c_dist = []
        normal_dot = []

        for j in range(pose.shape[0]):
            c2w_mat_j = pose[j]
            n1  = np.matmul(c2w_mat_j[:3,:3],np.array([[0,0,1]]).T).T
            n1 = n1/np.linalg.norm(n1)
            t1 = c2w_mat_j[:3,3]

            normal_dot.append( 1.0 - (n0 * n1).sum())
            c2c_dist.append( math.sqrt( ((t0 - t1)**2).sum() ))

            
            # print(normal_dot[-1]*80,c2c_dist[-1])
            # input('cc')
        c2c_dist = np.array(c2c_dist)
        normal_dot = np.array(normal_dot)
        
        weights = c2c_dist +  50*normal_dot
        score = torch.from_numpy(weights)
        score = 1.0/(score+1e-6)
        # print('score',score.shape)
        weight_topK, ind_topK = torch.topk(score, k=8,dim=0) # [B, H] score.size()[0]
        ind_topK = ind_topK.numpy().astype(np.int64)
        
        
        for jk in ind_topK:
            jd_ind = jk  
            
            print(weights[jk],jk)
            url_ref = reference_images[jk]
            data_dict[str(i)]['ref'] .append([str(jk),url_ref])
            

    return data_dict

train_poses = './test_samples/llff_fern_novelviews/poses.npy'
train_image_root = '/newdata/kunzhou/dataset/vfi_benchmarks/nerf_llff_data/fern/images_4'
rendered_poses = './test_samples/llff_fern_novelviews/query_pose.npy'
rendered_image_root = './test_samples/llff_fern_novelviews/novel_images'
data_dict = view_selection(train_poses,train_image_root,rendered_poses,rendered_image_root)
with open(os.path.join('{}/test_idx_novelview.json'.format('/newdata/kunzhou/project/rendering/NeRFLiX_package/inference/test_samples')),'w') as f:
    json.dump(data_dict,f, indent=2)