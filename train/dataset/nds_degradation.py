from scipy.signal import convolve2d
import json 
import cv2 
import numpy as np
import math 
import random
def getYchannel(url):
    img = cv2.imread(url)
    img_label = cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)[:,:,[0]]
    return img_label
def gen_gaussian(major_size, kernel_size, scale=1.0, ratio=1.0, angle=0, norm=True):

    
    c1, c2 = kernel_size // 2, kernel_size // 2
    sigma1 = major_size // 2 * scale
    sigma2 = sigma1 * ratio
    kernel = np.zeros((kernel_size, kernel_size))
    angle = math.pi * angle / 180.0
    for y in range(kernel_size):
        for x in range(kernel_size):
            delta1 = ((x - c1) * math.cos(angle) - (y - c2) * math.sin(angle)) ** 2 / sigma1 ** 2
            delta2 = ((x - c1) * math.sin(angle) + (y - c2) * math.cos(angle)) ** 2 / sigma2 ** 2
            kernel[y][x] = math.exp(-0.5 * (delta1 + delta2))

    if norm:
        # make sum of each kernel as 1
        kernel /= kernel.sum()

    return kernel
def blurimg(img,kernel):
    chl_l = []
    for i in range(3):
        chl_l.append(convolve2d(img[:, :, i], kernel, mode='same', boundary='symm'))
    return np.stack(chl_l, axis=2)
# def imgProcessing(url,quality=80):
#     img = cv2.imread(url)
#     # quality = random.randint(70,90)
#     _, frame_en = cv2.imencode('.jpg', img_lr.copy(), [int(cv2.IMWRITE_JPEG_QUALITY), quality])
#     img_lr = cv2.imdecode(frame_en, 1)

#     img = cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)[:,:,]
def compress(img,quality):
    _, frame_en = cv2.imencode('.jpg', img.copy(), [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    frame_en = cv2.imdecode(frame_en, 1)
    return frame_en
    # H,W,C = img.shape 
    # prob = np.random.normal(0.5, 0.1, (H,W,C))
    # max_v,min_v = prob.max(),prob.min()
    # prob = 0.5*(prob - min_v)/(max_v - min_v)
    # return frame_en * prob + img *(1-prob)

def imgProcessing(img,quality=80,kernel=None):
    img_lr = img_label.copy() #cv2.cvtColor(img_label.copy(),cv2.COLOR_BGR2YCrCb)[:,:,0]
    return img_label.astype(np.float)/255.0,img_lr.astype(np.float)/255.0
def multiple_blur(img,t=3,coord_map=None):
    H,W,C = img.shape
    weights = [cv2.resize(np.random.normal(1.0/(t+1), 0.08, (H//4,W//4,C)),(W,H))]
    imgs = [img.copy()]
    scale_list = [0.2,0.4,0.8]
    for i in range(t):
        major_size = 2 * random.randint(1,7) + 1
        kernel_size = 3 + random.randint(1,5)*i*2
        # scale = np.random.random() + 0.2 0-0.6 3-15
        scale = scale_list[i]
        ratio = np.random.random() * 0.6
        angle = np.random.random() * 180 
        # print('scale',scale)
        kernel = gen_gaussian(major_size, kernel_size, scale, ratio, angle, norm=True)
        img_b = blurimg(img,kernel)
        if coord_map is None:
            weight = np.random.normal(1.0/(t+1), 0.1, (H//4,W//4,C))
            weight = cv2.resize(weight,(W,H))
        else:
            mask = defineHighlightArea(H,W,coord_map)
            max_v,min_v = mask.max(),mask.min()

            weight = (1.0/(t+1)) *(mask - min_v)/( max_v-min_v+1e-6)
        imgs.append(img_b)
        weights.append(weight)
    img_stack = np.stack(imgs,0)
    weight_stack = np.stack(weights,0)
    
    weight_stack_exp = np.exp(weight_stack)
    weight_stack_exp = weight_stack_exp / np.sum(weight_stack_exp,0,keepdims = True)
    img_f = (weight_stack_exp * img_stack).sum(0)

    # lr_mask = np.random.normal(0.0, 0.1, (H//4,W//4))
    # noise_mask = cv2.resize(lr_mask,(W,H))
    # valid_idx = np.where(noise_mask<0.2)
    # img_f[valid_idx[0],valid_idx[1]] = img[valid_idx[0],valid_idx[1]]

    return img_f
    # 
    # prob = np.random.normal(0.5, 0.1, (H,W,C))
    # max_v,min_v = prob.max(),prob.min()
    # prob = (prob - min_v)/(max_v - min_v)
    # # print('---prob--',prob.max(),prob.min(),prob.mean())
    # return img_f * prob + img *(1-prob)

def process(img,coord_map):

    # img = cv2.imread('/newdata/kunzhou/dataset/vfi_benchmarks/nerf_llff_data/fern/images_4/image001.png')
    H,W,C = img.shape 

    # img = color_jet(img)

    noise_level = np.random.random() * 0.02
    noise = np.random.normal(0.0, noise_level*255, (H,W,C))
    img_processed = np.clip(img + noise,0,255)

    # lr_mask = np.random.normal(0.0, 0.1, (H//4,W//4))
    # noise_mask = cv2.resize(lr_mask,(W,H))
    # valid_idx = np.where(noise_mask<0.1)
    # img_processed[valid_idx[0],valid_idx[1]] = img[valid_idx[0],valid_idx[1]]
    mask = np.zeros((H,W,3))
    for i in range(2):
        mask += defineHighlightArea(H,W,coord_map)
    max_v,min_v = mask.max(),mask.min()
    mask = (mask - min_v)/( max_v-min_v+1e-6)
    img_processed = mask * img_processed + (1-mask)*img.copy()


    img_processed = multiple_blur(img_processed,3,coord_map)

    jpeg_quality = random.randint(20,95)
    img_processed = compress(img_processed,jpeg_quality)
    # img_processed = compress(img_processed,15)

    img_processed = reposition(img_processed,ratio = 0.15)

    img_processed = multiple_blur(img_processed,3)

    return img_processed,jpeg_quality*0.01,noise_level*90
def reposition(img,ratio = 0.08):
    H,W,C = img.shape 
    mask = np.abs(np.random.normal(0.0, 0.1, (H,W)))
    valid_idx = np.where(mask>ratio)
    valid_idx_0 = valid_idx[0] + np.random.normal(0, 1, (valid_idx[0].shape[0])).astype(np.int)
    valid_idx_1 = valid_idx[1] + np.random.normal(0, 1, (valid_idx[1].shape[0])).astype(np.int)
    valid_idx_0 = np.clip(valid_idx_0,0,H-1)
    valid_idx_1 = np.clip(valid_idx_1,0,W-1)
    # print('valid_idx_0',valid_idx_0.shape,valid_idx_0.max(),valid_idx_0.min())
    img[valid_idx[0],valid_idx[1]] = img[valid_idx_0,valid_idx_1]
    return img
def add_blocknoise(img):
    H,W,C = img.shape 
    mask = np.abs(np.random.normal(0.0, 0.1, (H,W)))
    valid_idx = np.where(mask>0.3)
    img[valid_idx[0],valid_idx[1]] = np.random.random() *0.1*img.max()
    return img
def color_jet(image):
    scale = None
    for n_c in range(3):
        if scale is None:
            scale = random.uniform(0.95, 1.05)
        else:
            scale = random.uniform(scale - 0.02, scale + 0.02)
        image[:, :,n_c] = np.clip(np.power(image[:, :,n_c],scale), 0, 255)

    return image
def defineHighlightArea(H,W,coord_map):

    mask = np.zeros((H*W,3))

    max_size = max(H,W)
    c1, c2 = random.randint(0,max_size),random.randint(0,max_size)

    major_size = random.randint( int( max_size* 0.05)*2+1, int( max_size* 0.1)*2+1) 
    sigma1 = major_size
    sigma2 = sigma1 * np.random.random()
    angle = math.pi * np.random.random()

    
    delta1 = ((coord_map[:,0] - c1) * math.cos(angle) - (coord_map[:,1] - c2) * math.sin(angle)) ** 2 / sigma1 ** 2
    delta2 = ((coord_map[:,0] - c1) * math.sin(angle) + (coord_map[:,1] - c2) * math.cos(angle)) ** 2 / sigma2 ** 2
    # print(delta1.shape,delta2.shape,H*W)
    for k in range(3):
        mask[:,k] = np.exp(-0.5 * (delta1 + delta2))
    mask = mask.reshape((H,W,3))
    
    max_v,min_v = mask.max(),mask.min()
    mask = (mask - min_v) / (max_v - min_v + 1e-6)
    mask = np.clip(mask,0,1)
    return mask 
def defineCoorMap(H,W):
    coord_map = np.zeros((H,W,2))
    for y in range(H):
        for x in range(W):
            coord_map[y,x,0] = x
            coord_map[y,x,1] = y
    coord_map = coord_map.reshape((-1,2))
    return coord_map
def graytocolor(img,maxmin):
    # print('maxmin',maxmin)
    img = (img - maxmin[1])/(maxmin[0] - maxmin[1])
    print('img mean',img.mean())
    img = np.clip(img*255,0,255).astype(np.uint8)
    # print('img',img.shape)
    icolor = cv2.applyColorMap(img,colormap=cv2.COLORMAP_JET)
    return icolor
