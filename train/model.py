import warnings
warnings.filterwarnings("ignore")
import sys 

sys.path.append('/newdata/kunzhou/project/rendering/NeRFLiX_package')
import torch
import torch.nn as nn
import torch.nn.init as init
import functools
import torch.nn.functional as F



import torch
import torch.nn as nn
import torch.nn.init as init
import functools
import torch.nn.functional as F
from utils.modules.basic_conv import *
from utils.modules.charbonnier_loss import L1_Charbonnier_loss
from Sep_STS_Encoder import SepSTSLayer # patch-wise aggregation module
from libs.dcnv2.dcn_v2 import * # for 3090 # pixel-wise aggregation with DCNs

class DCN_sep(DCNv2):
    '''Use other features to generate offsets and masks'''

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1,
                 deformable_groups=1):
        super(DCN_sep, self).__init__(in_channels, out_channels, kernel_size, stride, padding,
                                      dilation, deformable_groups)

        channels_ = self.deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1]
        self.conv_offset_mask = nn.Conv2d(self.in_channels, channels_, kernel_size=self.kernel_size,
                                          stride=self.stride, padding=self.padding, bias=True)
        self.init_offset()

    def init_offset(self):
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def forward(self, input, fea):
        '''input: input features for deformable conv
        fea: other features used for generating offsets and mask'''
        out = self.conv_offset_mask(fea)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)

        # offset_mean = torch.mean(torch.abs(offset))
        # if offset_mean > 100:
        #     logger.warning('Offset mean is {}, larger than 100.'.format(offset_mean))

        mask = torch.sigmoid(mask)
        return dcn_v2_conv(input, offset, mask, self.weight, self.bias, self.stride, self.padding,
                           self.dilation, self.deformable_groups)
def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)
class ResidualBlock_noBN(nn.Module):
    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out
class FeatureEncoder(torch.nn.Module):
    def __init__(self,inc=3, nf=64, N_RB=5):
        super(FeatureEncoder, self).__init__()
        RB = functools.partial(ResidualBlock_noBN, nf=nf)
        self.conv_pre = torch.nn.Sequential(
            nn.Conv2d(inc, nf, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=False),
            nn.Conv2d(nf, nf, 3, 2, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=False),
            # make_layer(RB,2),
        ) 
        

        self.conv_first = torch.nn.Sequential(
            nn.Conv2d(nf, nf, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=False),
            nn.Conv2d(nf, nf, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=False),
            make_layer(RB,2),
        ) 
        self.down_scale1 = torch.nn.Sequential(
            nn.Conv2d(nf, nf, 3, 2, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=False),
            make_layer(RB,2),
        )
        self.down_scale2 = torch.nn.Sequential(
            nn.Conv2d(nf, nf, 3, 2, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=False),
            make_layer(RB,N_RB),
        )


        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=False)

    def forward(self, x,prior=None):
        if prior is not None:
            x = torch.cat([x,prior],1)
        x = self.conv_pre(x)
        fea_d0 = self.lrelu(self.conv_first(x))
        
        fea_d1 = self.down_scale1(fea_d0)
        fea_d2 = self.down_scale2(fea_d1)
        

        return [fea_d0,fea_d1,fea_d2]

class mix_pcd_align(nn.Module):
    def __init__(self,nf=64,groups=8):
        super(mix_pcd_align, self).__init__()
        # L3: level 3, 1/4 spatial size
        self.L3_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.L3_offset_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L3_dcnpack = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1,
                                  deformable_groups=groups)
        
        # L2 
        self.L2_offset_conv1_l3 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.L2_offset_conv2_l3 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.L2_offset_conv3_l3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L2_dcnpack_l3= DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1,
                                  deformable_groups=groups)

        self.L2_offset_conv1 = nn.Conv2d(nf * 3, nf, 3, 1, 1, bias=True)
        self.L2_offset_conv2 = nn.Conv2d(nf * 3, nf, 3, 1, 1, bias=True)
        self.L2_offset_conv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L2_dcnpack = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1,
                                  deformable_groups=groups)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.L1_offset_conv1_l3 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.L1_offset_conv2_l3 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.L1_offset_conv3_l3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L1_dcnpack_l3= DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1,
                                  deformable_groups=groups)

        self.L1_offset_conv1_l2 = nn.Conv2d(nf * 3, nf, 3, 1, 1, bias=True)
        self.L1_offset_conv2_l2 = nn.Conv2d(nf * 3, nf, 3, 1, 1, bias=True)
        self.L1_offset_conv3_l2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L1_dcnpack_l2= DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1,
                                  deformable_groups=groups)
        
        self.L1_offset_conv1 = nn.Conv2d(nf * 4, nf, 3, 1, 1, bias=True)
        self.L1_offset_conv2 = nn.Conv2d(nf * 4, nf, 3, 1, 1, bias=True)
        self.L1_offset_conv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L1_dcnpack = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1,
                                  deformable_groups=groups)


        # Cascading DCN
        self.cas_offset_conv1 = nn.Conv2d(nf * 4, nf, 3, 1, 1, bias=True)
        self.cas_offset_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.cas_dcnpack = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1,
                                   deformable_groups=groups)

        self.fuse =  nn.Conv2d(nf * 3, nf, 3, 1, 1, bias=True)

    def forward(self, nbr_fea_l, ref_fea_l, return_off=False):
        # L3
        L3_offset = torch.cat([nbr_fea_l[2], ref_fea_l[2]], dim=1)
        L3_offset = self.lrelu(self.L3_offset_conv1(L3_offset))
        L3_offset = self.lrelu(self.L3_offset_conv2(L3_offset))
        L3_fea = self.lrelu(self.L3_dcnpack(nbr_fea_l[2], L3_offset))
        # print('L3_fea',L3_fea.shape)

        # pre_align 
        L3_offset = F.interpolate(L3_offset, scale_factor=2, mode='bilinear', align_corners=False)

        nbr_fea_l3_up = F.interpolate(L3_fea, scale_factor=2, mode='bilinear', align_corners=False)
        L2_offset_l3 = torch.cat([nbr_fea_l3_up, ref_fea_l[1]], dim=1)
        L2_offset_l3 = self.lrelu(self.L2_offset_conv1_l3(L2_offset_l3))
        
        L2_offset_l3 = self.lrelu(self.L2_offset_conv2_l3(torch.cat([L2_offset_l3, L3_offset * 2], dim=1)))
        L2_offset_l3 = self.lrelu(self.L2_offset_conv3_l3(L2_offset_l3))
        L2_fea_l3 = self.L2_dcnpack_l3(nbr_fea_l3_up, L2_offset_l3)
        # print('L2_fea_l3',L2_fea_l3.shape)


        L2_offset = torch.cat([nbr_fea_l[1], L2_fea_l3,ref_fea_l[1]], dim=1)
        L2_offset = self.lrelu(self.L2_offset_conv1(L2_offset))
        L2_offset = self.lrelu(self.L2_offset_conv2(torch.cat([L2_offset, L2_offset_l3,L3_offset * 2], dim=1)))
        L2_offset = self.lrelu(self.L2_offset_conv3(L2_offset))
        L2_fea = self.L2_dcnpack_l3(nbr_fea_l[1], L2_offset)

        # print('L2_fea',L2_fea.shape)

        L2_offset = F.interpolate(L2_offset, scale_factor=2, mode='bilinear', align_corners=False)

        L1_fea_l3_up =  F.interpolate(L2_fea_l3, scale_factor=2, mode='bilinear', align_corners=False)
        L1_offset_l3 = torch.cat([L1_fea_l3_up, ref_fea_l[0]], dim=1)
        L1_offset_l3 = self.lrelu(self.L1_offset_conv1_l3(L1_offset_l3))
        L1_offset_l3 =    self.lrelu(self.L1_offset_conv2_l3(torch.cat([L1_offset_l3,L2_offset * 2], dim=1)))
        L1_offset_l3 =   self.lrelu(self.L1_offset_conv3_l3(L1_offset_l3))
        L1_fea_l3 = self.L1_dcnpack_l3(L1_fea_l3_up, L1_offset_l3)
        # print('L1_fea_l3',L1_fea_l3.shape)

        L1_fea_l2_up =  F.interpolate(L2_fea, scale_factor=2, mode='bilinear', align_corners=False)
        L1_offset_l2 = torch.cat([L1_fea_l2_up,L1_fea_l3, ref_fea_l[0]], dim=1)
        L1_offset_l2 =    self.lrelu(self.L1_offset_conv1_l2(L1_offset_l2))
        L1_offset_l2 =    self.lrelu(self.L1_offset_conv2_l2(torch.cat([L1_offset_l2,L1_offset_l3,L2_offset * 2], dim=1)))
        L1_offset_l2 =    self.lrelu(self.L1_offset_conv3_l2(L1_offset_l2))
        L1_fea_l2 = self.L1_dcnpack_l2(L1_fea_l2_up, L1_offset_l2)
        # print('L1_fea_l2',L1_fea_l2.shape)


        L1_offset = torch.cat([nbr_fea_l[0],L1_fea_l3, L1_fea_l2,ref_fea_l[0]], dim=1)
        L1_offset =    self.lrelu(self.L1_offset_conv1(L1_offset))
        L1_offset =    self.lrelu(self.L1_offset_conv2(torch.cat([L1_offset,L1_offset_l2,L1_offset_l3,L2_offset * 2], dim=1)))
        L1_offset =    self.lrelu(self.L1_offset_conv3(L1_offset))
        L1_fea = self.L1_dcnpack(nbr_fea_l[0], L1_offset)

        offset = torch.cat([L1_fea,L1_fea_l3, L1_fea_l2,ref_fea_l[0]], dim=1)
        offset = self.lrelu(self.cas_offset_conv1(offset))
        offset = self.lrelu(self.cas_offset_conv2(offset))

        fea_fuse = self.fuse(torch.cat([L1_fea,L1_fea_l3, L1_fea_l2], dim=1))
        fea = self.lrelu(self.cas_dcnpack(fea_fuse, offset))

        # print('fea',fea.shape)
        return fea,offset

class HR_Align(nn.Module):
    def __init__(self, nf=64, groups=8):
        super(HR_Align, self).__init__()
        self.offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.offset_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.joint_combine = nn.Conv2d(nf*2, nf, 3, 1, 1, bias=True)
        RB_f = functools.partial(ResidualBlock_noBN, nf=nf)
        self.rbs = make_layer(RB_f, 5)

        self.dcnpack = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1,
                                  deformable_groups=groups)

        # self.sep_module = sepConv()
        # self.hr_align = Sep_Align(nf=nf,ks=21)
        self.scaling = torch.nn.Sequential(
            nn.Conv2d(nf*2, nf, 3, 1, 1, bias=True),
            torch.nn.Sigmoid(),
        )

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=False)
    def forward(self,nbr_fea,ref_fea,pre_offset_fea=None):
        _offset = torch.cat([nbr_fea,ref_fea], dim=1)
        _offset = self.lrelu(self.offset_conv1(_offset))
        _offset = self.lrelu(self.offset_conv2(_offset))

        if pre_offset_fea is None:
            offset_fea = torch.cat([_offset,_offset],1)
        else:
            offset_fea_init = torch.cat([_offset,pre_offset_fea],1)
            pre_offset_fea = pre_offset_fea * self.scaling(offset_fea_init)
            offset_fea = torch.cat([_offset,pre_offset_fea],1) 
        offset = self.rbs(self.joint_combine(offset_fea))
        # align_fea = self.lrelu(self.hr_align(nbr_fea, offset,self.sep_module))
        align_fea = self.lrelu(self.dcnpack(nbr_fea, offset))
        return align_fea,offset
    def _decoder(self,nbr_fea,offset):
        align_fea = self.lrelu(self.dcnpack(nbr_fea, offset))
        return align_fea

class SimpleNet(nn.Module):
    def __init__(self, nf=128, front_RB=8, back_RB=40, nbr=2, groups=8):

        super(SimpleNet, self).__init__()
        self.nbr = nbr
        self.nframes = 2 * nbr + 1

        self.charloss = L1_Charbonnier_loss()

        nf1 = 128

        RB_f = functools.partial(ResidualBlock_noBN, nf=nf1)
        self.fea_extract = FeatureEncoder(inc=3,nf=nf, N_RB=5)
        self.denoise_extract = FeatureEncoder(inc=3+2,nf=nf, N_RB=8)

        self.py_align = mix_pcd_align(nf=nf)

        self.hd_align = HR_Align(nf=nf)
       
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=False)

        self.fuse1 = nn.Conv2d(nf * 2 + 2, nf1, 3, 1, 1, bias=True)

        self.recon = make_layer(RB_f, back_RB)

        self.up_conv1 = nn.Conv2d(nf1, 64 * 4, 3, 1, 1, bias=True)
        # self.up_conv2 = nn.Conv2d(64, 64 * 4, 3, 1, 1, bias=True)
        self.hr_conv = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.out_conv = nn.Conv2d(64, 3, 1, 1, bias=True)
        self.ps = nn.PixelShuffle(upscale_factor=2)

        self.l0_attention =  SepSTSLayer(nf, depth=8, num_frames=2, num_heads=8, window_size=(2,8,8))
        self.l0_fuse = nn.Conv3d(nf, nf, kernel_size=3, stride=1, padding=1, bias=False)
        self.fuseAtt = nn.Conv2d(nf * 2 +2, nf, 3, 1, 1, bias=True)

    def _initialize_weights(self, scale=0.1):
        # for residual block
        for M in [self.recon]:
            for m in M.modules():
                if isinstance(m, nn.Conv2d):
                    init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                    m.weight.data *= scale
                    if m.bias is not None:
                        m.bias.data.zero_()
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0.0, std=0.0001)
                    nn.init.constant_(m.bias, 0.0) 
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def resize_up4(self,x):
        return F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)

    def forward(self,x,noise_prior,gt=None,k=3):
        x0,x1,x2 = x[:,0].contiguous(),x[:,1].contiguous(),x[:,2].contiguous()
        # print(x0.shape,x1.shape,x2.shape)
        # print('noise_prior',noise_prior.shape,x.shape)
        b,n,c,h,w = x.size()

        noise_prior = noise_prior.view(b,2,1,1)
        fea_l_d0,fea_l_d1,fea_l_d2 =  self.fea_extract(x0)
        fea_r_d0,fea_r_d1,fea_r_d2  = self.fea_extract(x2)
        fea_m_d0,fea_m_d1,fea_m_d2 =  self.denoise_extract(x1,noise_prior.repeat(1,1,h,w))
        # print(fea_l_d0.shape,fea_l_d1.shape,fea_l_d2.shape)
        # input('check')
        align_m_l,offset_ml = self.py_align([fea_l_d0,fea_l_d1,fea_l_d2],[fea_m_d0,fea_m_d1,fea_m_d2])
        align_m_r,offset_mr = self.py_align([fea_r_d0,fea_r_d1,fea_r_d2],[fea_m_d0,fea_m_d1,fea_m_d2])

        align_m_l_s2,offset_m_l_s2 = self.hd_align(align_m_l,align_m_r,offset_ml*0.5)
        align_m_r_s2,offset_m_r_s2 = self.hd_align(align_m_r,align_m_l,offset_mr*0.5)
        out_fea = torch.cat([align_m_l_s2,align_m_r_s2],1)
        fuse_fea = self.lrelu(self.fuse1(torch.cat([out_fea,noise_prior.repeat(1,1,h//2,w//2)],1)))

        for i in range(k):
            # if i % 2 == 1:
                # align_m_l_s2,offset_m_l_s2 = self.hd_align(align_m_l_s2,align_m_r_s2,offset_m_l_s2*0.5)
                # align_m_r_s2,offset_m_r_s2 = self.hd_align(align_m_r_s2,align_m_l_s2,offset_m_r_s2*0.5)

            l0_lm = self.l0_attention(torch.stack([align_m_l_s2,align_m_r_s2],2))
            l0_rm = self.l0_fuse(l0_lm)

            if i < k-1:
                align_m_l_s2,align_m_r_s2 = l0_rm[:,:,0].contiguous(),l0_rm[:,:,1].contiguous()
            else:
                out_fea1 = l0_rm.view(b,-1,h//2,w//2)
                # print('out_fea1',out_fea1.shape)
                fuse_fea1 = self.lrelu(self.fuseAtt(torch.cat([out_fea1,noise_prior.repeat(1,1,h//2,w//2)],1)))
                # print('fuse_fea1',i,k,fuse_fea1.shape)
        

        recon_fea = self.recon(fuse_fea + fuse_fea1)
        hr_fea = self.lrelu(self.ps(self.up_conv1(recon_fea)))
        residue = self.out_conv(self.lrelu(self.hr_conv(hr_fea))) 
        # abs_v = torch.abs(residue)
        out =x1 + residue #* abs_v

        if gt is not None:
        #     # loss = ((out - gt)**2).sum()
            loss = self.charloss(out,gt)
            return dict(loss = loss )
        return x1,out
        
if __name__ == '__main__':
    import cv2
    import time 
    torch.cuda.set_device(0)
    device = torch.device('cuda')
    net = SimpleNet()
    net.eval()
    net.to(device)
    print("model have {:.3f}M paramerters in total".format(sum(x.numel() for x in net.parameters())/1000000.0))

    ins = torch.rand((1,3, 3, 256, 256)).to(device)
    noise = torch.rand((1,2)).to(device)
    # ins = ins.to(device)
    # census_transform = CensusTransform().to(device)
    # with torch.no_grad():
    #     out = census_transform(ins)
    # print(out.shape)
    # input('check out')
    # pm_loss = PatchMatching()
    N = 50
    st = time.time()
    with torch.no_grad():
        for i in range(N):
            out = net(ins,noise)
    print(out.shape,(time.time() - st)/N)
    