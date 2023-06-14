
import torch
import torchvision
from torch import nn
import torch.nn.functional as F
import numpy as np
import os
class ConvHead(torch.nn.Module):
    def __init__(self,inChannel=3,outChannel=128,dilation=1,BatchNorm = torch.nn.BatchNorm2d):
        super(ConvHead,self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=inChannel, out_channels=64, kernel_size=3, stride=2,
                            padding=(3 - 1) // 2, dilation=dilation),
            BatchNorm(64),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1,
                            padding=(3 - 1) // 2, dilation=dilation),
            BatchNorm(64),
            torch.nn.Conv2d(in_channels=64, out_channels=outChannel, kernel_size=3, stride=1,
                            padding=(3 - 1) // 2, dilation=dilation),
            BatchNorm(outChannel),
            torch.nn.ReLU()
        )
    def forward(self,x ):
        # print(x.shape)
        y = self.conv(x)
        # print(y.shape)
        return y

class ResDenseBlock(torch.nn.Module):
    def __init__(self,inChannel=128,outChannel=128,BatchNorm = torch.nn.BatchNorm2d):
        super(ResDenseBlock,self).__init__()
        self.conv_head = torch.nn.Sequential(
            # torch.nn.Conv2d(in_channels=inChannel, out_channels=inChannel//2, kernel_size=3, stride=1,
            #                 padding=(3 - 1) // 2, dilation=1),
            # BatchNorm(inChannel//2),
            torch.nn.Conv2d(in_channels=inChannel, out_channels=inChannel//2, kernel_size=3, stride=1,
                            padding=(3 - 1) // 2, dilation=1),
            BatchNorm(inChannel//2),
            torch.nn.ReLU()
        )
        self.conv_line_0 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=inChannel//2, out_channels=inChannel//2, kernel_size=3, stride=1,
                            padding=(3 - 1) // 2, dilation=1),
            BatchNorm(inChannel//2),
            torch.nn.Conv2d(in_channels=inChannel//2, out_channels=inChannel//2, kernel_size=3, stride=1,
                            padding=(3 - 1) // 2, dilation=1),
            BatchNorm(inChannel//2),
            torch.nn.ReLU()
        )
        self.conv_line_1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=inChannel // 2, out_channels=inChannel // 2, kernel_size=3, stride=1,
                            padding=(3 - 1) // 2*2, dilation=2),
            BatchNorm(inChannel // 2),
            torch.nn.Conv2d(in_channels=inChannel // 2, out_channels=inChannel // 2, kernel_size=3, stride=1,
                            padding=(3 - 1) // 2 *2, dilation=2),
            BatchNorm(inChannel // 2),
            torch.nn.ReLU()
        )
        self.conv_last = torch.nn.Sequential(

            torch.nn.Conv2d(in_channels=inChannel, out_channels=inChannel, kernel_size=1, stride=1,
                            padding=0 , dilation=1),
            BatchNorm(inChannel ),
            torch.nn.ReLU()
        )
    def forward(self,x ):
        # print('ResDenseBlock x',x.shape)
        y = self.conv_head(x)
        y_0 = self.conv_line_0(y)
        y_1 = self.conv_line_1(y)
        y = torch.cat([y_0,y_1],1)

        return self.conv_last(y) + x

class ConvBarnch(torch.nn.Module):
    def __init__(self,inChannel=128,outChannel=512,dilation=1,BatchNorm = torch.nn.BatchNorm2d):
        super(ConvBarnch,self).__init__()
        self.conv_l0 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=inChannel, out_channels=inChannel , kernel_size=3, stride=1,
                            padding=(3 - 1) // 2*dilation, dilation=dilation),
            BatchNorm(inChannel),
            # torch.nn.Conv2d(in_channels=inChannel, out_channels=inChannel, kernel_size=3, stride=1,
            #                 padding=(3 - 1) // 2*dilation, dilation=dilation),
            # BatchNorm(inChannel),
            torch.nn.ReLU(),

            ResDenseBlock(inChannel=inChannel,outChannel=inChannel,BatchNorm = BatchNorm),
            # ResDenseBlock(inChannel=inChannel, outChannel=inChannel, BatchNorm=BatchNorm)
        )
        self.conv_l1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=inChannel , out_channels=inChannel*2, kernel_size=3, stride=1,
                            padding=(3 - 1) // 2*dilation, dilation=dilation),
            BatchNorm(inChannel*2),
            # torch.nn.Conv2d(in_channels=inChannel*2, out_channels=inChannel*2, kernel_size=3, stride=1,
            #                 padding=(3 - 1) // 2*dilation, dilation=dilation),
            # BatchNorm(inChannel*2),
            torch.nn.ReLU(),

            ResDenseBlock(inChannel=inChannel*2, outChannel=inChannel*2, BatchNorm=BatchNorm),
            # ResDenseBlock(inChannel=inChannel*2, outChannel=inChannel*2, BatchNorm=BatchNorm)
        )
        self.conv_l2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=inChannel*2, out_channels=inChannel * 3, kernel_size=3, stride=1,
                            padding=(3 - 1) // 2*dilation, dilation=dilation),
            BatchNorm(inChannel * 3),

            ResDenseBlock(inChannel=inChannel * 3, outChannel=inChannel * 3, BatchNorm=BatchNorm),
            # ResDenseBlock(inChannel=inChannel * 3, outChannel=inChannel * 3, BatchNorm=BatchNorm)
        )
    def forward(self,x ):

        lx_0 = self.conv_l0(x)

        lx_1 = self.conv_l1(torch.nn.functional.interpolate(lx_0,size = [lx_0.size()[-2]//2,lx_0.size()[-1]//2],mode = 'bilinear',align_corners=True))
        lx_2 = self.conv_l2(torch.nn.functional.interpolate(lx_1,size = [lx_1.size()[-2]//2,lx_1.size()[-1]//2],mode = 'bilinear',align_corners=True))


        return lx_0,lx_1,lx_2

class FFM(torch.nn.Module):
    def __init__(self,c0,c1,c2=None,BatchNorm = torch.nn.BatchNorm2d):
        super(FFM, self).__init__()
        self.avgPooling = torch.nn.AdaptiveAvgPool2d(1)
        self.reWeight = torch.nn.Sequential(  torch.nn.Conv2d(c0+c1, min(c0,c1)*2, kernel_size=1, bias=False),
                                        BatchNorm(min(c0,c1)*2),
                                        torch.nn.ReLU(inplace=False),
                                        torch.nn.Conv2d(min(c0, c1) * 2, c0+c1, kernel_size=1, bias=False),
                                        BatchNorm(c0+c1),
                                        torch.nn.Sigmoid())
        self.lastLayer = False if c2 is None else True
        if c2 is not None:
            self.lastConv = torch.nn.Conv2d(c0+c1,c2,kernel_size=1,bias = True)
    def forward(self,x,y):
        m = torch.cat([x,y],1)
        m_flat = self.avgPooling(m)
        # print('m_flat',m_flat.shape)
        m_weight = self.reWeight(m_flat)
        # print('m_weight',m_weight.max(),m_weight.min())
        if self.lastLayer:
            return self.lastConv(m*m_weight)
        else:
            return m*m_weight
class JPU(torch.nn.Module):
    def __init__(self,inChannels=[256,512,1024,2048],conChannels = [128,128,256,512],\
                 dilations = [1,2,4,8]):
        super(JPU, self).__init__()
        self.convList = []
        convcum = 0

        for idx,c in enumerate(inChannels):
            convcum += conChannels[idx]
            self.convList.append(nn.Sequential(
                nn.Conv2d(c, conChannels[idx], kernel_size=1, bias=False),
                # BatchNorm(conChannels[idx]),
                nn.ReLU(inplace=False)
            ))
        self.convList = nn.ModuleList(self.convList)

        self.dConvList = []
        for idx,c in enumerate(dilations):
            self.dConvList.append(nn.Sequential( # padding = (kernel_size - 1)//stride*dilation
                nn.Conv2d(convcum, convcum//len(dilations), kernel_size=3,padding=dilations[idx], bias=False,dilation = dilations[idx]),
                # BatchNorm(convcum//len(dilations)),
                nn.ReLU(inplace=False)
            ))
        self.dConvList = nn.ModuleList(self.dConvList)

        inDim = (convcum//len(dilations))*len(dilations)
        outDim = convcum


        self.lastConv = nn.Conv2d(inDim, outDim, kernel_size=1, bias=True)
    def forward(self,insList):
        size = insList[0].size()
        out = []
        for idx,f in enumerate(self.convList):
            
            y = f(insList[idx])
            y = torch.nn.functional.interpolate(y,size = [size[-2],size[-1]],mode = 'bilinear',align_corners=True)

            out.append(y)
        feat = torch.cat(out,1)

        out = []
        for idx,f in enumerate(self.dConvList):
            y = f(feat)
            # print('dconv',feat.shape,f)
            out.append(y)
        feat = torch.cat(out, 1)
        # print('feat 1',feat.shape)
        return self.lastConv(feat)

if __name__ == '__main__':
    torch.cuda.set_device(0)
    device = torch.device('cuda')
    # net = DeBlurNet(config)
    nf = 32
    net = JPU(inChannels=[nf,nf,nf],conChannels = [nf,nf,nf],dilations = [1,2,4])
    net.to(device)
    print("model have {:.3f}M paramerters in total".format(sum(x.numel() for x in net.parameters())/1000000.0)) 

    ins = [torch.randn(2,32,64,64).to(device),torch.randn(2,32,32,32).to(device),torch.randn(2,32,16,16).to(device)]
    out = net(ins)
    print(out.shape)