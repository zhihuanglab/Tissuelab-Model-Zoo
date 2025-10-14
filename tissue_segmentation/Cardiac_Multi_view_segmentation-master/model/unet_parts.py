#!/usr/bin/python

# sub-parts of the U-Net model

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from model.init_weight import init_weights
class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch,norm=nn.BatchNorm2d,if_SN=False, activation=nn.ReLU):
        super(double_conv, self).__init__()
        if not if_SN:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                norm(out_ch),
                activation(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                norm(out_ch),
                activation(inplace=True)
                )
        else:
            self.conv = nn.Sequential(
                spectral_norm(nn.Conv2d(in_ch, out_ch, 3, padding=1)),
                norm(out_ch),
                activation(inplace=True),
                spectral_norm(nn.Conv2d(out_ch, out_ch, 3, padding=1)),
                norm(out_ch),
                activation(inplace=True)
            )

    def forward(self, x):
        x = self.conv(x)
        return x
class conv2DBatchNorm(nn.Module):
    def __init__(self, in_channels, n_filters, k_size,  stride, padding, bias=True):
        super(conv2DBatchNorm, self).__init__()

        self.cb_unit = nn.Sequential(nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                               padding=padding, stride=stride, bias=bias),
                                 nn.BatchNorm2d(int(n_filters)),)

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        outputs = self.cb_unit(inputs)
        return outputs

class conv2DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, n_filters, k_size,  stride, padding, bias=True):
        super(conv2DBatchNormRelu, self).__init__()

        self.cbr_unit = nn.Sequential(nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                                padding=padding, stride=stride, bias=bias),
                                 nn.BatchNorm2d(int(n_filters)),
                                 nn.ReLU(inplace=True),)

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs
class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, n=2, ks=3, stride=1, padding=1):
        super(unetConv2, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        if is_batchnorm:
            for i in range(1, n+1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.BatchNorm2d(out_size),
                                     nn.ReLU(inplace=True),)
                setattr(self, 'conv%d'%i, conv)
                in_size = out_size

        else:
            for i in range(1, n+1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.ReLU(inplace=True),)
                setattr(self, 'conv%d'%i, conv)
                in_size = out_size

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n+1):
            conv = getattr(self, 'conv%d'%i)
            x = conv(x)

        return x
class inconv(nn.Module):
    def __init__(self, in_ch, out_ch,norm=nn.BatchNorm2d,if_SN=False, activation=nn.ReLU):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch,norm=norm,if_SN=if_SN,activation=activation)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch,norm=nn.BatchNorm2d,if_SN=False,activation=nn.ReLU):
        super(down, self).__init__()

        self.mpconv = nn.Sequential(
        nn.MaxPool2d(2),
        double_conv(in_ch, out_ch,norm=norm,if_SN=if_SN,activation=activation)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class CSELayer(nn.Module):
    def __init__(self, channel):
        super(CSELayer, self).__init__()
        self.spatial_conv = nn.Sequential(nn.Conv2d(channel,1,1),
                                          nn.Sigmoid()
        )


    def forward(self, x):
        b, c,h,w = x.size()
        y = self.spatial_conv(x)
        return x * y


class up(nn.Module):
    def __init__(self, in_ch_1,in_ch_2, out_ch, type='bilinear',dropout=None,norm=nn.BatchNorm2d,if_SN=False, activation=nn.ReLU):
        super(up, self).__init__()
        self.type=type
        if type=='bilinear':
            self.up = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
        elif type=='deconv':
            self.up = nn.ConvTranspose2d((in_ch_1+in_ch_2)//2, (in_ch_1+in_ch_2)//2, 2, stride=2)
        elif type == 'nearest':
            self.up = nn.Upsample(scale_factor=2)
        else:
            self.up=None
        if type=='bilinear_additive':
            self.conv = double_conv (in_ch_1//2+in_ch_2,out_ch,norm=norm,if_SN=if_SN,activation=activation)
        else:
            self.conv = double_conv(in_ch_1+in_ch_2, out_ch,norm=norm,if_SN=if_SN,activation=activation)
        self.dropout=dropout
        if self.dropout:
            self.drop=nn.Dropout2d(p=dropout)
            print ('enable dropout')

    def forward(self, x1, x2):

        if self.type=='bilinear_additive':
            from model.custom_layers import bilinear_additive_upsampling
            x1=bilinear_additive_upsampling(x1,x1.size(1)//2)
        else:
            x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        if self.dropout and self.training:
            x=self.drop(x)
        x = self.conv(x)
        return x

class sqe_up(nn.Module):
    def __init__(self, in_ch_1,in_ch_2, out_ch, type='bilinear'):
        super(sqe_up, self).__init__()
        self.type=type
        if type=='bilinear':
            self.up = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
        elif type=='deconv':
            self.up = nn.ConvTranspose2d((in_ch_1+in_ch_2)//2, (in_ch_1+in_ch_2)//2, 2, stride=2)
        elif type == 'nearest':
            self.up = nn.Upsample(scale_factor=2)
        else:
            self.up=None

        if type=='bilinear_additive':
            self.conv = double_conv (in_ch_1//2+in_ch_2,out_ch)
        else:
            self.conv = double_conv(in_ch_1+in_ch_2, out_ch)

        self.sqe=SELayer(in_ch_1+in_ch_2)
        self.cqe=CSELayer(out_ch)

    def forward(self, x1, x2):

        if self.type=='bilinear_additive':
            from model.custom_layers import bilinear_additive_upsampling
            x1=bilinear_additive_upsampling(x1,x1.size(1)//2)
        else:
            x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        out=self.sqe(x)
        feature= self.conv(out)
        out = feature+self.cqe(feature)
        return out


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x
class outconv_relu(nn.Module):
    def __init__(self, in_ch, out_ch,activation=nn.ReLU):
        super(outconv_relu, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)
        if activation is None:
            self.relu=None
        elif isinstance(activation,nn.Linear):
            self.relu= activation()
        else:
            self.relu=activation(inplace=True)

    def forward(self, x):
        if self.relu is None:
            x =self.conv(x)
        else:
            x = self.relu(self.conv(x))
        return x

class UnetUp3(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, is_batchnorm=True,z_scale_factor=1):
        super(UnetUp3, self).__init__()
        if is_deconv:
            self.conv = UnetConv3(in_size, out_size, is_batchnorm)
            if z_scale_factor==1:
                self.up = nn.ConvTranspose3d(in_size, out_size, kernel_size=(4,4,z_scale_factor), stride=(2,2,z_scale_factor), padding=(1,1,0))
            else:
                self.up = nn.ConvTranspose3d(in_size, out_size, kernel_size=(4,4,4), stride=(2,2,2),padding=(1,1,1))

        else:
            self.conv = UnetConv3(in_size+out_size, out_size, is_batchnorm)
            self.up = nn.Upsample(scale_factor=(2, 2, z_scale_factor), mode='trilinear')

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('UnetConv3') != -1: continue
            init_weights(m, init_type='kaiming')
        self.z_scale_factor=z_scale_factor

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset = outputs2.size()[2] - inputs1.size()[2]
        offset_z= outputs2.size()[4] - inputs1.size()[4]

        padding = 2 * [offset // 2, offset // 2, offset_z//2]

        outputs1 = F.pad(inputs1, padding)
        return self.conv(torch.cat([outputs1, outputs2], 1))


class UnetConv3(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, kernel_size=(3,3,1), padding_size=(1,1,0), init_stride=(1,1,1)):
        super(UnetConv3, self).__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, init_stride, padding_size),
                                       nn.BatchNorm3d(out_size),
                                       nn.ReLU(inplace=True),)
            self.conv2 = nn.Sequential(nn.Conv3d(out_size, out_size, kernel_size, 1, padding_size),
                                       nn.BatchNorm3d(out_size),
                                       nn.ReLU(inplace=True),)
        else:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, init_stride, padding_size),
                                       nn.ReLU(inplace=True),)
            self.conv2 = nn.Sequential(nn.Conv3d(out_size, out_size, kernel_size, 1, padding_size),
                                       nn.ReLU(inplace=True),)

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs

