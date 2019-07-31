import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_, constant_
#from .util import conv, predict_flow, deconv, crop_like#custom component

__all__ = [
    'flownets', 'flownets_bn'
]


class FlowNetS(nn.Module):
    expansion = 1

    def __init__(self,batchNorm=True):
        super(FlowNetS,self).__init__()

        self.batchNorm = batchNorm
        self.conv1   = conv(self.batchNorm,   6,   64, kernel_size=7, stride=2)
        self.conv2   = conv(self.batchNorm,  64,  128, kernel_size=5, stride=2)

        self.conv3   = conv(self.batchNorm, 128,  256, kernel_size=5, stride=2)
        self.conv3_1 = conv(self.batchNorm, 256,  256)

        self.conv4   = conv(self.batchNorm, 256,  512, stride=2)
        self.conv4_1 = conv(self.batchNorm, 512,  512)

        self.conv5   = conv(self.batchNorm, 512,  512, stride=2)
        self.conv5_1 = conv(self.batchNorm, 512,  512)

        self.conv6   = conv(self.batchNorm, 512, 1024, stride=2)
        self.conv6_1 = conv(self.batchNorm,1024, 1024)

        self.deconv5 = deconv(1024,512)
        self.deconv4 = deconv(1026,256)
        self.deconv3 = deconv(770,128)
        self.deconv2 = deconv(386,64)

        self.predict_flow6 = predict_flow(1024)
        self.predict_flow5 = predict_flow(1026)
        self.predict_flow4 = predict_flow(770)
        self.predict_flow3 = predict_flow(386)
        self.predict_flow2 = predict_flow(194)

        self.upsampled_flow6_to_5 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow5_to_4 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)

        ##init weights
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)

    def forward(self, x):
        #out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(self.conv1(x))#out_conv2[1,128,96,128]
        out_conv3 = self.conv3_1(self.conv3(out_conv2))#out_conv3[1,256,48,64]
        out_conv4 = self.conv4_1(self.conv4(out_conv3))#out_conv4[1,512,24,32]
        out_conv5 = self.conv5_1(self.conv5(out_conv4))#out_conv5[1,512,12,16]
        out_conv6 = self.conv6_1(self.conv6(out_conv5))#out_conv6[1,1024,6,8]

        #refinement
        flow6       = self.predict_flow6(out_conv6)#flow6 [1,2,6,8]
        flow6_up    = crop_like(self.upsampled_flow6_to_5(flow6), out_conv5)#[1,2,48,64]
        out_deconv5 = crop_like(self.deconv5(out_conv6), out_conv5)#[1,512,12,16]

        concat5 = torch.cat((out_conv5,out_deconv5,flow6_up),1)#torch.Size([1, 1026, 12, 16])
        flow5       = self.predict_flow5(concat5)
        flow5_up    = crop_like(self.upsampled_flow5_to_4(flow5), out_conv4)
        out_deconv4 = crop_like(self.deconv4(concat5), out_conv4)

        concat4 = torch.cat((out_conv4,out_deconv4,flow5_up),1)#torch.Size([1, 770, 24, 32])
        flow4       = self.predict_flow4(concat4)#torch.Size([1, 2, 24, 32])
        flow4_up    = crop_like(self.upsampled_flow4_to_3(flow4), out_conv3)#torch.Size([1, 2, 48, 64])
        out_deconv3 = crop_like(self.deconv3(concat4), out_conv3)#torch.Size([1, 128, 48, 64])

                            #256+128+2 = 386 按axis = 1 cat
        concat3 = torch.cat((out_conv3,out_deconv3,flow4_up),1)#torch.Size([1, 386, 48, 64])
        flow3       = self.predict_flow3(concat3)#torch.Size([1, 2, 48, 64])
        flow3_up    = crop_like(self.upsampled_flow3_to_2(flow3), out_conv2)#torch.Size([1, 2, 96, 128])
        out_deconv2 = crop_like(self.deconv2(concat3), out_conv2)#torch.Size([1, 64, 96, 128])

        concat2 = torch.cat((out_conv2,out_deconv2,flow3_up),1)#torch.Size([1, 194, 96, 128])
        flow2 = self.predict_flow2(concat2)#torch.Size([1, 2, 96, 128])

        if self.training:
            return flow2,flow3,flow4,flow5,flow6
        else:
            return flow2#flow2 结果是1,2,96,128 ，论文上是136x320

    # public func, for specific using
    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]




## utils

def crop_like(input, target):
    if input.size()[2:] == target.size()[2:]:
        return input
    else:
        return input[:, :, :target.size(2), :target.size(3)]
def predict_flow(in_planes):
    return nn.Conv2d(in_channels=in_planes,out_channels=2,kernel_size=3,stride=1,padding=1,bias=False)


def deconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=False),
        nn.LeakyReLU(0.1,inplace=True)
    )
def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1,inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.LeakyReLU(0.1,inplace=True)
        )




#for test?

def flownets(data=None):
    """FlowNetS model architecture from the
    "Learning Optical Flow with Convolutional Networks" paper (https://arxiv.org/abs/1504.06852)

    Args:
        data : pretrained weights of the network. will create a new one if not set
    """
    model = FlowNetS(batchNorm=False)
    if data is not None:
        model.load_state_dict(data['state_dict'])
    return model


def flownets_bn(data=None):
    """FlowNetS model architecture from the
    "Learning Optical Flow with Convolutional Networks" paper (https://arxiv.org/abs/1504.06852)

    Args:
        data : pretrained weights of the network. will create a new one if not set
    """
    model = FlowNetS(batchNorm=True)
    if data is not None:
        model.load_state_dict(data['state_dict'])
    return model

