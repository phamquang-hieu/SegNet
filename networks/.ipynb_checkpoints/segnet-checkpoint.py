import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from copy import deepcopy
from torch.nn.utils import spectral_norm
from torchvision.models import VGG16_BN_Weights 

class SegNet(nn.Module):
    def __init__(self, num_classes, in_channels=3, pretrained=True):
        super(SegNet, self).__init__()
        vgg_bn = models.vgg16_bn(weights=VGG16_BN_Weights.DEFAULT)
        encoder = list(vgg_bn.features.children())
        '''
        # VGG16
        [0] Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        [1] BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        [2] ReLU(inplace=True)
        [3] Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        [4] BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        [5] ReLU(inplace=True)
        [6] MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        [7] Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        [8] BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        [9] ReLU(inplace=True)
        [10] Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        [11] BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        [12] ReLU(inplace=True)
        [13] MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        [14] Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        [15] BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        [16] ReLU(inplace=True)
        [17] Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        [18] BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        [19] ReLU(inplace=True)
        [20] Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        [21] BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        [22] ReLU(inplace=True)
        [23] MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        [24] Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        [25] BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        [26] ReLU(inplace=True)
        [27] Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        [28] BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        [29] ReLU(inplace=True)
        [30] Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        [31] BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        [32] ReLU(inplace=True)
        [33] MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        [34] Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        [35] BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        [36] ReLU(inplace=True)
        [37] Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        [38] BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        [39] ReLU(inplace=True)
        [40] Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        [41] BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        [42] ReLU(inplace=True)
        [43] MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        '''
        # adapt the first layer to input channel
        if in_channels != 3:
            encoder[0] = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, stride=2, return_indices=True)
        
        self.encoder_stages = []
        i = 0
        current_stage = []
        while i < len(encoder):
            if isinstance(encoder[i], nn.Conv2d):
                current_stage.append(spectral_norm(encoder[i]))
            else:
                current_stage.append(encoder[i])
            i+=1
            if isinstance(encoder[i], nn.MaxPool2d):
                i += 1
                self.encoder_stages.append(nn.Sequential(*current_stage))
                current_stage = []
        
        self.decoder_stages = []
        i = 0
        current_stage = []
        while i < len(encoder):  
            if isinstance(encoder[i], nn.Conv2d) and encoder[i].in_channels < encoder[i].out_channels:
                if i == 0:
                    decoder_conv = spectral_norm(nn.Conv2d(encoder[i].out_channels, num_classes, kernel_size=encoder[i].kernel_size, padding=encoder[i].padding))
                    decoder_bn = nn.BatchNorm2d(num_classes)
                else:    
                    decoder_conv = spectral_norm(nn.Conv2d(encoder[i].out_channels, encoder[i].in_channels, kernel_size=encoder[i].kernel_size, padding=encoder[i].padding))
                    decoder_bn = nn.BatchNorm2d(encoder[i].in_channels)
            else: 
                decoder_conv = deepcopy(encoder[i])
                decoder_bn = deepcopy(encoder[i+1])
                
            current_stage.append(nn.Sequential(*[decoder_conv, decoder_bn, deepcopy(encoder[i+2])]))
            i+=3
            if isinstance(encoder[i], nn.MaxPool2d):
                i += 1
                current_stage.reverse()
                self.decoder_stages.append(nn.Sequential(*current_stage))
                current_stage = []
        self.decoder_stages.reverse()
        self._weight_init(self.decoder_stages)
        
        # This is for registering encoder's and decoder's stages with torch
        self.encoder_stages = nn.ModuleList(self.encoder_stages)
        self.decoder_stages = nn.ModuleList(self.decoder_stages)
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        indices = []
        sizes = []
    
        for stage in self.encoder_stages:
            x = stage(x)
            sizes.append(x.size())
            x, index = self.pool(x)
            indices.append(index)
        for stage in self.decoder_stages:
            x = self.unpool(x, indices=indices.pop(), output_size=sizes.pop())
            x = stage(x)
        return x
    
    def modules(self):
        res = []
        for stage in self.encoder_stages:
            res.append(stage)
        for stage in self.decoder_stages:
            res.append(stage)
        res.append(self.pool)
        res.append(self.unpool)
        return res
    
    def _weight_init(self, stages):
        for modules in stages:
            for module in modules.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight)
                    if module.bias is not None:
                        module.bias.data.zero_()
                elif isinstance(module, nn.BatchNorm2d):
                    module.weight.data.fill_(1)
                    module.bias.data.zero_()
                    
