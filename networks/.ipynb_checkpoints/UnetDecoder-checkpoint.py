import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm

class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
        # self.bn1 = nn.GroupNorm(4, middle_channels, affine=True)
        self.bn1 = nn.BatchNorm2d(out_channels, affine=True)
        self.conv2 = spectral_norm(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
        # self.bn2 = nn.GroupNorm(4, out_channels, affine=True)
        self.bn2 = nn.BatchNorm2d(out_channels, affine=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = nn.Dropout2d(0.1)(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out

class UnetDecoder(nn.Module):
    def __init__(self, encoder_hidden_sizes, n_classes=12):
        super().__init__()
        self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.hidden_sizes = [0] + encoder_hidden_sizes[::-1]
        self.blocks = nn.ModuleList([VGGBlock(self.hidden_sizes[i]+self.hidden_sizes[i+1], self.hidden_sizes[i+1]) for i in range(len(self.hidden_sizes)-1)])
        self.classifier = spectral_norm(nn.Conv2d(self.hidden_sizes[-1], n_classes, kernel_size=1, stride=1, padding=0))
        self._xavier_init()
    def _xavier_init(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight) 
                if module.bias is not None:
                    module.bias.data.zero_()       
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

    def _kaiming_init(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()
                
    def forward(self, encoder_features):
        encoder_features = encoder_features[::-1]
        x = self.blocks[0](encoder_features[0])
        for i in range(1, len(self.blocks)):
            x = self.up_sample(x)
            if (x.shape[-2], x.shape[-1]) != (encoder_features[i].shape[-2], encoder_features[i].shape[-1]):
                x = F.interpolate(x, (encoder_features[i].shape[-2], encoder_features[i].shape[-1]), mode='bilinear')
            # print(torch.cat([encoder_features[i], x], dim=1).shape)
            x = self.blocks[i](torch.cat([encoder_features[i], x], dim=1))
        x = self.classifier(x)
        return x