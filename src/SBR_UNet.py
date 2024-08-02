import torch
import torch.nn as nn
import torch.nn.init as init
import math

"""
Building Network architecture based on the paper:
Robust single-shot 3D fluorescence imaging in scattering media
with a simulator-trained neural network
(https://doi.org/10.1364/OE.514072)

The network has two brances. One branch takes in RFV input (refocused volume)
and the other takes in stack (light-field measurements). 

Modification to the SBR-Net architecture:
Instead of having each branch as Residual Blocks, we are using U-Net architecture for each of the branches.
"""
##########
# Setting up Kaiming He initialization that will be used once for all
# Conv2d layers. Biases must be set to zero.
# This ensures that all layers remain close to 0 and 1.
##########
def kaiming_he_init(layer):
    if isinstance(layer, nn.Conv2d):
        init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')

        if layer.bias is not None:
            init.zeros_(layer.bias)


##########
# U-Net Architecture class, referenced from:
# https://towardsdatascience.com/cook-your-first-u-net-in-pytorch-b3297a844cf3
# padding = 1 s.t. output is not cropped
##########
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        # Encoder section of the U-Net (downsampling)
        self.encoder1 = self.double_conv(in_channels, 64)
        self.encoder2 = self.double_conv(64, 128)
        self.encoder3 = self.double_conv(128, 256)
        self.encoder4 = self.double_conv(256, 512)

        self.center = self.double_conv(512, 1024)

        # Decoder section of the U-Net (upsampling)
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder1 = self.double_conv(1024, 512)
        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder2 = self.double_conv(512, 256)
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder3 = self.double_conv(256, 128)
        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder4 = self.double_conv(128, 64)

        # Output layer
        self.outconv = nn.Conv2d(64, out_channels, kernel_size=1)

    def double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(nn.MaxPool2d(kernel_size=2)(e1))
        e3 = self.encoder3(nn.MaxPool2d(kernel_size=2)(e2))
        e4 = self.encoder4(nn.MaxPool2d(kernel_size=2)(e3))

        center = self.center(nn.MaxPool2d(kernel_size=2)(e4))

        # Decoder
        d1 = self.upconv1(center)
        d1 = torch.cat([d1, e4], dim=1)
        d1 = self.decoder1(d1)

        d2 = self.upconv2(d1)
        d2 = torch.cat([d2, e3], dim=1)
        d2 = self.decoder2(d2)

        d3 = self.upconv3(d2)
        d3 = torch.cat([d3, e2], dim=1)
        d3 = self.decoder3(d3)

        d4 = self.upconv4(d3)
        d4 = torch.cat([d4, e1], dim=1)
        d4 = self.decoder4(d4)

        out = self.outconv(d4)

        return out
    

##########
# Building SBR network
##########
class SBR_Net(nn.Module):
    def __init__(self, in_channels_rfv, in_channels_stack, kernel_size=3, stride=1):
        super(SBR_Net, self).__init__()
        self.rfv_conv = nn.Conv2d(in_channels=in_channels_rfv, out_channels=48, 
                      kernel_size=kernel_size, stride=stride, padding=1)
        self.stack_conv = nn.Conv2d(in_channels=in_channels_stack, out_channels=48, 
                      kernel_size=kernel_size, stride=stride, padding=1)

        self.rfv_unet = UNet(in_channels_rfv, 24)
        self.stack_unet = UNet(in_channels_stack, 24)
        # self.final_conv = nn.Conv2d(48, 24, kernel_size=1)

        self.conv_48_to_48 = nn.Conv2d(in_channels=48, out_channels=48, 
                      kernel_size=kernel_size, stride=stride, padding=1)
        
        self.conv_48_to_24 = nn.Conv2d(in_channels=48, out_channels=24,
                      kernel_size=kernel_size, stride=stride, padding=1)
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, rfv_input, stack_input):
        # Branch 1 (RFV)
        out_rfv = self.rfv_conv(rfv_input)
        residual_rfv = out_rfv
        out_rfv = self.rfv_unet(out_rfv)
        
        out_rfv += residual_rfv / math.sqrt(2)
        out_rfv = self.conv_48_to_48(out_rfv)

        # Branch 2 (Stack)
        out_stack = self.stack_conv(stack_input)
        residual_stack = out_stack
        out_stack = self.stack_unet(out_stack)

        out_stack += residual_stack / math.sqrt(2)
        out_stack = self.conv_48_to_48(out_stack)

        # Fusion point
        out = (out_rfv + out_stack) / math.sqrt(2)

        out = self.conv_48_to_24(out)
        out = self.sigmoid(out)

        return out # Shape of output should be: Bx24x224x224