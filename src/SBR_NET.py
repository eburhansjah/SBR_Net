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
Each branch has 20 Residual Blocks: 
3x3 conv. Layer -> batch norm. -> ReLU ->  3x3 conv. layer -> batch norm.

The input of each branch of the ResBlock is added to the output of the same
ResBlock.
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
# Building Residual Block to be later used to build Res Branch
##########
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                      kernel_size=kernel_size, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, 
                      kernel_size=kernel_size, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels)
        )

        # For checking skip connection with projection
        if in_channels != out_channels or stride != 1:
            self.skip_connection = nn.Conv2d(in_channels=in_channels, 
                                             out_channels=out_channels, 
                                             kernel_size=kernel_size, 
                                             stride=stride, padding=1)
        else:
            self.skip_connection = nn.Identity()
    
    def forward(self, x):
        residual = self.skip_connection(x)
        out = self.conv1(x)
        out = self.conv2(out)

        out += residual / math.sqrt(2)

        return out


##########
# Building Res Branch from Res Blocks
##########
class ResBranch(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks):
        super(ResBranch, self).__init__()
        self.layers = self.make_layers(in_channels, out_channels, num_blocks)
    
    def make_layers(self, in_channels, out_channels, num_blocks):
        layers = []
        for block in range(num_blocks):
            layers.append(ResBlock(in_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)

##########
# Building SBR network
# in_channels_rfv = 24; in_channels_stack = 9
##########
class SBR_Net(nn.Module):
    def __init__(self, in_channels_rfv, in_channels_stack, num_blocks, kernel_size=3, stride=1):
        super(SBR_Net, self).__init__()
        self.rfv_conv = nn.Conv2d(in_channels=in_channels_rfv, out_channels=48, 
                      kernel_size=kernel_size, stride=stride, padding=1)
        self.stack_conv = nn.Conv2d(in_channels=in_channels_stack, out_channels=48, 
                      kernel_size=kernel_size, stride=stride, padding=1)
        
        self.rfv_res_branch = ResBranch(in_channels=48, out_channels=48, num_blocks=num_blocks)
        self.stack_res_branch = ResBranch(in_channels=48, out_channels=48, num_blocks=num_blocks)

        self.conv_48_to_48 = nn.Conv2d(in_channels=48, out_channels=48, 
                      kernel_size=kernel_size, stride=stride, padding=1)
        
        self.conv_48_to_24 = nn.Conv2d(in_channels=48, out_channels=24,
                      kernel_size=kernel_size, stride=stride, padding=1)
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, rfv_input, stack_input):
        # Branch 1 (RFV)
        out_rfv = self.rfv_conv(rfv_input)
        residual_rfv = out_rfv
        out_rfv = self.rfv_res_branch(out_rfv)
        
        out_rfv += residual_rfv / math.sqrt(2)
        out_rfv = self.conv_48_to_48(out_rfv)
        
        # Branch 2 (Stack)
        out_stack = self.stack_conv(stack_input)
        residual_stack = out_stack
        out_stack = self.stack_res_branch(out_stack)

        out_stack += residual_stack / math.sqrt(2)
        out_stack = self.conv_48_to_48(out_stack)

        # Fusion point
        out = (out_rfv + out_stack) / math.sqrt(2)
        
        out = self.conv_48_to_24(out)
        out = self.sigmoid(out) 

        return out # Shape of output should be: Bx24x224x224