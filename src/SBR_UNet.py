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
Instead of having each branch as Residual Blocks, we are using U-Net architecture
"""