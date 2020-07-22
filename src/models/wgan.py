"""
    @author Tuan Dinh tuandinh@cs.wisc.edu
    @date 08/14/2019
    Loading data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
    
    def forward(self, x):
        return x + torch.sigmoid(x)

def block(in_feat, out_feat, normalize=True, dropout=0.):
    layers = [nn.Linear(in_feat, out_feat, bias=not normalize)]
    if normalize:
        layers.append(nn.BatchNorm1d(out_feat))
    layers.append(nn.LeakyReLU())
    if dropout > 0.:
        layers.append(nn.Dropout(dropout))
    return layers
"""
"""
class Generator(nn.Module):
    def __init__(self, input_size=100, output_size=4225):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            *block(input_size, 128),
            *block(128, 256),
            *block(256, 1024),
            nn.Linear(1024, output_size),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

"""
"""
class Discriminator(nn.Module):
    def __init__(self, input_size=4225):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            *block(input_size, 1024, normalize=False, dropout=0.1),
            *block(1024, 256, normalize=False, dropout=0.1),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.model(x)
