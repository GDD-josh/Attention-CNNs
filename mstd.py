#!/usr/bin/env python3

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms

# Calculates the mean and standard deviations of the CIFAR dataset

data = dset.CIFAR100(root='cifar-100', train=True, download=False,
                    transform=transforms.ToTensor()).train_data
data = data.astype(np.float32)/255.

means = []
stdevs = []
for i in range(3):
    pixels = data[:,i,:,:].ravel()
    means.append(np.mean(pixels))
    stdevs.append(np.std(pixels))

print("means: {}".format(means))
print("stdevs: {}".format(stdevs))
print('transforms.Normalize(mean = {}, std = {})'.format(means, stdevs))
