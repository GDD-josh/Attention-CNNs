import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from torchvision.transforms import ToPILImage
from IPython.display import Image
to_img = ToPILImage()

# shows a CIFAR image before and after processing  

idx = 100
c100train =  torchvision.datasets.CIFAR10('cifar-10', train=True, transform=transforms.ToTensor(), download=True)
traindata = c100train.train_data

img = traindata[idx]
out = to_img(img)
out.show()

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5423671, 0.53410053, 0.5282784), (0.3012955, 0.29579896, 0.2906593))])

c100train =  torchvision.datasets.CIFAR10('cifar-10', train=True, transform=transform, download=True)
traindata = c100train.train_data

img = traindata[idx]
out = to_img(img)
out.show()
