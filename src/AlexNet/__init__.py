"""" 
AlexNet is a convolutional neural network that is used for image classification.
In this package, we have implemented the AlexNet architecture using PyTorch and specifically for the downsampled ImageNet64x64 dataset.
The architecture is implemented in the AlexNet.py file and the training and testing scripts are implemented in the train.py and test.py files respectively.
"""

from .AlexNet import AlexNet
from .train import train
from .test import test

__all__ = ['AlexNet', 'train', 'test']