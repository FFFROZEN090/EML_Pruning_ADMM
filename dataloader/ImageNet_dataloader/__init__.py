""" 
This pacakge is used to load the ImageNet dataset.
The ImageNet dataset is a large dataset that is used for image classification.
This dataloader are mainly designed to load the ImageNet dataset in the PyTorch framework of AlexNet.
"""

from .ImageNet_dataloader import ImageNetDataset
from .ImageNet_dataloader import ImageNetDataLoader

__all__ = ['ImageNetDataset', 'ImageNetDataLoader']