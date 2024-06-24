import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torchvision import transforms
from matplotlib import pyplot as plt
from datasets import load_dataset
from PIL import Image

class ImageNetDataset(IterableDataset):
    def __init__(self, split='train', img_size=224, transform=None):
        self.dataset = load_dataset('Maysee/tiny-imagenet', split=split, streaming=True)
        self.img_size = img_size
        self.transform = transform

    def preprocess_image(self, image):
        # Convert all images to RGB mode
        if image.mode != 'RGB':
            image = image.convert('RGB')
        # Ensure image is resized to the desired shape
        image = image.resize((self.img_size, self.img_size))
        return image

    def __iter__(self):
        for item in self.dataset:
            image = self.preprocess_image(item['image'])
            label = item['label']
            if self.transform:
                image = self.transform(image)
            yield image, label

class ImageNetDataLoader:
    def __init__(self, batch_size=80, img_size=224, use_cuda=False):
        self.batch_size = batch_size
        self.img_size = img_size
        self.use_cuda = use_cuda
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def create_dataloader(self):
        dataset = ImageNetDataset(split='train', img_size=self.img_size, transform=self.transform)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True)
        return dataloader

    def get_dataloader(self):
        dataloader = self.create_dataloader()
        return dataloader

def visualize_imagenet_batch(dataloader, num_samples=5):
    dataiter = iter(dataloader)
    images, labels = next(dataiter)

    images = images.numpy()
    images = images.transpose(0, 2, 3, 1)
    images = (images * 0.5) + 0.5

    for i in range(num_samples):
        plt.figure()
        plt.imshow(images[i])
        plt.title(f'Label: {labels[i].item()}')
        plt.axis('off')
        plt.show()
        # Save images if needed
        plt.savefig(f'image_{i}.png')

if __name__ == '__main__':
    batch_size = 64
    img_size = 224
    use_cuda = torch.cuda.is_available()

    dataloader_class = ImageNetDataLoader(batch_size, img_size, use_cuda)
    dataloader = dataloader_class.get_dataloader()

    visualize_imagenet_batch(dataloader, num_samples=5)
