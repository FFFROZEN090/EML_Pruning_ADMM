import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from matplotlib import pyplot as plt
from datasets import load_dataset
from PIL import Image

class ImageNetDataset(Dataset):
    def __init__(self, split='train', img_size=64, transform=None):
        # Loading dataset without streaming
        self.dataset = load_dataset('Maysee/tiny-imagenet', split=split)  # Removed streaming=True
        self.img_size = img_size
        self.transform = transform

    def preprocess_image(self, image):
        # Convert all images to RGB mode
        if image.mode != 'RGB':
            image = image.convert('RGB')
        # Ensure image is resized to the desired shape
        image = image.resize((self.img_size, self.img_size))
        return image

    def __len__(self):
        return len(self.dataset)  # Now this works because the dataset is fully loaded

    def __getitem__(self, idx):
        # Accessing data via indexing
        item = self.dataset[idx]
        image = self.preprocess_image(item['image'])
        if self.transform:
            image = self.transform(image)
        label = item['label']
        return image, label

class ImageNetDataLoader:
    def __init__(self, batch_size=80, img_size=64, use_cuda=False, mode='train'):
        self.batch_size = batch_size
        self.img_size = img_size
        self.use_cuda = use_cuda
        self.mode = mode
        self.transform = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def create_dataloader(self):
        dataset = ImageNetDataset(split=self.mode, img_size=self.img_size, transform=self.transform)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, pin_memory=False)
        return dataloader

    def get_dataloader(self):
        return self.create_dataloader()
    
def visualize_imagenet_batch(dataloader, num_samples=5):
    dataiter = iter(dataloader)
    images, labels = next(dataiter)
    print(labels)

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
    img_size = 64
    use_cuda = torch.cuda.is_available()

    dataloader_class = ImageNetDataLoader(batch_size, img_size, use_cuda)
    dataloader = dataloader_class.get_dataloader()

    visualize_imagenet_batch(dataloader, num_samples=5)
