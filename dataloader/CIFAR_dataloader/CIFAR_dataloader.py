import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from matplotlib import pyplot as plt

class CIFARDataset(Dataset):
    def __init__(self, split='train', img_size=32, transform=None):
        # Using CIFAR-10 dataset
        self.dataset = datasets.CIFAR10(root='./data', train=(split == 'train'),
                                        download=True, transform=transform)
        self.img_size = img_size

        # Check shape of dataset
        print(f"Number of images in {split} dataset: {len(self.dataset)}")
        

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Accessing data via indexing
        return self.dataset[idx]

class CIFARDataLoader:
    def __init__(self, batch_size=64, img_size=32, use_cuda=False, mode='train'):
        self.batch_size = batch_size
        self.img_size = img_size
        self.use_cuda = use_cuda
        self.mode = mode
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        ])

    def create_dataloader(self):
        dataset = CIFARDataset(split=self.mode, img_size=self.img_size, transform=self.transform)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, pin_memory=self.use_cuda)
        return dataloader

    def get_dataloader(self):
        return self.create_dataloader()

def visualize_cifar_batch(dataloader, num_samples=5):
    dataiter = iter(dataloader)
    images, labels = next(dataiter)

    images = images.numpy()
    images = images.transpose(0, 2, 3, 1)  # Change dimension order for plotting

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
    img_size = 32  # Standard CIFAR-10 image size
    use_cuda = torch.cuda.is_available()

    dataloader_class = CIFARDataLoader(batch_size, img_size, use_cuda)
    dataloader = dataloader_class.get_dataloader()

    visualize_cifar_batch(dataloader, num_samples=5)
