import sys
sys.path.append('.')
import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
from dataloader.CIFAR_dataloader.CIFAR_dataloader import CIFARDataLoader
from VGG_8 import VGG_8
import matplotlib.pyplot as plt


class Trainer:
    def __init__(self, model, dataloader, device):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        num_batches = 0  # Count the number of batches processed
        for data, target in tqdm.tqdm(self.dataloader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            num_batches += 1  # Increment batch count
        return total_loss / num_batches if num_batches > 0 else 0  # Avoid division by zero

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

class Validator:
    def __init__(self, model, dataloader, device):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

    def validate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in tqdm.tqdm(self.dataloader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()
                predicted = torch.argmax(output, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        avg_loss = total_loss / len(self.dataloader)
        accuracy = 100 * correct / total
        return avg_loss, accuracy

def visualize_imagenet_batch(data, target):
    images = data
    labels = target

    # Get the first 10 images and labels from the batch
    images = images[:10]
    labels = labels[:10]

    # Conver to cpu and numpy

    images = images.cpu().numpy()
    images = images.transpose(0, 2, 3, 1)
    images = (images * 0.5) + 0.5

    # Compose the 10 images into a single plot
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i])
        ax.set_title(f'Label: {labels[i].item()}')
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    plt.savefig('imagenet_batch.png')

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Each time only load 2 batches
    batch_size = 512
    img_size = 64

        
    best_loss = float('inf')
    model = VGG_8(10).to(device)
    for epoch in range(300):
        # Load data for training
        dataloader = CIFARDataLoader(batch_size=batch_size, use_cuda=torch.cuda.is_available(),mode='train').get_dataloader()
        trainer = Trainer(model, dataloader, device)
        loss = trainer.train_epoch()
        if loss < best_loss:
            best_loss = loss
            trainer.save_model('VGG_best_model.pth')
        # Load data for validation every 1/10 epochs
        if epoch % 1 == 0:
            dataloader = CIFARDataLoader(batch_size=batch_size, use_cuda=torch.cuda.is_available(),mode='valid').get_dataloader()
            validator = Validator(model, dataloader, device)
            val_loss, val_accuracy = validator.validate()
            print(f"Epoch: {epoch}, Loss: {loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
        


if __name__ == "__main__":
    main()