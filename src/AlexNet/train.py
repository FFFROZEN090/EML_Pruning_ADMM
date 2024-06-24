import sys
sys.path.append('/home/jli/EML_Pruning_ADMM')
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from dataloader.ImageNet_dataloader import ImageNetDataLoader
from AlexNet import AlexNet
import wandb


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
        for data, target in self.dataloader:
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

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Each time only load 2 batches
    batch_size = 1000
    img_size = 224

        
    best_loss = float('inf')
    model = AlexNet().to(device)
    for epoch in range(100):
        # Load data for training
        dataloader = ImageNetDataLoader(batch_size=batch_size, img_size=img_size, use_cuda=torch.cuda.is_available()).get_dataloader()
        trainer = Trainer(model, dataloader, device)
        loss = trainer.train_epoch()
        wandb.log({"loss": loss})
        if loss < best_loss:
            best_loss = loss
            trainer.save_model('best_model.pth')

    wandb.finish()

if __name__ == "__main__":
    wandb.init(
        project='EML_Pruning',
        config={
            "batch_size": 1000,
            "img_size": 224,
            "learning_rate": 0.001,
            "dataset": "ImageNet",
            "architecture": "AlexNet",
            "epochs": 100,
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        }
        )
    main()