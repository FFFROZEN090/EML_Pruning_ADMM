import torch

class Trainer:
    def __init__(self, model, dataloader, device, optimizer, criterion):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        num_batches = 0
        for data, target in self.dataloader:
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            num_batches += 1
        return total_loss / num_batches if num_batches > 0 else 0

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
