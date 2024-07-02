"""
This module implements the ADMM-based pruning algorithm.
ADMM pruning takes a model, a training dataloader, a loss function, and a few hyperparameters as input.
It has following pipeline:
1. Initialize Z and U
2. Train the model using the training dataloader
3. Update Z and U
4. Apply pruning mask to the model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

def admm_pruning(model, train_loader, criterion, epochs=5, rho=1e-3, pruning_ratio=0.5, lr=1e-3, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Initialize Z and U
    Z = {name: param.clone().detach().to(device) for name, param in model.named_parameters() if param.requires_grad}
    U = {name: torch.zeros_like(param).to(device) for name, param in model.named_parameters() if param.requires_grad}

    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        print(f"Epoch {epoch+1}/{epochs}")
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            
            for name, param in model.named_parameters():
                if param.requires_grad:
                    loss += rho / 2 * torch.norm(param - Z[name] + U[name]) ** 2
            
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        Z[name] = torch.max(param + U[name] - pruning_ratio, torch.zeros_like(param))
                        U[name] += param - Z[name]

    # Apply pruning mask
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.requires_grad:
                mask = (torch.abs(param) > pruning_ratio).float()
                param.mul_(mask)

    return model

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, 1, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)
        self.relu5 = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        y = self.relu3(y)
        y = self.fc2(y)
        y = self.relu4(y)
        y = self.fc3(y)
        y = self.relu5(y)
        return y

def count_nonzero_params(model):
    return sum(torch.count_nonzero(p).item() for p in model.parameters() if p.requires_grad)

def main():
    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load the saved model parameters
    model = LeNet5()
    model.load_state_dict(torch.load('/home/jli/EML_Pruning_ADMM/src/LeNet-5/models/mnist_best_model.pth'))

    # Make a copy of the model parameters
    model2 = LeNet5()
    model2.load_state_dict(torch.load('/home/jli/EML_Pruning_ADMM/src/LeNet-5/models/mnist_best_model.pth'))

    batch_size = 256
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root='./train', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()

    # Perform ADMM pruning
    pruned_model = admm_pruning(model, train_loader, criterion)

    # Fine-tuning the pruned model
    fine_tune_epochs = 3
    optimizer = optim.Adam(pruned_model.parameters(), lr=1e-3)
    for epoch in range(fine_tune_epochs):
        pruned_model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = pruned_model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    # Save the pruned model parameters
    torch.save(pruned_model.state_dict(), 'pruned_model.pth')

    # Compare the model sizes by counting the non-zero parameters 
    print(f"Original model size: {count_nonzero_params(model2)}")
    print(f"Pruned model size: {count_nonzero_params(pruned_model)}")


if __name__ == "__main__":
    main()
