"""
This module implements the ADMM-based pruning algorithm.
ADMM pruning takes a model, a training dataloader, a loss function, and a few hyperparameters as input.
It has following pipeline:
1. Initialize Z and U
2. Train the model using the training dataloader
3. Update Z and U
4. Apply pruning mask to the model
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

import sys
sys.path.append('src') # Suppose entry from root
from LeNet_5 import LeNet5
from .utils import train, evaluate, count_nonzero_params

def admm_pruning(model, train_loader, percent, epochs=10, rho=0.01, lr=1e-3, device=None):
    """
    ADMM-based pruning algorithm
    Args:
        model: A PyTorch model for pruning
        train_loader: A PyTorch dataloader for training
        percent: A list of pruning rates for each layer
        epochs: Number of training epochs
        rho: ADMM hyperparameter
        lr: Learning rate
        device: Device to run the model

    Returns:
        model: Pruned model
        mask: Pruning mask
    
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Initialize Z and U
    Z = ()
    U = ()
    for name, param in model.named_parameters():
        if name.split('.')[-1] == "weight":
            Z += (param.detach().clone(),)
            U += (torch.zeros_like(param),)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    def criterion(output, target):
        idx = 0
        loss = F.cross_entropy(output, target)
        for name, param in model.named_parameters():
            if name.split('.')[-1] == "weight":
                loss += rho / 2 * (param - Z[idx] + U[idx]).norm()
                # loss += alpha * param.norm()
                idx += 1
        return loss
    
    def update_X():
        X = ()
        for name, param in model.named_parameters():
            if name.split('.')[-1] == "weight":
                X += (param.detach().clone(),)
        return X
    
    def update_Z(X, U):
        with torch.no_grad():
            new_Z = ()
            idx = 0
            for x, u in zip(X, U):
                z = x + u
                z_np = z.clone().detach().cpu().numpy()
                pcen = np.percentile(abs(z_np), 100 * percent[idx])
                under_threshold = abs(z) < torch.tensor(pcen, device=device)
                z.data[under_threshold] = 0
                new_Z += (z,)
                idx += 1
            return new_Z
        
    def update_U(U, X, Z):
        new_U = ()
        for u, x, z in zip(U, X, Z):
            new_u = u + x - z
            new_U += (new_u,)
        return new_U
    
    def apply_prune(model, device):
        dict_mask = {}
        idx = 0
        for name, param in model.named_parameters():
            if name.split('.')[-1] == "weight":
                mask = prune_weight(param, device, percent[idx])
                param.data.mul_(mask)
                dict_mask[name] = mask
                idx += 1
        return dict_mask
    
    def prune_weight(weight, device, percent):
        weight_numpy = weight.detach().cpu().numpy()
        pcen = np.percentile(abs(weight_numpy), 100 * percent)
        under_threshold = abs(weight_numpy) < pcen
        weight_numpy[under_threshold] = 0
        mask = torch.Tensor(abs(weight_numpy) >= pcen).to(device)
        return mask

    for epoch in range(epochs):
        model.train()
        print(f"Epoch {epoch+1}/{epochs}")
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            
            loss.backward()
            optimizer.step()

        X = update_X()
        Z = update_Z(X, U)
        U = update_U(U, X, Z)
        mask = apply_prune(model, device)
        for name, param in model.named_parameters():
            if name in mask:
                param.data.mul_(mask[name])

        test_loss, accuracy = evaluate(model, train_loader, device, criterion)
        print(f'Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}')
    
    return model, mask

def main():
    use_wandb = True
    # WANDB_API = '7bf7e888843b3737561b6df5791c583ae0510730'
    # if use_wandb:
    #     import wandb
    #     wandb.login(key=WANDB_API)
    #     wandb.init(project='EML_Pruning', entity='wandb', name='lenet')
    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load the saved model parameters
    model = LeNet5()
    model.load_state_dict(torch.load('./models/mnist_best_model.pth'))

    # Make a copy of the model parameters
    model2 = LeNet5()
    model2.load_state_dict(torch.load('./models/mnist_best_model.pth'))

    batch_size = 256
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root='./train', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = datasets.MNIST(root='./test', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()

    test_loss, accuracy = evaluate(model, test_loader, device, criterion)
    print(f'Original model, Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}')

    # Perform ADMM pruning
    percent = [0.95, 0.95, 0.95, 0.95, 0.95]
    pruned_model, mask = admm_pruning(model, train_loader, percent)

    # Evaluate the pruned model
    test_loss, accuracy = evaluate(pruned_model, test_loader, device, criterion)
    print(f'Pruned model, Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}')

    # Retraining
    print("Retraining the pruned model")
    fine_tune_epochs = 5
    optimizer = optim.Adam(pruned_model.parameters(), lr=1e-3)
    for epoch in range(fine_tune_epochs):
        train(pruned_model, train_loader, device, optimizer, criterion, mask)
        test_loss, accuracy = evaluate(pruned_model, test_loader, device, criterion)
        print(f'Epoch: {epoch + 1}, Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}')

    print("Model finished fine-tuning")
    # Save the pruned model parameters
    torch.save(pruned_model.state_dict(), 'pruned_model.pth')

    # Compare the model sizes by counting the non-zero parameters 
    print(f"Original model size: {count_nonzero_params(model2)}")
    print(f"Pruned model size: {count_nonzero_params(pruned_model)}")


if __name__ == "__main__":
    main()
