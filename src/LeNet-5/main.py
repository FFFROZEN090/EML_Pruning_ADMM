import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from train import Trainer
from test import Tester
from LeNet5 import LeNet5


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 256
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root='./train', train=True, download=False, transform=transform)
    test_dataset = datasets.MNIST(root='./test', train=False, download=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = LeNet5().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    for epoch in range(30):
        trainer = Trainer(model, train_loader, device, optimizer, criterion)
        train_loss = trainer.train_epoch()

        tester = Tester(model, test_loader, device, criterion)
        test_loss, accuracy = tester.evaluate()

        print(f'Epoch: {epoch + 1}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}')

        if accuracy > best_acc:
            best_acc = accuracy
            trainer.save_model('models/mnist_best_model.pth')

    print("Model finished training")
    if not os.path.isdir("models"):
        os.mkdir("models")
    trainer.save_model('models/mnist_final_model.pth')


if __name__ == "__main__":
    main()
