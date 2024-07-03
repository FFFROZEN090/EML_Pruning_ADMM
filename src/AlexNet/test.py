import sys
sys.path.append('/home/jli/EML_Pruning_ADMM')
import torch
import torch.nn as nn
from dataloader.ImageNet_dataloader import ImageNetDataLoader
from torchvision import models
from AlexNet import AlexNet
import wandb

class Tester:
    def __init__(self, model, dataloader, device):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        # Use Accuracy as the metric
        self.criterion = nn.CrossEntropyLoss()

    def evaluate(self):
        self.model.eval()  # Set model to evaluate mode
        total_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in self.dataloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        avg_loss = total_loss / len(self.dataloader)
        accuracy = 100 * correct / total
        return avg_loss, accuracy

def load_model(path, device):
    model = AlexNet().to(device)
    model.load_state_dict(torch.load(path))
    return model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Configuration for test data
    batch_size = 100
    img_size = 224

    # Load model
    model_path = 'best_model.pth'
    model = load_model(model_path, device)

    # Setup data loader for test data
    dataloader = ImageNetDataLoader(batch_size=batch_size, img_size=img_size, use_cuda=torch.cuda.is_available(), mode='valid').get_dataloader()
    
    tester = Tester(model, dataloader, device)
    loss, accuracy = tester.evaluate()
    print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.2f}%")


    # Use pre-trained AlexNet model for comparison
    model = models.alexnet(pretrained=True)
    model = model.to(device)
    tester = Tester(model, dataloader, device)
    loss, accuracy = tester.evaluate()
    print(f"Pre-trained AlexNet Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()
