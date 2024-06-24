import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from custom_dataloader import CustomDataLoader
from alexnet import AlexNet

class Tester:
    def __init__(self, model, dataloader, device):
        self.model = model
        self.dataloader = dataloader
        self.device = device

    def evaluate(self):
        self.model.eval()
        total_correct = 0
        with torch.no_grad():
            for data, target in self.dataloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1, keepdim=True)
                total_correct += pred.eq(target.view_as(pred)).sum().item()
        accuracy = total_correct / len(self.dataloader.dataset)
        return accuracy

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_folder = 'path_to_data'
    test_batch_indices = [3]  # Assuming that batch 3 is for testing
    dataloader = CustomDataLoader(data_folder, test_batch_indices, use_cuda=torch.cuda.is_available()).get_dataloader()
    model = AlexNet().to(device)
    model.load_state_dict(torch.load('model_to_test.pth'))
    tester = Tester(model, dataloader, device)
    
    accuracy = tester.evaluate()
    print(f"Test Accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    main()
