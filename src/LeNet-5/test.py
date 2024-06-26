import torch

class Tester:
    def __init__(self, model, dataloader, device, criterion):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.criterion = criterion

    def evaluate(self):
        self.model.eval()
        total_loss = 0
        all_correct_num = 0
        all_sample_num = 0
        num_batches = 0
        with torch.no_grad():
            for data, target in self.dataloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()
                pred = torch.argmax(output, dim=-1)
                current_correct_num = pred == target
                all_correct_num += torch.sum(current_correct_num).item()
                all_sample_num += current_correct_num.size(0)
                num_batches += 1
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        accuracy = all_correct_num / all_sample_num if all_sample_num > 0 else 0
        return avg_loss, accuracy
