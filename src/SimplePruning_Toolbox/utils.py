import torch

def count_nonzero_params(model):
    return sum(torch.count_nonzero(p).item() for p in model.parameters() if p.requires_grad)

def train(model, dataloader, device, optimizer, criterion, mask=None):
    model.to(device)
    model.train()
    total_loss = 0
    num_batches = 0
    for data, target in dataloader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1
    avg_loss = total_loss / num_batches if num_batches > 0 else 0

    # Apply pruning mask
    if mask is not None:
        for name, param in model.named_parameters():
            if name in mask:
                param.data.mul_(mask[name])
    return avg_loss

def evaluate(model, dataloader, device, criterion):
    model.to(device)
    model.eval()
    total_loss = 0
    all_correct_num = 0
    all_sample_num = 0
    num_batches = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            pred = torch.argmax(output, dim=-1)
            current_correct_num = pred == target
            all_correct_num += torch.sum(current_correct_num).item()
            all_sample_num += current_correct_num.size(0)
            num_batches += 1
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    accuracy = all_correct_num / all_sample_num if all_sample_num > 0 else 0
    return avg_loss, accuracy