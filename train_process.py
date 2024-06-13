import torch
dtype = torch.cuda.FloatTensor


def train(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        # Remove the sequence length dimension while keeping the channel dimension
        outputs = outputs.squeeze(2)
        mse = torch.nn.MSELoss().type(dtype)
        loss = mse(outputs, targets)
        # Calculate SSIM loss
        # loss = 1 - ssim(outputs, targets, data_range=255, size_average=True)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            outputs = outputs.squeeze(2)
            mse = torch.nn.MSELoss().type(dtype)
            # Calculate SSIM score
            mse_score = mse(outputs, targets)

            loss = mse_score

            total_loss += loss.item()

    average_loss = total_loss / len(dataloader)
    return average_loss
