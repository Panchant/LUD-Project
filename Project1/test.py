from models.cnn import CNN
from utils.data_loader import get_data_loaders
import torch.nn as nn
import torch


data_dir = 'data/cifar-10-batches-py'  
_, _, test_loader = get_data_loaders(data_dir)


model = CNN()
model.load_state_dict(torch.load('model.pth')) 

criterion = nn.CrossEntropyLoss()


test_loss = 0.0
correct = 0
total = 0
with torch.no_grad():  
    for data in test_loader:
        inputs, labels = data
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)  
        total += labels.size(0)
        correct += (predicted == labels).sum().item()


print(f'Test Loss: {test_loss / len(test_loader):.4f}, '
      f'Test Accuracy: {100 * correct / total:.2f}%')