from models.cnn import CNN
from utils.data_loader import get_data_loaders
import torch.optim as optim
import torch.nn as nn
import torch

data_dir = 'data/cifar-10-batches-py'
train_loader, val_loader, _ = get_data_loaders(data_dir)

model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.NAdam(model.parameters(), lr=0.1)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()  
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data  

        optimizer.zero_grad()  
        outputs = model(inputs)  
        loss = criterion(outputs, labels)  
        loss.backward()  
        optimizer.step()  
        
        running_loss += loss.item()
        if i % 100 == 99: 
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {running_loss / 100:.4f}')
            running_loss = 0.0

    model.eval()  
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad(): 
        for data in val_loader:
            inputs, labels = data
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1) 
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {val_loss / len(val_loader):.4f}, '
          f'Validation Accuracy: {100 * correct / total:.2f}%')

torch.save(model.state_dict(), 'model.pth')