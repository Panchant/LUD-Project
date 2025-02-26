from models.cnn import CNN
from utils.data_loader import get_data_loaders
import torch.optim as optim
import torch.nn as nn
import torch
import wandb

wandb.init(project="cifar10_experiment", config={
    "learning_rate": 0.001,
    "optimizer": "NAdam",
    "epochs": 10,
    "batch_size": 64
})

data_dir = 'data/cifar-10-batches-py'
train_loader, val_loader, _ = get_data_loaders(data_dir)

model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.NAdam(model.parameters(), lr=0.001)

epochs = 10
for epoch in range(epochs):
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
            print(f'Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {running_loss / 100:.4f}')
            
            wandb.log({"Training Loss": running_loss / 100, "Step": epoch * len(train_loader) + i})

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
    
    val_accuracy = 100 * correct / total
    
    wandb.log({
        "Validation Loss": val_loss / len(val_loader),
        "Validation Accuracy": val_accuracy,
        "Epoch": epoch
    })

    print(f'Epoch [{epoch + 1}/{epochs}], Validation Loss: {val_loss / len(val_loader):.4f}, '
          f'Validation Accuracy: {100 * correct / total:.2f}%')

torch.save(model.state_dict(), 'model.pth')
wandb.finish()