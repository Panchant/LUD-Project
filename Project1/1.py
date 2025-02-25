import torch.nn as nn
import torch
import torchvision
import torch.optim as optim
import torchvision.transforms as transform
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import os

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

transform = transform.Compose([
    transform.ToTensor(),
    transform.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def load_cifar10(data_dir):
    train_data = []
    train_labels = []
    for i in range(1,6):
        file_path = os.path.join(data_dir,f'data_batch_{i}')
        data_dict = unpickle(file_path)
        train_data.append(data_dict[b'data'])
        train_labels.extend(data_dict[b'labels'])
    train_data = np.concatenate(train_data, axis = 0)
    train_labels = np.array(train_labels)

    test_file_path = os.path.join(data_dir, 'test_batch')
    test_dict = unpickle(test_file_path)
    test_data = test_dict[b'data']
    test_labels = np.array(test_dict[b'labels'])

    train_data = train_data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  
    test_data = test_data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  

    return train_data, train_labels, test_data, test_labels

data_dir = '../cifar-10-batches-py'

train_data, train_labels, test_data, test_labels = load_cifar10(data_dir)

def preprocess_data(data, labels, transform):
    data_tensor = torch.stack([transform(image) for image in data])  
    labels_tensor = torch.tensor(labels, dtype=torch.long)  
    return TensorDataset(data_tensor, labels_tensor)

train_val_dataset = preprocess_data(train_data, train_labels, transform)
test_dataset = preprocess_data(test_data, test_labels, transform)


train_size = int(0.8 * len(train_val_dataset))  # 80% 
val_size = len(train_val_dataset) - train_size  # 20% 
train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size])

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class CNNet(nn.Module):
    def __init__(self):
        super(CNNet, self).__init__()
        self.Conv1 = nn.Conv2d(3, 32, 3, padding = 1) #(32,32,32)
        self.Pool = nn.MaxPool2d(2, 2) #(32,16,16)
        self.Conv2 = nn.Conv2d(32, 64, 3, padding = 1) #(64,16,16)
        self.Conv3 = nn.Conv2d(64, 64, 3, padding = 1) #
        self.fc1 = nn.Linear(64*8*8,64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.Pool(torch.relu(self.Conv1(x)))
        x = self.Pool(torch.relu(self.Conv2(x)))
        x = torch.relu(self.Conv3(x))
        x = x.view(-1, 64*8*8)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

net = CNNet()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = 0.01)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)

epochs = 10
loss_history = []
for epoch in range(epochs):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:
            print(f'[{epoch+1}, {i+1}] loss: {running_loss / 100}')
            running_loss = 0.0
    
    epoch_loss = running_loss / len(train_loader)
    loss_history.append(epoch_loss)

correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

final_accuracy = 100 * correct / total

print(f'Accuracy of the network on the 10000 test images: {final_accuracy} %')

import matplotlib.pyplot as plt

# 可视化损失
plt.plot(range(1, epochs + 1), loss_history)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss History')
plt.show()

# 可视化准确率
plt.bar(1, final_accuracy, width=0.4, label='Final Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Final Accuracy on Test Set')
plt.legend()
plt.show()
