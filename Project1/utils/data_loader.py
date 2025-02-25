import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import torchvision.transforms as transforms

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_cifar10(data_dir):
    train_data = []
    train_labels = []
    for i in range(1, 6): 
        file_path = os.path.join(data_dir, f'data_batch_{i}')
        data_dict = unpickle(file_path)
        train_data.append(data_dict[b'data'])
        train_labels.extend(data_dict[b'labels'])
    train_data = np.concatenate(train_data, axis=0)
    train_labels = np.array(train_labels)
    
    test_file_path = os.path.join(data_dir, 'test_batch')
    test_dict = unpickle(test_file_path)
    test_data = test_dict[b'data']
    test_labels = np.array(test_dict[b'labels'])

    
    train_data = train_data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  
    test_data = test_data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  

    return train_data, train_labels, test_data, test_labels

def preprocess_data(data, labels, transform):
    data_tensor = torch.stack([transform(image) for image in data])  
    labels_tensor = torch.tensor(labels, dtype=torch.long)  
    return TensorDataset(data_tensor, labels_tensor)

def get_data_loaders(data_dir, batch_size=64):
    
    transform = transforms.Compose([
        transforms.ToTensor(),  
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  
    ])
   
    train_data, train_labels, test_data, test_labels = load_cifar10(data_dir)
 
    train_val_dataset = preprocess_data(train_data, train_labels, transform)
    test_dataset = preprocess_data(test_data, test_labels, transform)

    train_size = int(0.8 * len(train_val_dataset)) 
    val_size = len(train_val_dataset) - train_size  
    train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader