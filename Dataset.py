import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader,random_split

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(device)

# Data loader - train and test dataloader

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
data_dir = '/home/dgxfac14/Work/plantvillage dataset/color'

dataset = ImageFolder(data_dir, transform=transform)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Define data loaders for train and test sets
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)
