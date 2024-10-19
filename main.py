# Import necessary libraries
import nibabel as nib
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import ConcatDataset
from torchvision.transforms import v2
import matplotlib.pyplot as plt

# Check if a GPU is available and set the device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

transforms = v2.Compose([
    v2.RandomResizedCrop(size=(224, 224), antialias=True),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
# Custom dataset class for NIfTI images
class NiftiDataset(Dataset):
    def __init__(self, img_dir, label, transform=transforms):
        """
        Args:
            img_dir (str): Directory with all the images.
            label (int): Label for the images (0 for 'health', 1 for 'patient').
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.img_dir = img_dir
        self.label = label
        self.transform = transform
        self.img_names = [f for f in os.listdir(img_dir) if f.endswith('.nii.gz')]

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = nib.load(img_path).get_fdata()

        # Normalize the image and convert to float32
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        image = image.astype(np.float32)

        if self.transform:
            image = self.transform(image)

        # Add a channel dimension for the CNN (expected input shape: CxDxHxW)
        image = torch.from_numpy(image).unsqueeze(0)

        return image, self.label


# Define directories
train_health_dir = './Training/health/'
train_patient_dir = './Training/patient/'
test_health_dir = './Testing/health/'
test_patient_dir = './Testing/patient/'

# Create training and testing datasets
train_health_dataset = NiftiDataset(train_health_dir, label=0)
train_patient_dataset = NiftiDataset(train_patient_dir, label=1)
test_health_dataset = NiftiDataset(test_health_dir, label=0)
test_patient_dataset = NiftiDataset(test_patient_dir, label=1)

# Combine training datasets and testing datasets
train_dataset = ConcatDataset([train_health_dataset, train_patient_dataset])
test_dataset = ConcatDataset([test_health_dataset, test_patient_dataset])

# Data loaders
batch_size = 2
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)




# Define the CNN model (using the architecture you provided)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv3d(1, 8, kernel_size=5, stride=2, padding=0)
        self.pool = nn.MaxPool3d((2, 2, 2), stride=1)
        self.conv2 = nn.Conv3d(8, 16, kernel_size=(5, 5, 5), stride=2, padding=0)
        self.conv3 = nn.Conv3d(16, 32, kernel_size=(5, 5, 5), stride=2, padding=0)
        self.conv4 = nn.Conv3d(32, 64, kernel_size=(5, 5, 5), stride=2, padding=0)
        self.fc1 = nn.Linear(int(25088 / batch_size), 500)
        self.fc2 = nn.Linear(500, 120)
        self.fc3 = nn.Linear(120, 2)

    def forward(self, x):
        xSize = np.array(x.size())
        xD = xSize[2]
        xH = xSize[3]
        xW = xSize[4]
        x = x.reshape(batch_size, 1, xD, xH, xW)
        x = self.pool(F.relu(self.conv1(x.float())))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(batch_size, int(25088 / batch_size))
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc3(x)
        return x


# Instantiate the model, move it to the GPU, define the loss function, and the optimizer
net = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0005)


# Training function
def train_model(net, train_loader, criterion, optimizer, num_epochs=100):
    net.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}')


# Evaluation function
def evaluate_model(net, test_loader):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy}%')


# Train the model
train_model(net, train_loader, criterion, optimizer, num_epochs=100)

# Evaluate the model
evaluate_model(net, test_loader)
