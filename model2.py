import os
from monai.transforms import (
    Compose, LoadImage, EnsureChannelFirst, ScaleIntensity, ToTensor
)
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Custom dataset class for NIfTI images using MONAI
class NiftiDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Traverse through the dataset directory and collect image paths with labels
        for label_dir in ['health', 'patient']:
            label = 0 if label_dir == 'health' else 1
            label_path = os.path.join(root_dir, label_dir)
            for filename in os.listdir(label_path):
                if filename.endswith('.nii.gz'):
                    self.image_paths.append(os.path.join(label_path, filename))
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        image = self.transform(image_path) if self.transform else image_path
        # print(f"Image shape: {image.shape}")

        return image, torch.tensor(label, dtype=torch.long)

# Define transformations using MONAI
train_transforms = Compose([
    LoadImage(image_only=True),  # Load the NIfTI image as a numpy array
    EnsureChannelFirst(),        # Ensure that the image has a channel dimension
    ScaleIntensity(),            # Normalize the intensity to a standard range
    ToTensor()                   # Convert the image to a PyTorch tensor
])

# Define directories
train_dir = './Training'
test_dir = './Testing'

# Create dataset objects
train_dataset = NiftiDataset(root_dir=train_dir, transform=train_transforms)
test_dataset = NiftiDataset(root_dir=test_dir, transform=train_transforms)

# Create DataLoader objects for batching and shuffling
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2)
        )

        # Calculate the size of the flattened feature map
        self.fc_input_size = self._get_conv_output_size()

        self.fc_layers = nn.Sequential(
            nn.Linear(self.fc_input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def _get_conv_output_size(self):
        # Create a dummy input tensor with the same size as the input images
        dummy_input = torch.zeros(1, 1, 141, 199, 190)  # Adjust based on your input size
        dummy_output = self.conv_layers(dummy_input)
        return int(torch.prod(torch.tensor(dummy_output.shape[1:])))

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc_layers(x)
        return x

# Instantiate the model, loss function, and optimizer
model = SimpleCNN().to(device)
model = nn.DataParallel(model)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00005)

# Set up TensorBoard
writer = SummaryWriter()

sample_input = torch.zeros(1, 1, 141, 199, 190).to(device)
# writer.add_graph(model, sample_input)

# Training function
def train_model(model, train_loader, criterion, optimizer, num_epochs=100):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        writer.add_scalar('Training Loss', running_loss / len(train_loader), epoch)
        writer.add_scalar('Training Accuracy', accuracy, epoch)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}, Accuracy: {accuracy:.2f}%")



# Train the model
train_model(model, train_loader, criterion, optimizer, num_epochs=40)


# Evaluation function
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    accuracy = 100 * correct / total
    writer.add_scalar('Test Accuracy', accuracy)
    print(f'Test Accuracy: {accuracy:.2f}%')
    return all_labels, all_predictions

# Evaluate the model
all_labels, all_predictions = evaluate_model(model, test_loader)

# Confusion matrix
conf_matrix = confusion_matrix(all_labels, all_predictions)
print(f'Confusion Matrix:\n{conf_matrix}')
# Create a figure for the confusion matrix
fig, ax = plt.subplots()
cax = ax.matshow(conf_matrix, cmap=plt.cm.Blues)
fig.colorbar(cax)

# Set axis labels
ax.set_xlabel('Predicted')
ax.set_ylabel('True')

#writer.add_figure('Confusion Matrix', plt.figure().add_subplot().matshow(conf_matrix))

writer.add_figure('Confusion Matrix', fig)

# Precision, Recall, F1 Score
precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='binary')
print(f'Precision: {precision}, Recall: {recall}, F1 Score: {f1}')
writer.add_scalar('Precision', precision)
writer.add_scalar('Recall', recall)
writer.add_scalar('F1 Score', f1)

# AUC Score and ROC Curve
auc_score = roc_auc_score(all_labels, all_predictions)
fpr, tpr, _ = roc_curve(all_labels, all_predictions)
print(f'AUC Score: {auc_score}')
plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
# plt.show()
writer.add_figure('ROC Curve', plt.gcf())
#
# # Close the TensorBoard writer
writer.close()
#
