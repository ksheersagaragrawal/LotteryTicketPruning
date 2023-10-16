import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Define a Sequential model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__() 
        self.seq = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=4, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.seq(x)

# Initialize the model and define loss and optimizer
net = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

# Define data transformations and load CIFAR-10 dataset
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load the CIFAR10 dataset
train_dataset = datasets.CIFAR10(root='./CIFAR10', train=True, transform=transform, download=True)
test_dataset = datasets.CIFAR10(root='./CIFAR10', train=False, transform=transform, download=True)

# Create data loaders for batching and shuffling
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Training loop
epochs = 10
for epoch in range(epochs):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch + 1}, Loss: {running_loss / (i + 1)}')

print('Finished Training')
# Evaluation on the test set
unpruned_accuracy = ut.calculate_accuracy(net, test_loader)
print(f"Accuracy on the test set: {unpruned_accuracy}%")

unpruned_ece = ut.expected_calibration_error(net, test_loader,'results/CIFAR10/unpruned/unpruned.png')
print(f"ECE on the test set (Unpruned): {unpruned_ece}")

