import copy
import random
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import lotteryticket_v1.mainutil as ut


# Define a transformation to normalize pixel values to a range between -1 and 1
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Load the CIFAR10 dataset
train_dataset = datasets.CIFAR10(root='./CIFAR10', train=True, transform=transform, download=True)
test_dataset = datasets.CIFAR10(root='./CIFAR10', train=False, transform=transform, download=True)

# Create data loaders for batching and shuffling
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__() 
        # Define the layers for your model in the constructor
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def update_layers(self, new_layers):
        for i, new_layer in enumerate(new_layers):
            if hasattr(self, f'fc{i + 1}'):
                setattr(self, f'fc{i + 1}', new_layer)

    def forward(self, x):
        # Define the forward pass
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# # Set a random seed for reproducibility
# torch.manual_seed(42)
# class Cifar10CnnModel(nn.Module):
#     def __init__(self):
#         super(Cifar10CnnModel, self).__init__()
#         self.network = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2), # output: 64 x 16 x 16
#             nn.BatchNorm2d(64),

#             nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2), # output: 128 x 8 x 8
#             nn.BatchNorm2d(128),

#             nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2), # output: 256 x 4 x 4
#             nn.BatchNorm2d(256),

#             nn.Flatten(), 
#             nn.Linear(256*4*4, 1024),
#             nn.ReLU(),
#             nn.Linear(1024, 512),
#             nn.ReLU(),
#             nn.Linear(512, 10))
        
#     def update_layers(self, new_layers):
#         for i, new_layer in enumerate(new_layers):
#             if hasattr(self, f'fc{i + 1}'):
#                 setattr(self, f'fc{i + 1}', new_layer)
        
#     def forward(self, xb):
#         return self.network(xb)

# class SimpleCNN(nn.Module):
     
#     def __init__(self):
#         super(SimpleCNN, self).__init__()
#         self.conv1 = nn.Conv2d(3, 128, kernel_size=3, padding=1)
#         self.pool1 = nn.MaxPool2d(2, 2)
#         self.dropout1 = nn.Dropout(0.3)
#         self.conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
#         self.pool2 = nn.MaxPool2d(2, 2)
#         self.dropout2 = nn.Dropout(0.3)
#         self.conv3 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
#         self.conv4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
#         self.conv5 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
#         self.pool3 = nn.MaxPool2d(2, 2)
#         self.dropout3 = nn.Dropout(0.3)
#         self.fc1 = nn.Linear(256 * 4 * 4, 512)
#         self.dropout4 = nn.Dropout(0.5)
#         self.fc2 = nn.Linear(512, 256)
#         self.dropout5 = nn.Dropout(0.5)
#         self.fc3 = nn.Linear(256, 128)
#         self.dropout6 = nn.Dropout(0.5)
#         self.fc4 = nn.Linear(128, 10)

#     def update_layers(self, new_layers):
#         for i, new_layer in enumerate(new_layers):
#             if hasattr(self, f'fc{i + 1}'):
#                 setattr(self, f'fc{i + 1}', new_layer)

#     def forward(self, x):
#         x = self.dropout1(self.pool1(torch.relu(self.conv1(x))))
#         x = self.dropout2(self.pool2(torch.relu(self.conv2(x))))
#         x = self.dropout3(self.pool3(torch.relu(self.conv5(self.conv4(self.conv3(x))))))
#         x = x.view(-1, 256 * 4 * 4)
#         x = self.dropout4(torch.relu(self.fc1(x)))
#         x = self.dropout5(torch.relu(self.fc2(x)))
#         x = self.dropout6(torch.relu(self.fc3(x)))
#         x = self.fc4(x)
#         return x
   
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device('mps')
print(device)
# Initialize the model, optimizer, criterion & train the model
# model = Cifar10CnnModel().to(device)
model = SimpleCNN().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
criterion = nn.CrossEntropyLoss()
epochs = 10
untrained_model = copy.deepcopy(model)
ut.train(model, train_loader, criterion, optimizer, epochs = epochs, device=device)

# Evaluation on the train set
unpruned_accuracy = ut.calculate_accuracy(model, train_loader, device=device)
print(f"Accuracy on the train set: {unpruned_accuracy}%")

# Evaluation on the test set
unpruned_accuracy = ut.calculate_accuracy(model, test_loader, device=device)
print(f"Accuracy on the test set: {unpruned_accuracy}%")

unpruned_ece = ut.expected_calibration_error(model, test_loader,'results/CIFAR10/unpruned/unpruned.png', device=device)
print(f"ECE on the test set (Unpruned): {unpruned_ece}")

print('Starting Pruning')
# Plot Accuracy vs Pruning Ratio
pruning_ratios = np.arange(.1, 1, .05)
oneshot_pruning_accuracies = []
oneshot_pruning_ece = []
oneshot_reinit_pruning_accuracies = []
oneshot_reinit_pruning_ece = []
oneshot_random_reinit_pruning_accuracies = []
oneshot_random_reinit_pruning_ece = []
iterative_pruning_accuracies = []
iterative_pruning_ece = []
iterative_reinit_pruning_accuracies = []
iterative_reinit_pruning_ece = []

for prune_ratio in pruning_ratios:

    print('Iterative pruning')
    # Iterative pruning accuracy
    iterative_pruning_model = copy.deepcopy(untrained_model)
    iterative_pruning_model = ut.iterative_pruning(iterative_pruning_model, input_shape = 64 * 8 * 8, output_shape = 10, train_loader = train_loader, prune_ratio = prune_ratio, prune_iter = 5, max_iter = 10, device=device)
    iterative_pruning_accuracies.append(ut.calculate_accuracy(iterative_pruning_model, test_loader, device=device))

    # Iterative pruning ECE
    iterative_pruning_ece.append(ut.expected_calibration_error(iterative_pruning_model, test_loader,'results/CIFAR10/iterative/iterative'+str(prune_ratio)+'.png', device=device))

    # One-shot pruning accuracy
    print('One-shot pruning')
    one_shot_model = copy.deepcopy(model)
    unpruned_layers = ut.oneshot_pruning(one_shot_model, input_shape = 64 * 8 * 8, output_shape = 10, prune_ratio = prune_ratio)
    one_shot_model.update_layers(unpruned_layers)
    oneshot_pruning_accuracies.append(ut.calculate_accuracy(one_shot_model, test_loader, device=device))
    # One-shot pruning ECE
    oneshot_pruning_ece.append(ut.expected_calibration_error(one_shot_model, test_loader,'results/CIFAR10/oneshot/oneshot'+str(prune_ratio)+'.png', device=device))
    
    # One-shot reinit pruning accuracy
    print('One-shot reinit pruning')
    one_shot_reinit_model = copy.deepcopy(model)
    reinit_model = copy.deepcopy(untrained_model)
    unpruned_layers = ut.oneshot_reinit_pruning(one_shot_reinit_model, untrained_model, input_shape = 64 * 8 * 8, output_shape = 10, prune_ratio = prune_ratio)
    reinit_model.update_layers(unpruned_layers)
    # Retrain the model
    ut.train(reinit_model, train_loader, criterion, optimizer, epochs = 10, device=device) 
    oneshot_reinit_pruning_accuracies.append(ut.calculate_accuracy(reinit_model, test_loader, device=device))
    # One-shot pruning ECE
    oneshot_reinit_pruning_ece.append(ut.expected_calibration_error(reinit_model, test_loader, 'results/CIFAR10/oneshot_reinit/oneshot_reinit'+str(prune_ratio)+'.png', device=device))

    # One-shot random reinit pruning accuracy
    torch.manual_seed(19)
    random_model = SimpleCNN().to(device)
    torch.manual_seed(42)
    one_shot_random_reinit_model = copy.deepcopy(model)
    unpruned_layers = ut.oneshot_reinit_pruning(one_shot_random_reinit_model, random_model, input_shape = 64 * 8 * 8, output_shape = 10, prune_ratio = prune_ratio)
    random_model.update_layers(unpruned_layers)
    # Retrain the model
    ut.train(random_model, train_loader, criterion, optimizer, epochs = 10, device=device) 
    oneshot_random_reinit_pruning_accuracies.append(ut.calculate_accuracy(random_model, test_loader, device=device))
    # One-shot pruning ECE
    oneshot_random_reinit_pruning_ece.append(ut.expected_calibration_error(random_model, test_loader, 'results/CIFAR10/oneshotrandom_reinit/oneshotrandom_reinit'+str(prune_ratio)+'.png', device=device))

print('Plotting')
# Plot Accuracy vs Pruning Ratio
plt.figure().clear()
plt.axhline(y = unpruned_accuracy, color = 'b', linestyle = '--')
plt.plot(pruning_ratios, oneshot_pruning_accuracies, marker='x')
plt.plot(pruning_ratios, oneshot_reinit_pruning_accuracies, marker='o')
plt.plot(pruning_ratios, oneshot_random_reinit_pruning_accuracies, marker='*')
plt.plot(pruning_ratios, iterative_pruning_accuracies, marker='.')
plt.legend(["No pruning","One-shot", "One-shot Re-Init", "One-shot random Re-Init", "Iterative"], loc ="lower left")
plt.title("Accuracy vs. Pruning Ratio")
plt.xlabel("Pruning Ratio")
plt.ylabel("Accuracy")
plt.grid()
plt.savefig('results/CIFAR10/acc_vs_pm.png')

# Plot ECE vs Pruning Ratio
plt.figure().clear()
plt.axhline(y = unpruned_ece, color = 'b', linestyle = '--')
plt.plot(pruning_ratios, oneshot_pruning_ece, marker='x')
plt.plot(pruning_ratios, oneshot_reinit_pruning_ece, marker='o')
plt.plot(pruning_ratios, oneshot_random_reinit_pruning_ece, marker='*')
plt.plot(pruning_ratios, iterative_pruning_ece, marker='.')
plt.legend(["No pruning","One-shot","One-shot Reinit", "One-shot Random Reinit","iterative_pruning_accuracies"], loc ="upper left")
plt.title("Expected Calibration Error (ECE) vs. Pruning Ratio")
plt.xlabel("Pruning Ratio")
plt.ylabel("ECE")
plt.grid()
plt.savefig('results/CIFAR10/ece_vs_pruning_ratio.png')