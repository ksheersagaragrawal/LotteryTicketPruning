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
import util as ut


# Define a transformation to normalize pixel values to a range between -1 and 1
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Load the CIFAR10 dataset
train_dataset = datasets.CIFAR10(root='./CIFAR10', train=True, transform=transform, download=True)
test_dataset = datasets.CIFAR10(root='./CIFAR10', train=False, transform=transform, download=True)

# Create data loaders for batching and shuffling
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Set a random seed for reproducibility
torch.manual_seed(42)

class SimpleCNN(nn.Module):
     
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.3)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(0.3)
        self.conv3 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.dropout4 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.dropout5 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, 128)
        self.dropout6 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(128, 10)

    def update_layers(self, new_layers):
        for i, new_layer in enumerate(new_layers):
            if hasattr(self, f'fc{i + 1}'):
                setattr(self, f'fc{i + 1}', new_layer)

    def forward(self, x):
        x = self.dropout1(self.pool1(torch.relu(self.conv1(x))))
        x = self.dropout2(self.pool2(torch.relu(self.conv2(x))))
        x = self.dropout3(self.pool3(torch.relu(self.conv5(self.conv4(self.conv3(x))))))
        x = x.view(-1, 256 * 4 * 4)
        x = self.dropout4(torch.relu(self.fc1(x)))
        x = self.dropout5(torch.relu(self.fc2(x)))
        x = self.dropout6(torch.relu(self.fc3(x)))
        x = self.fc4(x)
        return x
   
# Initialize the model, optimizer, criterion & train the model
model = SimpleCNN()    
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()
epochs = 10
untrained_model = copy.deepcopy(model)
ut.train(model, train_loader, criterion, optimizer, epochs = 10)

# Evaluation on the test set
unpruned_accuracy = ut.calculate_accuracy(model, test_loader)
print(f"Accuracy on the test set: {unpruned_accuracy}%")

unpruned_ece = ut.expected_calibration_error(model, test_loader,'results/CIFAR10/unpruned/unpruned.png')
print(f"ECE on the test set (Unpruned): {unpruned_ece}")

# # Iterative pruning Lottery Ticket Hypothesis
# # Iterative Pruning strategy 2 as defined in Lotetry Ticket Hypothesis paper
# # def iterative_reinit_pruning( model, input_shape, output_shape, train_loader, prune_ratio, prune_iter, max_iter = 1):
    
# #      # Per round pune ratio and number of epochs for every fine tuning
# #     per_round_prune_ratio = prune_ratio/prune_iter
# #     if prune_ratio > 0:
# #         per_round_prune_ratio = 1 - (1 - prune_ratio) ** (1 / prune_iter)

# #     criterion = nn.CrossEntropyLoss()
# #     optimizer = optim.Adam(model.parameters(), lr=0.001)
# #     for epoch in range(prune_iter):
# #         untrained_model = copy.deepcopy(model)
# #         ut.train(model, train_loader, criterion, optimizer)
# #         unpruned_layers = ut.oneshot_reinit_pruning(model, untrained_model, input_shape, output_shape, per_round_prune_ratio)
# #         model.update_layers(unpruned_layers)
# #     ut.train(model, train_loader, criterion, optimizer)
# #     return model

# # Plot Accuracy vs Pruning Ratio
# pruning_ratios = np.arange(.1, 1, .05)
# oneshot_pruning_accuracies = []
# oneshot_pruning_ece = []
# oneshot_reinit_pruning_accuracies = []
# oneshot_reinit_pruning_ece = []
# oneshot_random_reinit_pruning_accuracies = []
# oneshot_random_reinit_pruning_ece = []
# iterative_pruning_accuracies = []
# iterative_pruning_ece = []
# iterative_reinit_pruning_accuracies = []
# iterative_reinit_pruning_ece = []

# for prune_ratio in pruning_ratios:

#     # Iterative pruning accuracy
#     iterative_pruning_model = copy.deepcopy(untrained_model)
#     iterative_pruning_model = ut.iterative_pruning(iterative_pruning_model, input_shape = 32*32*3, output_shape = 10, train_loader = train_loader, prune_ratio = prune_ratio, prune_iter = 5, max_iter = 10)
#     iterative_pruning_accuracies.append(ut.calculate_accuracy(iterative_pruning_model, test_loader))
#     # Iterative pruning ECE
#     iterative_pruning_ece.append(ut.expected_calibration_error(iterative_pruning_model, test_loader,'results/CIFAR10/iterative/iterative'+str(prune_ratio)+'.png'))

#     # Iterative pruning strategy 2 accuracy
#     # iterative_reinit_pruning_model = copy.deepcopy(untrained_model)
#     # iterative_reinit_pruning_model = iterative_reinit_pruning(iterative_reinit_pruning_model, input_shape = 28*28, output_shape = 10, train_loader = train_loader, prune_ratio = prune_ratio, prune_iter = 5, max_iter = 1)
   
#     # One-shot pruning accuracy
#     one_shot_model = copy.deepcopy(model)
#     unpruned_layers = ut.oneshot_pruning(one_shot_model, input_shape = 32*32*3, output_shape = 10, prune_ratio = prune_ratio)
#     one_shot_model.update_layers(unpruned_layers)
#     oneshot_pruning_accuracies.append(ut.calculate_accuracy(one_shot_model, test_loader))
#     # One-shot pruning ECE
#     oneshot_pruning_ece.append(ut.expected_calibration_error(one_shot_model, test_loader,'results/CIFAR10/oneshot/oneshot'+str(prune_ratio)+'.png'))
    
#     # One-shot reinit pruning accuracy
#     one_shot_reinit_model = copy.deepcopy(model)
#     unpruned_layers = ut.oneshot_reinit_pruning(one_shot_reinit_model, untrained_model, input_shape = 32*32*3, output_shape = 10, prune_ratio = prune_ratio)
#     one_shot_reinit_model.update_layers(unpruned_layers)
#     # Retrain the model
#     ut.train(one_shot_reinit_model, train_loader, criterion, optimizer, epochs = 1) 
#     oneshot_reinit_pruning_accuracies.append(ut.calculate_accuracy(one_shot_reinit_model, test_loader))
#     # One-shot pruning ECE
#     oneshot_reinit_pruning_ece.append(ut.expected_calibration_error(one_shot_reinit_model, test_loader, 'results/CIFAR10/oneshot_reinit/oneshot_reinit'+str(prune_ratio)+'.png'))

#     # One-shot random reinit pruning accuracy
#     torch.manual_seed(19)
#     random_model = MLP()
#     torch.manual_seed(42)
#     one_shot_random_reinit_model = copy.deepcopy(model)
#     unpruned_layers = ut.oneshot_reinit_pruning(one_shot_random_reinit_model, random_model, input_shape =32*32*3, output_shape = 10, prune_ratio = prune_ratio)
#     one_shot_random_reinit_model.update_layers(unpruned_layers)
#     # Retrain the model
#     ut.train(one_shot_random_reinit_model, train_loader, criterion, optimizer, epochs = 1) 
#     oneshot_random_reinit_pruning_accuracies.append(ut.calculate_accuracy(one_shot_random_reinit_model, test_loader))
#     # One-shot pruning ECE
#     oneshot_random_reinit_pruning_ece.append(ut.expected_calibration_error(one_shot_random_reinit_model, test_loader, 'results/CIFAR10/oneshotrandom_reinit/oneshotrandom_reinit'+str(prune_ratio)+'.png'))

# # Plot Accuracy vs Pruning Ratio
# plt.figure().clear()
# plt.axhline(y = unpruned_accuracy, color = 'b', linestyle = '--')
# plt.plot(pruning_ratios, oneshot_pruning_accuracies, marker='x')
# plt.plot(pruning_ratios, oneshot_reinit_pruning_accuracies, marker='o')
# plt.plot(pruning_ratios, oneshot_random_reinit_pruning_accuracies, marker='*')
# plt.plot(pruning_ratios, iterative_pruning_accuracies, marker='.')
# #plt.plot(pruning_ratios, iterative_reinit_pruning_accuracies, marker='+')
# plt.legend(["No pruning","One-shot","One-shot Reinit", "One-shot Random Reinit","iterative_pruning_accuracies"], loc ="lower left")
# plt.title("Accuracy vs. Pruning Ratio")
# plt.xlabel("Pruning Ratio")
# plt.ylabel("Accuracy")
# plt.grid()
# plt.savefig('results/CIFAR10/acc_vs_pm.png')

# # Plot ECE vs Pruning Ratio
# plt.figure().clear()
# plt.axhline(y = unpruned_ece, color = 'b', linestyle = '--')
# plt.plot(pruning_ratios, oneshot_pruning_ece, marker='x')
# plt.plot(pruning_ratios, oneshot_reinit_pruning_ece, marker='o')
# plt.plot(pruning_ratios, oneshot_random_reinit_pruning_ece, marker='*')
# plt.plot(pruning_ratios, iterative_pruning_ece, marker='.')
# #plt.plot(pruning_ratios, iterative_reinit_pruning_ece, marker='+')
# plt.legend(["No pruning","One-shot","One-shot Reinit", "One-shot Random Reinit","iterative_pruning_accuracies"], loc ="upper left")
# plt.title("Expected Calibration Error (ECE) vs. Pruning Ratio")
# plt.xlabel("Pruning Ratio")
# plt.ylabel("ECE")
# plt.grid()
# plt.savefig('results/CIFAR10/ece_vs_pruning_ratio.png')