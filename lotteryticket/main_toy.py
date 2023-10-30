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
from uce import uceloss

# Define a transformation to normalize pixel values to a range between -1 and 1
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Define a function to apply random rotations to an image
def random_rotate(image, max_angle=15):
    angle = np.random.uniform(-max_angle, max_angle)
    return transforms.functional.rotate(image, angle)

# Define a transformation to normalize pixel values to a range between -1 and 1 and apply random rotations
transform_with_rotation = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: random_rotate(x)),
    transforms.Normalize((0.5,), (0.5,))
])

# Load the MNIST dataset
train_dataset = datasets.FashionMNIST(root='./FashionMNIST', train=True, transform=transform, download=True)
test_dataset = datasets.FashionMNIST(root='./FashionMNIST', train=False, transform=transform, download=True)
#test_dataset_rotated = datasets.MNIST(root='./MNIST', train=False, transform=transform_with_rotation, download=True)

# Create data loaders for batching and shuffling
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Randomly select samples from the training set for retraining-
retrain_indices = random.sample(range(len(train_dataset)), int(len(train_dataset) * 0.3))
retrain_subset = torch.utils.data.Subset(train_dataset, retrain_indices)
# Create a DataLoader for retraining
retrain_loader = DataLoader(retrain_subset, batch_size=batch_size, shuffle=True)

fine_tune_indices = random.sample(range(len(train_dataset)), int(len(train_dataset) * 0.1))
fine_tune_subset = torch.utils.data.Subset(train_dataset, fine_tune_indices)
# Create a DataLoader for fine-tuning
fine_tune_loader = DataLoader(fine_tune_subset, batch_size=batch_size, shuffle=True)

# Set a random seed for reproducibility
torch.manual_seed(42)

# Define a simple MLP model
class MLP(nn.Module):
    def __init__(self, dropout = False, p = 0.5, N = 20, training = False):
        super(MLP, self).__init__()
        self.dropout = dropout
        self.N = N
        self.training = training
        self.p = p
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 140)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(140, 130)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(130, 120)
        self.relu = nn.ReLU()
        self.fc4 = nn.Linear(120, 110)
        self.relu = nn.ReLU()
        self.fc5 = nn.Linear(110, 10)

    def update_layers(self, new_layers):
        for i, new_layer in enumerate(new_layers):
            if hasattr(self, f'fc{i + 1}'):
                setattr(self, f'fc{i + 1}', new_layer)
    
    def mc_dropout(self, x):
        x_list = F.dropout(x, p=self.p, training=True)
        x_list = self.fc5(x_list).unsqueeze(0)
        for i in range(self.N - 1):
            x_tmp = F.dropout(x, p=self.p, training=True)
            x_tmp = self.fc5(x_tmp).unsqueeze(0)
            x_list = torch.cat([x_list, x_tmp], dim=0)

        x_list = x_list / (1-self.p)
        return x_list

    def forward(self, x):
        x = self.flatten(x)
        if self.dropout:
            x = F.dropout(x, p=self.p, training=self.training)
            x = x / (1-self.p)

        x = self.fc1(x)
        if self.dropout:
            x = F.dropout(x, p=self.p, training=self.training)
            x = x / (1-self.p)
        x = self.relu(x)

        x = self.fc2(x)
        if self.dropout:
            x = F.dropout(x, p=self.p, training=self.training)
            x = x / (1-self.p)
        x= self.relu(x)

        x= self.fc3(x)
        if self.dropout:
            x = F.dropout(x, p=self.p, training=self.training)
            x = x / (1-self.p)
        x= self.relu(x)

        x= self.fc4(x)
        if self.dropout:
            x = F.dropout(x, p=self.p, training=self.training)
            x = x / (1-self.p)
        x= self.relu(x)

        if self.dropout:
            x = self.mc_dropout(x)
        else:
            x = self.fc5(x)
        
            x = torch.softmax(x, dim=1)
        return x
   
# Initialize the model, optimizer, criterion & train the model
model = MLP(dropout=True, p=0.3, training=True)    
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
epochs = 1
untrained_model = copy.deepcopy(model)
ut.train(model, train_loader, criterion, optimizer, epochs, bayesian = True)

def test(net, bayesian):
    logits = []
    labels = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data, target
            output = net(data)
            if bayesian:
                output = torch.softmax(output, dim=2).mean(dim=0)
            else:
                output = torch.softmax(output, dim=1)
            logits.append(output.detach())
            labels.append(target.detach())

    return torch.cat(logits, dim=0), torch.cat(labels, dim=0)

logits, labels = test(model, bayesian=True)

uce1, err1, entr1 = uceloss(logits, labels)
print(uce1.item())
plt.plot([0, 1], [0, 1], linestyle='--')
plt.plot(entr1, err1)
plt.show()
unpruned_layers = ut.oneshot_pruning(model, input_shape = 28*28, output_shape = 10, prune_ratio = 0.4)
model.update_layers(unpruned_layers)
logits, labels = test(model, bayesian=True)
uce1, err1, entr1 = uceloss(logits, labels)
print(uce1.item())
plt.plot([0, 1], [0, 1], linestyle='--')
plt.plot(entr1, err1)
plt.show()





# Evaluation on the test set
# unpruned_accuracy = ut.calculate_accuracy(model, test_loader)
# print(f"Accuracy on the test set: {unpruned_accuracy}%")

# unpruned_ece = ut.expected_calibration_error(model, test_loader,'results/FashionMNIST/unpruned/unpruned.png')
# print(f"ECE on the test set (Unpruned): {unpruned_ece}")

# Iterative pruning Lottery Ticket Hypothesis
# Iterative Pruning strategy 2 as defined in Lotetry Ticket Hypothesis paper
# def iterative_reinit_pruning( model, input_shape, output_shape, train_loader, prune_ratio, prune_iter, max_iter = 1):
    
#      # Per round pune ratio and number of epochs for every fine tuning
#     per_round_prune_ratio = prune_ratio/prune_iter
#     if prune_ratio > 0:
#         per_round_prune_ratio = 1 - (1 - prune_ratio) ** (1 / prune_iter)

#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.001)
#     for epoch in range(prune_iter):
#         untrained_model = copy.deepcopy(model)
#         ut.train(model, train_loader, criterion, optimizer)
#         unpruned_layers = ut.oneshot_reinit_pruning(model, untrained_model, input_shape, output_shape, per_round_prune_ratio)
#         model.update_layers(unpruned_layers)
#     ut.train(model, train_loader, criterion, optimizer)
#     return model

# Plot Accuracy vs Pruning Ratio
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
#     iterative_pruning_model = ut.iterative_pruning(iterative_pruning_model, input_shape = 28*28, output_shape = 10, train_loader = train_loader,fine_tune_loader=fine_tune_loader, prune_ratio = prune_ratio, prune_iter = 5, max_iter = 1)
#     iterative_pruning_accuracies.append(ut.calculate_accuracy(iterative_pruning_model, test_loader))
#     # Iterative pruning ECE
#     iterative_pruning_ece.append(ut.expected_calibration_error(iterative_pruning_model, test_loader,'results/FashionMNIST/iterative/iterative'+str(prune_ratio)+'.png'))

#     # Iterative pruning strategy 2 accuracy
#     # iterative_reinit_pruning_model = copy.deepcopy(untrained_model)
#     # iterative_reinit_pruning_model = iterative_reinit_pruning(iterative_reinit_pruning_model, input_shape = 28*28, output_shape = 10, train_loader = train_loader, prune_ratio = prune_ratio, prune_iter = 5, max_iter = 1)
   
#     # One-shot pruning accuracy
#     one_shot_model = copy.deepcopy(model)
#     unpruned_layers = ut.oneshot_pruning(one_shot_model, input_shape = 28*28, output_shape = 10, prune_ratio = prune_ratio)
#     one_shot_model.update_layers(unpruned_layers)
#     oneshot_pruning_accuracies.append(ut.calculate_accuracy(one_shot_model, test_loader))
#     # One-shot pruning ECE
#     oneshot_pruning_ece.append(ut.expected_calibration_error(one_shot_model, test_loader,'results/FashionMNIST/oneshot/oneshot'+str(prune_ratio)+'.png'))
    
#     # One-shot reinit pruning accuracy
#     one_shot_reinit_model = copy.deepcopy(model)
#     unpruned_layers = ut.oneshot_reinit_pruning(one_shot_reinit_model, untrained_model, input_shape = 28*28, output_shape = 10, prune_ratio = prune_ratio)
#     one_shot_reinit_model.update_layers(unpruned_layers)
#     # Retrain the model
#     ut.train(one_shot_reinit_model, retrain_loader, criterion, optimizer, epochs = 1) 
#     oneshot_reinit_pruning_accuracies.append(ut.calculate_accuracy(one_shot_reinit_model, test_loader))
#     # One-shot pruning ECE
#     oneshot_reinit_pruning_ece.append(ut.expected_calibration_error(one_shot_reinit_model, test_loader, 'results/FashionMNIST/oneshot_reinit/oneshot_reinit'+str(prune_ratio)+'.png'))

#     # One-shot random reinit pruning accuracy
#     torch.manual_seed(19)
#     random_model = MLP()
#     torch.manual_seed(42)
#     one_shot_random_reinit_model = copy.deepcopy(model)
#     unpruned_layers = ut.oneshot_reinit_pruning(one_shot_random_reinit_model, random_model, input_shape = 28*28, output_shape = 10, prune_ratio = prune_ratio)
#     one_shot_random_reinit_model.update_layers(unpruned_layers)
#     # Retrain the model
#     ut.train(one_shot_random_reinit_model, retrain_loader, criterion, optimizer, epochs = 1) 
#     oneshot_random_reinit_pruning_accuracies.append(ut.calculate_accuracy(one_shot_random_reinit_model, test_loader))
#     # One-shot pruning ECE
#     oneshot_random_reinit_pruning_ece.append(ut.expected_calibration_error(one_shot_random_reinit_model, test_loader, 'results/FashionMNIST/oneshotrandom_reinit/oneshotrandom_reinit'+str(prune_ratio)+'.png'))

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
# plt.savefig('results/FashionMNIST/acc_vs_pm.png')

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
# plt.savefig('results/FashionMNIST/ece_vs_pruning_ratio.png')