import pruning_methods as pm
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import copy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import zipfile
import pandas as pd

# Load the dataset from 'iris.data' file
column_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
iris_data = pd.read_csv("iris.data", names=column_names)

# Map class labels to integers
class_mapping = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
iris_data["class"] = iris_data["class"].map(class_mapping)

# Split the dataset into features (X) and labels (y)
X = iris_data.iloc[:, :-1].values
y = iris_data.iloc[:, -1].values

# Split the data into training and testing sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Convert data to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
X_val_tensor  = torch.FloatTensor(X_val)
y_train_tensor  = torch.LongTensor(y_train)
y_val_tensor  = torch.LongTensor(y_val)

# Define the neural network model using nn.Sequential
model = nn.Sequential(
    nn.Linear(4, 20),  # Input: 4 features, Output: 10 neurons
    nn.ReLU(),
    nn.Linear(20, 3),  # Output: 3 classes for Iris dataset
    nn.Softmax(dim=1)
)

# Copying the initial wieghts of the model
pre_training_model = copy.deepcopy(model)

#Training the model
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
pm.train(model,X_train_tensor,X_val_tensor, y_train_tensor, y_val_tensor, epochs = 100)
unpruned_accuracy = pm.calculate_accuracy(model, X_val_tensor, y_val_tensor)
ece_unpruned = pm.expected_calibration_error_multi(model(X_train_tensor).detach().numpy(), y_train_tensor.detach().numpy(), 10)

# Plot Accuracy vs Pruning Ratio
num_prune_iter = 20
pruning_ratios = np.linspace(0.1, 0.9, num_prune_iter)
itr_pruning_accuracies = []
oneshot_pruning_reinit_accuracies = []
oneshot_pruning_accuracies = []
ece = []

for prune_ratio in pruning_ratios:
    pre_training_model_cpy = copy.deepcopy(pre_training_model)
    ece.append([])

    # one-shot pruning
    oneshot_pruned_model = pm.oneshot_pruning(model, input_shape =  4, output_shape = 3, prune_ratio = prune_ratio)
    pm.train(oneshot_pruned_model,X_train_tensor,X_val_tensor, y_train_tensor, y_val_tensor, epochs = 100)
    accuracy = pm.calculate_accuracy(oneshot_pruned_model, X_val_tensor, y_val_tensor)
    oneshot_pruning_accuracies.append(accuracy)
    ece[-1].append(pm.expected_calibration_error_multi(oneshot_pruned_model(X_train_tensor).detach().numpy(), y_train_tensor.detach().numpy(), 10))
    #pm.plot_decision_boundary(oneshot_pruned_model, 'oneshot_pruned_model/'+f'{prune_ratio*10}'[:3] +'.png', X_train, y_train)

    # re-initiliased one-shot pruning, pre_training_model_cpy gets pruned and udpated
    oneshot_reinitialised_pruned_model = pm.oneshot_pruning_reinit(model,pre_training_model_cpy, input_shape = 4, output_shape = 3, prune_ratio = prune_ratio)
    pm.train(oneshot_reinitialised_pruned_model,X_train_tensor,X_val_tensor, y_train_tensor, y_val_tensor, epochs = 100)
    accuracy = pm.calculate_accuracy(oneshot_reinitialised_pruned_model, X_val_tensor, y_val_tensor)
    oneshot_pruning_reinit_accuracies.append(accuracy)
    ece[-1].append(pm.expected_calibration_error_multi(oneshot_reinitialised_pruned_model(X_train_tensor).detach().numpy(), y_train_tensor.detach().numpy(), 10))
    #pm.plot_decision_boundary(oneshot_reinitialised_pruned_model, 'oneshot_reinitialised_pruned_model/'+ f'{prune_ratio*10}'[:3]+'.png', X_train, y_train)

    # iterative pruning
    itr_pruned_model = pm.iterative_pruning(model, X_train_tensor, y_train_tensor, prune_ratio = prune_ratio, prune_iter = 5, max_iter = 100, input_shape = 4, output_shape = 3)
    accuracy = pm.calculate_accuracy(model, X_val_tensor, y_val_tensor)
    itr_pruning_accuracies.append(accuracy)
    ece[-1].append(pm.expected_calibration_error_multi(itr_pruned_model(X_train_tensor).detach().numpy(), y_train_tensor.detach().numpy(), 10))
    # plot_decision_boundary(oneshot_reinitialised_pruned_model, 'oneshot_reinitialised_pruned_model/'+ f'{prune_ratio*10}'[:3]+'.png')

plt.figure().clear()
plt.axhline(y = unpruned_accuracy, color = 'b', linestyle = '--')
plt.plot(pruning_ratios, oneshot_pruning_accuracies, marker='x')
plt.plot(pruning_ratios, oneshot_pruning_reinit_accuracies, marker='o')
plt.plot(pruning_ratios, itr_pruning_accuracies, marker='*')
plt.legend(["No pruning","One-shot","Re-init One-shot","Iterative"], loc ="lower right")
plt.title("Accuracy vs. Pruning Ratio")
plt.xlabel("Pruning Ratio")
plt.ylabel("Accuracy")
plt.grid()
plt.savefig('img/iris_pruning_acc_vs_pm.png')

plt.figure().clear()
plt.plot(pruning_ratios, [ece[i][0] for i in range(len(ece))], marker='x')
plt.plot(pruning_ratios, [ece[i][1] for i in range(len(ece))], marker='o')
plt.plot(pruning_ratios, [ece[i][2] for i in range(len(ece))], marker='s')
plt.axhline(y = ece_unpruned, color = 'b', linestyle = '--')
plt.title("ECE vs. Pruning Ratio")
plt.legend(["One-shot","Re-init One-shot", "Iterative", "Unpruned"], loc ="upper left")
plt.xlabel("Pruning Ratio")
plt.ylabel("ECE")
plt.grid()
plt.savefig('img/iris_pruning_ece_vs_pm.png')
