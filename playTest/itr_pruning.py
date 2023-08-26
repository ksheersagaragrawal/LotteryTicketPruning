import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import copy

# Defining Accuracy
def calculate_accuracy(model, X, y):
    with torch.no_grad():
        outputs = model(X)
        predictions = outputs.argmax(dim=-1)
        accuracy = (predictions == y).sum().item() / len(y)
    return accuracy

# Training loop
def train(model,X_train_tensor, X_val_tensor,y_train_tensor, y_val_tensor, epochs = 1000):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor)
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

# Create a circles dataset
X, y = make_circles(n_samples=1000, factor=0.5, noise=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.LongTensor(y_train)
X_val_tensor = torch.FloatTensor(X_val)
y_val_tensor = torch.LongTensor(y_val)

# Plot data
#plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=10, cmap=plt.cm.RdBu)

# Defining the model
model = nn.Sequential(
    nn.Linear(2, 20),
    nn.ReLU(),
    nn.Linear(20, 18),
    nn.ReLU(),
    nn.Linear(18, 16),
    nn.ReLU(),
    nn.Linear(16, 14),
    nn.ReLU(),
    nn.Linear(14, 2),
    nn.Sigmoid()
)

# Copying the initial wieghts of the model
pre_training_model = copy.deepcopy(model)

# One-shotPrune the model
def structured_oneshot_pruning( post_training_model, pre_training_model, prune_ratio = 0.2, input_shape = 2, output_shape = 2):
    pruned_layers = []  
    layer_index = 0
    for name, param in post_training_model.named_parameters():
        if 'weight' in name:
            if param.shape[0] == output_shape:
                layer = nn.Linear(input_shape, output_shape)
                with torch.no_grad():
                    layer.data = pre_training_model[layer_index].weight.data
                pruned_layers.append(layer)
                pruned_layers.append(nn.Sigmoid())
                continue
            
            sorted_layers = torch.linalg.norm(param.data, ord=1, dim=1).argsort(dim=-1)
            print(sorted_layers)
            layers_not_pruned = sorted(sorted_layers[int(prune_ratio*param.data.shape[0]):])
            print(layers_not_pruned)
            layer_data = pre_training_model[layer_index].weight.data[layers_not_pruned, :] #initialising unpruned neurons with pre-trainied values
            layer = nn.Linear(input_shape, layer_data.shape[0])
            input_shape = layer_data.shape[0]
            with torch.no_grad():
                layer.data = layer_data
            pruned_layers.append(layer)
            pruned_layers.append(nn.ReLU())
            layer_index=layer_index+2	#skipping every alternate relu layer
    return nn.Sequential(*pruned_layers)

# Non-reinitialised One-shot Pruning the model
def nonreinitialised_oneshot_pruning( post_training_model, prune_ratio = 0.2, input_shape = 2, output_shape = 2):
    pruned_layers = []  
    for name, param in post_training_model.named_parameters():
        if 'weight' in name:
            if param.shape[0] == output_shape:
                layer = nn.Linear(input_shape, output_shape)
                with torch.no_grad():
                    layer.data = param.data
                pruned_layers.append(layer)
                pruned_layers.append(nn.Sigmoid())
                continue
            sorted_layers = torch.linalg.norm(param.data, ord=1, dim=1).argsort(dim=-1)
            print(sorted_layers)
            layers_not_pruned = sorted(sorted_layers[int(prune_ratio*param.data.shape[0]):])
            print(layers_not_pruned)   
            layer_data = param.data[layers_not_pruned,:]
            layer = nn.Linear(input_shape, layer_data.shape[0])
            input_shape = layer_data.shape[0]
            with torch.no_grad():
                layer.data = layer_data
            pruned_layers.append(layer)
            pruned_layers.append(nn.ReLU())
    return nn.Sequential(*pruned_layers)

# Iterative Pruning the model
def structured_iterative_pruning( current_model, X_train_tensor, X_val_tensor, y_train_tensor, y_val_tensor, prune_ratio, prune_iter, max_iter = 1000,input_shape = 2, output_shape = 2):
    per_round_prune_ratio = prune_ratio/prune_iter
    if prune_ratio > 0:
        per_round_prune_ratio = 1 - (1 - prune_ratio) ** (1 / prune_iter)
    per_round_max_iter = int(max_iter / prune_iter)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(current_model.parameters(), lr=0.01)

    for epoch in range(max_iter):
        optimizer.zero_grad()
        outputs = current_model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % per_round_max_iter == 0:
            for name, param in current_model.named_parameters():
                if 'weight' in name:
                    if param.shape[0] == output_shape:
                        continue
                    sorted_layers = torch.linalg.norm(param.data, ord=1, dim=1).argsort(dim=-1)
                    num_neurons_to_keep = int(param.shape[0] * (per_round_prune_ratio))
                    pruned_indices = sorted_layers[:num_neurons_to_keep]
                    with torch.no_grad():
                        param.data[pruned_indices, :] = 0.0        

#Training the model
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
train(model,X_train_tensor,X_val_tensor, y_train_tensor, y_val_tensor, epochs = 1000)
torch.save(model.state_dict(), 'trained_model.pickle')
unpruned_accuracy = calculate_accuracy(model, X_val_tensor, y_val_tensor)

# Plot Accuracy vs Pruning Ratio
pruning_ratios = np.linspace(0.1, 0.9, 20)
itr_pruning_accuracies = []
oneshot_pruning_accuracies = []
nonreinitialised_oneshot_pruning_accuracies = []

current_model = copy.deepcopy(pre_training_model)
structured_iterative_pruning(current_model,X_train_tensor,X_val_tensor, y_train_tensor, y_val_tensor, prune_ratio=0.4, prune_iter=4, max_iter=2000, input_shape=2, output_shape=2)
    
for prune_ratio in pruning_ratios:
    current_model = copy.deepcopy(pre_training_model)

    #iterative pruning plotting
    structured_iterative_pruning(current_model,X_train_tensor,X_val_tensor, y_train_tensor, y_val_tensor, prune_ratio=prune_ratio, prune_iter=4, max_iter=2000, input_shape=2, output_shape=2)
    torch.save(model.state_dict(), 'itr_model.pickle')
    accuracy = calculate_accuracy(current_model, X_val_tensor, y_val_tensor)
    itr_pruning_accuracies.append(accuracy)

    # one shot pruning plotting
    # pruned_oneshot_model = structured_oneshot_pruning(model,pre_training_model,prune_ratio = prune_ratio, input_shape = 2, output_shape = 2)
    # train(pruned_oneshot_model,X_train_tensor,X_val_tensor, y_train_tensor, y_val_tensor, epochs = 1000)
    # accuracy = calculate_accuracy(pruned_oneshot_model, X_val_tensor, y_val_tensor)
    # oneshot_pruning_accuracies.append(accuracy)

    # non-reinitialised one shot pruning
    # nonreinitialised_oneshot_pruned_model = nn.Sequential()
    # nonreinitialised_oneshot_pruned_model = nonreinitialised_oneshot_pruning(model, prune_ratio=prune_ratio, input_shape = 2, output_shape = 2)
    # accuracy = calculate_accuracy(nonreinitialised_oneshot_pruned_model, X_val_tensor, y_val_tensor)
    # nonreinitialised_oneshot_pruning_accuracies.append(accuracy)
plt.axhline(y = unpruned_accuracy, color = 'b', linestyle = '--')
plt.plot(pruning_ratios, itr_pruning_accuracies, marker='o')
#plt.plot(pruning_ratios, oneshot_pruning_accuracies, marker='x')
#plt.plot(pruning_ratios, nonreinitialised_oneshot_pruning_accuracies, marker='*')
plt.legend(["No pruning","iterative", "oneshot", "non-reinit one shot"], loc ="lower right")
plt.title("Accuracy vs. Pruning Ratio")
plt.xlabel("Pruning Ratio")
plt.ylabel("Accuracy")
plt.grid()
plt.show()


