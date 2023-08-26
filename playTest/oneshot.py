import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch.nn.init as init
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
def train(model,X_train_tensor, X_val_tensor,y_train_tensor, y_val_tensor, epochs = 100):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
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

# In this method, we prune prune_ratio fatures in each layer
# The nn.Sequential method randomly initialises when called
def oneshot_pruning( post_training_model, input_shape, output_shape, prune_ratio = 0.2):
    unpruned_layers = [] 
    layer_index = 0
    layers_pruned = []
    for name, param in post_training_model.named_parameters():
        if 'weight' in name:
            # Not pruning the last output layer
            if param.shape[0] == output_shape:
                layer = nn.Linear(input_shape, output_shape)
                with torch.no_grad():
                    layer.data = post_training_model[layer_index].weight.data[:, [col for col in range(param.data.shape[1]) if col not in layers_pruned]]
                unpruned_layers.append(layer)
                unpruned_layers.append(nn.Sigmoid())
                continue
            # Sorting the features in a layer based on l1 norm
            param_with_skipped_input = post_training_model[layer_index].weight.data[:, [col for col in range(param.data.shape[1]) if col not in layers_pruned]]
            sorted_layers = torch.linalg.norm(param_with_skipped_input, ord=1, dim=1).argsort(dim=-1)
            layers_not_pruned = sorted(sorted_layers[int(prune_ratio*param_with_skipped_input.shape[0]):])
            layers_pruned = sorted(sorted_layers[:int(prune_ratio*param_with_skipped_input.shape[0])])

            # Initialising unpruned neurons with pre-training values
            layer_data = param_with_skipped_input[layers_not_pruned, :] 
            layer = nn.Linear(input_shape, layer_data.shape[0])
            input_shape = layer_data.shape[0]
            with torch.no_grad():
                layer.data = layer_data
            unpruned_layers.append(layer)
            unpruned_layers.append(nn.ReLU())
            #skipping every alternate relu layer
            layer_index=layer_index+2
    return nn.Sequential(*unpruned_layers)  

# In this method of one-shot structured re-initialsed pruning,
# We prune pm fatures in each layer based on L1 norm, and remove pruned layers outgoing edge
# The nn.Sequential method randomly initialises when called 
# So we copy the initial values of pre_training_model using indexing
def oneshot_pruning_reinit( post_training_model, pre_training_model, input_shape, output_shape, prune_ratio = 0.2):
    unpruned_layers = [] 
    layer_index = 0
    layers_pruned = []
    for name, param in post_training_model.named_parameters():
        if 'weight' in name:
            # Not pruning the last output layer
            if param.shape[0] == output_shape:
                layer = nn.Linear(input_shape, output_shape)
                with torch.no_grad():
                    layer.data = pre_training_model[layer_index].weight.data[:, [col for col in range(param.data.shape[1]) if col not in layers_pruned]]
                unpruned_layers.append(layer)
                unpruned_layers.append(nn.Sigmoid())
                continue
            # Sorting the features in a layer based on l1 norm
            param_with_skipped_input = pre_training_model[layer_index].weight.data[:, [col for col in range(param.data.shape[1]) if col not in layers_pruned]]
            sorted_layers = torch.linalg.norm(param_with_skipped_input, ord=1, dim=1).argsort(dim=-1)
            layers_not_pruned = sorted(sorted_layers[int(prune_ratio*param_with_skipped_input.shape[0]):])
            layers_pruned = sorted(sorted_layers[:int(prune_ratio*param_with_skipped_input.shape[0])])

            # Initialising unpruned neurons with pre-training values
            layer_data = param_with_skipped_input[layers_not_pruned, :] 
            layer = nn.Linear(input_shape, layer_data.shape[0])
            input_shape = layer_data.shape[0]
            with torch.no_grad():
                layer.data = layer_data
            unpruned_layers.append(layer)
            unpruned_layers.append(nn.ReLU())
            #skipping every alternate relu layer
            layer_index=layer_index+2
    model = nn.Sequential(*unpruned_layers)
    index = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            param.data = unpruned_layers[index].data
            index=index+2
    return model
            

#Training the model
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
train(model,X_train_tensor,X_val_tensor, y_train_tensor, y_val_tensor, epochs = 100)
torch.save(model.state_dict(), 'trained_model.pickle')
unpruned_accuracy = calculate_accuracy(model, X_val_tensor, y_val_tensor)

# Plot Accuracy vs Pruning Ratio
pruning_ratios = np.linspace(0.1, 0.9, 20)
itr_pruning_accuracies = []
oneshot_pruning_reinit_accuracies = []
oneshot_pruning_accuracies = []
nonreinitialised_oneshot_pruning_accuracies = []

for prune_ratio in pruning_ratios:
    pre_training_model_cpy = copy.deepcopy(pre_training_model)

    # one-shot pruning
    oneshot_pruned_model = oneshot_pruning(model, input_shape = 2, output_shape = 2, prune_ratio = prune_ratio)
    train(oneshot_pruned_model,X_train_tensor,X_val_tensor, y_train_tensor, y_val_tensor, epochs = 100)
    accuracy = calculate_accuracy(oneshot_pruned_model, X_val_tensor, y_val_tensor)
    oneshot_pruning_accuracies.append(accuracy)

    # re-initiliased one-shot pruning, pre_training_model_cpy gets pruned and udpated
    oneshot_reinitialised_pruned_model = oneshot_pruning_reinit(model,pre_training_model_cpy, input_shape = 2, output_shape = 2, prune_ratio = prune_ratio)
    train(oneshot_reinitialised_pruned_model,X_train_tensor,X_val_tensor, y_train_tensor, y_val_tensor, epochs = 100)
    accuracy = calculate_accuracy(oneshot_reinitialised_pruned_model, X_val_tensor, y_val_tensor)
    oneshot_pruning_reinit_accuracies.append(accuracy)
    
plt.axhline(y = unpruned_accuracy, color = 'b', linestyle = '--')
plt.plot(pruning_ratios, oneshot_pruning_accuracies, marker='x')
plt.plot(pruning_ratios, oneshot_pruning_reinit_accuracies, marker='o')
plt.legend(["No pruning","One-shot","Re-init One-shot"], loc ="lower right")
plt.title("Accuracy vs. Pruning Ratio")
plt.xlabel("Pruning Ratio")
plt.ylabel("Accuracy")
plt.grid()
plt.savefig('oneshot_acc_vs_pm.png')
plt.show()


