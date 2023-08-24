import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import copy

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
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=10, cmap=plt.cm.RdBu)

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

#Training the model
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
train(model,X_train_tensor,X_val_tensor, y_train_tensor, y_val_tensor, epochs = 1000)
torch.save(model.state_dict(), 'model.pickle')

#Print the accuracy 
y_val_outputs = model(X_val_tensor).argmax(dim=-1)
print((y_val_outputs == y_val_tensor).sum().item() / len(y_val_tensor))

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

#Training the model after pruning
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
pruned_model = nn.Sequential()
pruned_oneshot_model = structured_oneshot_pruning(model,pre_training_model,prune_ratio = 0.2, input_shape = 2, output_shape = 2)
train(pruned_oneshot_model,X_train_tensor,X_val_tensor, y_train_tensor, y_val_tensor, epochs = 1000)

#Print the accuracy 
y_val_outputs = pruned_oneshot_model(X_val_tensor).argmax(dim=-1)
print((y_val_outputs == y_val_tensor).sum().item() / len(y_val_tensor))

# Iterative Pruning the model
def structured_iterative_pruning( model, X_train_tensor, y_train_tensor, prune_ratio = 0.5, max_iter = 1000, prune_iter = 3,input_shape = 2, output_shape = 2):
    per_round_prune_ratio = prune_ratio/prune_iter
    if prune_ratio > 0:
        per_round_prune_ratio = 1 - (1 - prune_ratio) ** (
            1 / prune_iter
        )
    
    per_round_prune_ratios = [per_round_prune_ratio] * len(model)
    per_round_prune_ratios[-1] /= 2
  
    per_round_max_iter = int(max_iter / prune_iter)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(max_iter):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % per_round_max_iter == 0:
            layer_index = 0
            Input_shape = input_shape
            Output_shape = output_shape
            for name, param in model.named_parameters():
                if 'weight' in name:
                    if param.shape[0] == Output_shape:
                        layer = nn.Linear(Input_shape, Output_shape)
                        with torch.no_grad():
                            layer.data = model[layer_index].weight.data
                        model[layer_index] = layer
                        continue
                    sorted_layers = torch.linalg.norm(param.data, ord=1, dim=1).argsort(dim=-1)
                    layers_not_pruned = sorted(sorted_layers[int(per_round_prune_ratio*param.data.shape[0]):])
                    layer_data = pre_training_model[layer_index].weight.data[layers_not_pruned, :] #initialising unpruned neurons with pre-trainied values
                    layer = nn.Linear(Input_shape, layer_data.shape[0])
                    Input_shape = layer_data.shape[0]
                    with torch.no_grad():
                        layer.data = layer_data
                    model[layer_index] = layer
                    layer_index=layer_index+2	#skipping every alternate relu layer
    torch.save(model.state_dict(), 'model.st_itr_pruned')            

#Iterative pruning & Printing the accuracy 
structured_iterative_pruning(pre_training_model, X_train_tensor, y_train_tensor, prune_ratio = 0.2, max_iter = 1000, prune_iter = 5,input_shape = 2, output_shape = 2)
train(pre_training_model,X_train_tensor,X_val_tensor, y_train_tensor, y_val_tensor, epochs = 1000)
y_val_outputs = pre_training_model(X_val_tensor).argmax(dim=-1)
print((y_val_outputs == y_val_tensor).sum().item() / len(y_val_tensor))
