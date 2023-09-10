import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st

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


def reliability_diagram(mean, sigma, Y, color="blue", label="Model", marker_size=6, path="Reliability Diagram.png"):
    fig, ax = plt.subplots()
    df = pd.DataFrame()
    df["mean"] = mean
    df["sigma"] = sigma
    df["Y"] = Y
    df["z"] = (df["Y"] - df["mean"]) / df["sigma"]
    df["perc"] = st.norm.cdf(df["z"])
    print(df["perc"])
    print(df["z"])
    k = np.arange(0, 1.1, 0.1)
    counts = []
    df2 = pd.DataFrame()
    df2["Interval"] = k
    df2["Ideal"] = k
    for i in range(0, 11):
        l = df[df["perc"] < 0.5 + i * 0.05]
        l = l[l["perc"] >= 0.5 - i * 0.05]
        counts.append(len(l) / len(df))
    df2["Counts"] = counts

    ax.plot(k, counts, color=color, label=label)

    ax.scatter(k, counts, color=color,s=marker_size)
    ax.scatter(k, k,color="green",s=marker_size)
    ax.set_yticks(k)
    ax.set_xticks(k)
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    # ax.legend()
    ax.set_xlabel("decile")
    ax.set_ylabel("ratio of points")
    ax.plot(k, k, color="green")
    ax.set_title("Reliability Diagram")
    fig.savefig(path)

# Plot decision boundary
def plot_decision_boundary(model, filename, X_train, y_train):
    x = torch.linspace(-1.5, 1.5, 1000)
    y = torch.linspace(-1.5, 1.5, 1000)
    xv, yv = torch.meshgrid(x, y)

    # Reshape xv and yv
    xv = xv.reshape(-1)
    yv = yv.reshape(-1)

    # Stack xv and yv into a tensor
    grid = torch.stack((xv, yv), dim=1)
    # Feed the grid tensor into the model

    model.eval()
    with torch.no_grad():
        outputs = model(grid)
        
    # Get the predicted class
    _, y_pred = torch.max(outputs, 1)
    
    fig, ax = plt.subplots(figsize=(5, 5))


    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=10, cmap=plt.cm.RdBu)
    im = ax.contourf(x, y, y_pred.reshape(1000, 1000), alpha=0.3, cmap=plt.cm.RdBu)
    ax.set_title(f"Decision Boundary {filename}")
    plt.colorbar(im , ax=ax)
    fig.savefig(filename)

def get_data(weight_data, layer_shape, layers_pruned):
    return weight_data[:, [col for col in range(layer_shape[1]) if col not in layers_pruned]]

def add_layer(unpruned_layers, input_shape, output_shape, layer_data, activation = nn.ReLU()):
	layer = nn.Linear(input_shape, output_shape)
	with torch.no_grad():
		layer.data = layer_data
	unpruned_layers.append(layer)
	unpruned_layers.append(activation)

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
            # if param.shape[0] == output_shape:
            if layer_index == len(post_training_model)-2:
                add_layer(unpruned_layers, input_shape, output_shape, get_data(post_training_model[layer_index].weight.data, param.data.shape, layers_pruned), nn.Sigmoid())
                continue
            # Sorting the features in a layer based on l1 norm
            param_with_skipped_input = get_data(post_training_model[layer_index].weight.data, param.data.shape, layers_pruned)
            sorted_layers = torch.linalg.norm(param_with_skipped_input, ord=1, dim=1).argsort(dim=-1)
            layers_not_pruned = sorted(sorted_layers[int(prune_ratio*param_with_skipped_input.shape[0]):])
            layers_pruned = sorted(sorted_layers[:int(prune_ratio*param_with_skipped_input.shape[0])])

            # Initialising unpruned neurons with pre-training values
            layer_data = param_with_skipped_input[layers_not_pruned, :] 
            add_layer(unpruned_layers, input_shape, layer_data.shape[0], layer_data)
            input_shape = layer_data.shape[0]
            #skipping every alternate relu layer
            layer_index=layer_index+2
    return nn.Sequential(*unpruned_layers)  

# In this method of one-shot structured re-initialsed pruning,
# We prune pm features in each layer based on L1 norm, and remove pruned layers outgoing edge
# The nn.Sequential method randomly initialises when called 
# So we copy the initial values of pre_training_model using indexing
def oneshot_pruning_reinit( post_training_model, pre_training_model, input_shape, output_shape, prune_ratio = 0.2):
    unpruned_layers = [] 
    layer_index = 0
    layers_pruned = []
    for name, param in post_training_model.named_parameters():
        if 'weight' in name:
            # Not pruning the last output layer
            if layer_index == len(post_training_model)-2:
                add_layer(unpruned_layers, input_shape, output_shape, get_data(pre_training_model[layer_index].weight.data, param.data.shape, layers_pruned), nn.Sigmoid())
                continue
            # Sorting the features in a layer based on l1 norm
            param_with_skipped_input = pre_training_model[layer_index].weight.data[:, [col for col in range(param.data.shape[1]) if col not in layers_pruned]]
            sorted_layers = torch.linalg.norm(param_with_skipped_input, ord=1, dim=1).argsort(dim=-1)
            layers_not_pruned = sorted(sorted_layers[int(prune_ratio*param_with_skipped_input.shape[0]):])
            layers_pruned = sorted(sorted_layers[:int(prune_ratio*param_with_skipped_input.shape[0])])

            # Initialising unpruned neurons with pre-training values
            layer_data = param_with_skipped_input[layers_not_pruned, :] 
            add_layer(unpruned_layers, input_shape, layer_data.shape[0], layer_data)
            input_shape = layer_data.shape[0]
            #skipping every alternate relu layer
            layer_index=layer_index+2
    model = nn.Sequential(*unpruned_layers)
    index = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            param.data = unpruned_layers[index].data
            index=index+2
    return model

# Iterative Pruning 
# We prune pm features in each layer based on L1 norm, and remove pruned layers outgoing edge
# The nn.Sequential method randomly initialises when called 
# So we copy the weights of the model before pruning using indexing
# The accuracy drops after pruning so we fine tune the model
def iterative_pruning(model, X_train_tensor, y_train_tensor, prune_ratio, prune_iter, max_iter = 100,input_shape = 2, output_shape = 2):
    
    # Per round pune ratio and number of epochs for every fine tuning
    per_round_prune_ratio = prune_ratio/prune_iter
    if prune_ratio > 0:
        per_round_prune_ratio = 1 - (1 - prune_ratio) ** (1 / prune_iter)
    per_round_max_iter = int(max_iter / prune_iter)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    for epoch in range(max_iter):    
        # Fine tuning the model
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

        # Pruing per_round_prune_ratio of layers every max_iter/prune_iter iteration
        if (epoch + 1) % per_round_max_iter == 0:
            unpruned_layers = [] 
            layer_index = 0
            layers_pruned = []
            for name, param in model.named_parameters():
                if 'weight' in name:
                    # Not pruning the last output layer
                    if param.shape[0] == output_shape:
                        layer = nn.Linear(input_shape, output_shape)
                        with torch.no_grad():
                            layer.data = model[layer_index].weight.data[:, [col for col in range(param.data.shape[1]) if col not in layers_pruned]]
                        unpruned_layers.append(layer)
                        unpruned_layers.append(nn.Sigmoid())
                        continue
                    # Sorting the features in a layer based on l1 norm
                    param_with_skipped_input = model[layer_index].weight.data[:, [col for col in range(param.data.shape[1]) if col not in layers_pruned]]
                    sorted_layers = torch.linalg.norm(param_with_skipped_input, ord=1, dim=1).argsort(dim=-1)
                    layers_not_pruned = sorted(sorted_layers[int(per_round_prune_ratio*param_with_skipped_input.shape[0]):])
                    layers_pruned = sorted(sorted_layers[:int(per_round_prune_ratio*param_with_skipped_input.shape[0])])

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
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.01)