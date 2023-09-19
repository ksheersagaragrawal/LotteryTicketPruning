import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
from sklearn.metrics import accuracy_score

# Defining Accuracy
def calculate_accuracy(model, X, y):
    with torch.no_grad():
        outputs = model(X)
        predictions = outputs.argmax(dim=1).numpy()
        accuracy = accuracy_score(y, predictions)
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


def expected_calibration_error_multi(samples, true_labels, M=3):
    # uniform binning approach with M number of bins
    bin_boundaries = np.linspace(0, 1, M + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    confidences = np.max(samples, axis=1)               
    # get predictions from confidences (positional in this case)
    predicted_label = np.argmax(samples, axis=1).astype(float)

    # get a boolean list of correct/false predictions
    accuracies = predicted_label==true_labels

    ece = np.zeros(1)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # determine if sample is in bin m (between bin lower & upper)
        in_bin = np.logical_and(confidences > bin_lower.item(), confidences <= bin_upper.item())
        # can calculate the empirical probability of a sample falling into bin m: (|Bm|/n)
        prop_in_bin = in_bin.astype(float).mean()

        if prop_in_bin.item() > 0:
            # get the accuracy of bin m: acc(Bm)
            accuracy_in_bin = accuracies[in_bin].astype(float).mean()
            # get the average confidence of bin m: conf(Bm)
            avg_confidence_in_bin = confidences[in_bin].mean()
            # calculate |acc(Bm) - conf(Bm)| * (|Bm|/n) for bin m and add to the total ECE
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece

def get_data(weight_data, layer_shape, layers_pruned):
    return weight_data[:, [col for col in range(layer_shape[1]) if col not in layers_pruned]]

def add_layer(unpruned_layers, input_shape, output_shape, layer_wt_data, layer_bias_data, activation=nn.ReLU()):
    layer = nn.Linear(input_shape, output_shape)
    with torch.no_grad():
        layer.weight.data = layer_wt_data
        layer.bias.data = layer_bias_data
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
                add_layer(unpruned_layers, input_shape, output_shape, get_data(post_training_model[layer_index].weight.data, param.data.shape, layers_pruned),post_training_model[layer_index].bias.data, nn.Softmax(dim=1))
                continue
            # Sorting the features in a layer based on l1 norm
            wt_param_with_skipped_input = get_data(post_training_model[layer_index].weight.data, param.data.shape, layers_pruned)
            bias_param_with_skipped_input = post_training_model[layer_index].bias.data
            sorted_layers = torch.linalg.norm(wt_param_with_skipped_input, ord=1, dim=1).argsort(dim=-1)
            layers_not_pruned = sorted(sorted_layers[int(prune_ratio*wt_param_with_skipped_input.shape[0]):])
            layers_pruned = sorted(sorted_layers[:int(prune_ratio*wt_param_with_skipped_input.shape[0])])
            layers_not_pruned_indices = torch.tensor([tensor.item() for tensor in layers_not_pruned])

            # Initialising unpruned neurons with pre-training values
            layer_wt_data = wt_param_with_skipped_input[layers_not_pruned, :]
            layer_bias_data = bias_param_with_skipped_input[layers_not_pruned_indices]
            add_layer(unpruned_layers, input_shape, layer_wt_data.shape[0], layer_wt_data, layer_bias_data)
            input_shape = layer_wt_data.shape[0]
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
                add_layer(unpruned_layers, input_shape, output_shape, get_data(pre_training_model[layer_index].weight.data, param.data.shape, layers_pruned),pre_training_model[layer_index].bias.data, nn.Softmax(dim=1))
                continue
            # Sorting the features in  a layer based on l1 norm
            wt_param_with_skipped_input = pre_training_model[layer_index].weight.data[:, [col for col in range(param.data.shape[1]) if col not in layers_pruned]]
            bias_param_with_skipped_input = pre_training_model[layer_index].bias.data
            sorted_layers = torch.linalg.norm(wt_param_with_skipped_input, ord=1, dim=1).argsort(dim=-1)
            layers_not_pruned = sorted(sorted_layers[int(prune_ratio*wt_param_with_skipped_input.shape[0]):])
            layers_pruned = sorted(sorted_layers[:int(prune_ratio*wt_param_with_skipped_input.shape[0])])
            layers_not_pruned_indices = torch.tensor([tensor.item() for tensor in layers_not_pruned])

            # Initialising unpruned neurons with pre-training values
            layer_wt_data = wt_param_with_skipped_input[layers_not_pruned, :]
            layer_bias_data = bias_param_with_skipped_input[layers_not_pruned_indices]
            add_layer(unpruned_layers, input_shape, layer_wt_data.shape[0], layer_wt_data, layer_bias_data)
            input_shape = layer_wt_data.shape[0]
            #skipping every alternate relu layer
            layer_index=layer_index+2
    model = nn.Sequential(*unpruned_layers)
    index = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            model[index].weight.data = unpruned_layers[index].weight.data
            model[index].bias.data = unpruned_layers[index].bias.data
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
                            layer.weight.data = model[layer_index].weight.data[:, [col for col in range(param.data.shape[1]) if col not in layers_pruned]]
                            layer.bias.data = model[layer_index].bias.data
                        unpruned_layers.append(layer)
                        continue
                    # Sorting the features in a layer based on l1 norm
                    wt_param_with_skipped_input = model[layer_index].weight.data[:, [col for col in range(param.data.shape[1]) if col not in layers_pruned]]
                    bias_param_with_skipped_input = model[layer_index].bias.data
                    sorted_layers = torch.linalg.norm(wt_param_with_skipped_input, ord=1, dim=1).argsort(dim=-1)
                    layers_not_pruned = sorted(sorted_layers[int(per_round_prune_ratio*wt_param_with_skipped_input.shape[0]):])
                    layers_pruned = sorted(sorted_layers[:int(per_round_prune_ratio*wt_param_with_skipped_input.shape[0])])
                    layers_not_pruned_indices = torch.tensor([tensor.item() for tensor in layers_not_pruned])

                    # Initialising unpruned neurons with pre-training values
                    layer_wt_data = wt_param_with_skipped_input[layers_not_pruned, :]
                    layer_bias_data = bias_param_with_skipped_input[layers_not_pruned_indices] 
                    layer = nn.Linear(input_shape, layer_wt_data.shape[0])
                    input_shape = layer_wt_data.shape[0]
                    with torch.no_grad():
                        layer.weight.data = layer_wt_data
                        layer.bias.data = layer_bias_data
                    unpruned_layers.append(layer)
                    unpruned_layers.append(nn.ReLU())
                    #skipping every alternate relu layer
                    layer_index=layer_index+2
            pruned_model = nn.Sequential(*unpruned_layers)
            index = 0
            for name, param in pruned_model.named_parameters():
                if 'weight' in name:
                    pruned_model[index].weight.data = unpruned_layers[index].weight.data
                    pruned_model[index].bias.data = unpruned_layers[index].bias.data
                    index=index+2
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(pruned_model.parameters(), lr=0.01)

            return pruned_model
