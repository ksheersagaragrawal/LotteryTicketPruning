import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn.functional as F
import copy

# Defining Accuracy
def calculate_accuracy(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return (100 * correct / total) 

# Training loop
def train(model, train_loader, criterion, optimizer, epochs=1, bayesian=False):
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            if bayesian:
                output = torch.log_softmax(model(data, bayesian=bayesian), dim=2).mean(dim=0)
                loss = criterion(output, target)
            else:
                output = model(data, bayesian=bayesian)
                loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item()}")

def get_data(weight_data, layers_pruned):
    return weight_data[:, [col for col in range(weight_data.shape[1]) if col not in layers_pruned]]

def add_layer(unpruned_layers, input_shape, output_shape, layer_wt_data, layer_bias_data):
    layer = nn.Linear(input_shape, output_shape)
    with torch.no_grad():
        layer.weight.data = layer_wt_data
        layer.bias.data = layer_bias_data
    unpruned_layers.append(layer)

def expected_calibration_error(model, test_loader, title, M=10):
    # uniform binning approach with M number of bins
    bin_boundaries = np.linspace(0, 1, M + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    predicted_labels = []
    true_labels = []
    confidences = []
    reliabilities = []

   # keep confidences / predicted "probabilities" as they are
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            output = F.softmax(output, dim=1)
            _, predicted = torch.max(output.data, 1)
            predicted_labels.append(predicted)
            true_labels.append(target)
            confidences.append(_)

    # get binary class predictions from confidences
    predicted_labels = torch.cat(predicted_labels).numpy()
    true_labels = torch.cat(true_labels).numpy()
    confidences = torch.cat(confidences).numpy()

    # get a boolean list of correct/false predictions
    accuracies = predicted_labels==true_labels   

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
            bin_accuracy = np.mean(true_labels[in_bin]==predicted_labels[in_bin])
            reliabilities.append(bin_accuracy)
        else:
            reliabilities.append(0.0)  # Avoid division by zero

    # Plot the reliability diagram
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")  # Diagonal reference line
    plt.plot(np.linspace(0, 1, M), reliabilities, marker="o", linestyle="-", color="blue")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Empirical Accuracy")
    plt.title(f"Reliability Diagram (ECE = {ece[0]:.4f})")
    plt.grid(True)
    plt.savefig(title)
    return ece

# In this method, we prune prune_ratio fatures in each layer
# The nn.Sequential method randomly initialises when called
def oneshot_pruning( model, input_shape, output_shape, prune_ratio = 0.2):
    unpruned_layers = [] 
    layers_pruned = []
    layer_index = 0
    for name, module in model.named_children():
        
        if isinstance(module, nn.Linear):
            layer_index += 1
            # Not pruning the last output layer if param.shape[0] == output_shape:
            if layer_index==5:
                add_layer(unpruned_layers, input_shape, output_shape, get_data(module.weight, layers_pruned),module.bias)
                continue
            # Sorting the features in a layer based on l1 norm
            weight_with_skipped_input = get_data(module.weight, layers_pruned)
            bias_param_with_skipped_input = module.bias.data
            sorted_layers = torch.linalg.norm(weight_with_skipped_input, ord=1, dim=1).argsort(dim=-1)
            layers_not_pruned = sorted(sorted_layers[int(prune_ratio*weight_with_skipped_input.shape[0]):])
            layers_pruned = sorted(sorted_layers[:int(prune_ratio*weight_with_skipped_input.shape[0])])
            layers_not_pruned_indices = torch.tensor([tensor.item() for tensor in layers_not_pruned])
            
            # Initialising unpruned neurons with pre-training values
            layer_wt_data = weight_with_skipped_input[layers_not_pruned, :]
            layer_bias_data = bias_param_with_skipped_input[layers_not_pruned_indices]
            add_layer(unpruned_layers, input_shape, layer_wt_data.shape[0], layer_wt_data, layer_bias_data)
            input_shape = layer_wt_data.shape[0]
    return unpruned_layers

def oneshot_reinit_pruning( model, untrained_model, input_shape, output_shape, prune_ratio = 0.2):
    unpruned_layers = [] 
    layers_pruned = []
    layer_index = 0
    for (name, module), (_, untrained_module) in zip(model.named_children(), untrained_model.named_children()):
        if isinstance(module, nn.Linear):
            layer_index += 1
            # Not pruning the last output layer if param.shape[0] == output_shape:
            if layer_index==5:
                add_layer(unpruned_layers, input_shape, output_shape, get_data(module.weight, layers_pruned),module.bias)
                continue
            # Sorting the features in a layer based on l1 norm
            trained_weight_with_skipped_input = get_data(module.weight, layers_pruned)
            weight_param_skipped_input = get_data(untrained_module.weight, layers_pruned)
            bias_param_with_skipped_input = untrained_module.bias.data
            sorted_layers = torch.linalg.norm(trained_weight_with_skipped_input, ord=1, dim=1).argsort(dim=-1)
            layers_not_pruned = sorted(sorted_layers[int(prune_ratio*trained_weight_with_skipped_input.shape[0]):])
            layers_pruned = sorted(sorted_layers[:int(prune_ratio*trained_weight_with_skipped_input.shape[0])])
            layers_not_pruned_indices = torch.tensor([tensor.item() for tensor in layers_not_pruned])
            
            # Initialising unpruned neurons with pre-training values
            layer_wt_data = weight_param_skipped_input[layers_not_pruned, :]
            layer_bias_data = bias_param_with_skipped_input[layers_not_pruned_indices]
            add_layer(unpruned_layers, input_shape, layer_wt_data.shape[0], layer_wt_data, layer_bias_data)
            input_shape = layer_wt_data.shape[0]
    return unpruned_layers

# Iterative Pruning 
# We prune pm features in each layer based on L1 norm, and remove pruned layers outgoing edge
# The nn.Sequential method randomly initialises when called 
# So we copy the weights of the model before pruning using indexing
# The accuracy drops after pruning so we fine tune the model
def iterative_pruning( model, input_shape, output_shape, train_loader,fine_tune_loader, prune_ratio, prune_iter, max_iter = 1):
    
    # Per round pune ratio and number of epochs for every fine tuning
    per_round_prune_ratio = prune_ratio/prune_iter
    if prune_ratio > 0:
        per_round_prune_ratio = 1 - (1 - prune_ratio) ** (1 / prune_iter)
    per_round_max_iter = int(938 / prune_iter)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(max_iter):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if (batch_idx + 1) % per_round_max_iter == 0:
                unpruned_layers = [] 
                layers_pruned = []
                layer_index = 0
                for name, module in model.named_children():
    
                    if isinstance(module, nn.Linear):
                        layer_index+=1
                        # Not pruning the last output layer if param.shape[0] == output_shape:
                        if layer_index==5:
                            add_layer(unpruned_layers, input_shape, output_shape, get_data(module.weight, layers_pruned),module.bias)
                            continue
                        # Sorting the features in a layer based on l1 norm
                        weight_with_skipped_input = get_data(module.weight, layers_pruned)
                        bias_param_with_skipped_input = module.bias.data
                        sorted_layers = torch.linalg.norm(weight_with_skipped_input, ord=1, dim=1).argsort(dim=-1)
                        layers_not_pruned = sorted(sorted_layers[int(per_round_prune_ratio*weight_with_skipped_input.shape[0]):])
                        layers_pruned = sorted(sorted_layers[:int(per_round_prune_ratio*weight_with_skipped_input.shape[0])])
                        layers_not_pruned_indices = torch.tensor([tensor.item() for tensor in layers_not_pruned])
                        
                        # Initialising unpruned neurons with pre-training values
                        layer_wt_data = weight_with_skipped_input[layers_not_pruned, :]
                        layer_bias_data = bias_param_with_skipped_input[layers_not_pruned_indices]
                        
                        add_layer(unpruned_layers, input_shape, layer_wt_data.shape[0], layer_wt_data, layer_bias_data)
                        input_shape = layer_wt_data.shape[0]
                model.update_layers(unpruned_layers)
                train(model, fine_tune_loader, criterion, optimizer, epochs = 1) 
    return model

def plot_reliability_diagrams(acc2, acc1, err1, ece2, ece1, uce1, save_path):
    # Create a single figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot the reliability diagram for acc2
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray")  # Diagonal reference line
    axes[0].plot(np.linspace(0, 1, len(acc2)), acc2, marker="o", linestyle="-", color="blue")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Frequentist Confidence")
    axes[0].grid(True)

    # Add ece2 as a label to the plot
    axes[0].text(0.5, 0.1, f"ECE: {str(ece2)}", color="red")

    # Plot the reliability diagram for err1
    axes[1].plot([0, 1], [0, 1], linestyle="--", color="gray")  # Diagonal reference line
    axes[1].plot(np.linspace(0, 1, len(acc1)), acc1, marker="o", linestyle="-", color="blue")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("MC Dropout Confidence") 
    axes[1].grid(True)

    # Add ece1 as a label to the plot
    axes[1].text(0.5, 0.1, f"ECE: {str(ece1)}", color="red")

    # Plot the reliability diagram for acc1
    axes[2].plot([0, 1], [0, 1], linestyle="--", color="gray")  # Diagonal reference line
    axes[2].plot(np.linspace(0, 1, len(err1)), err1, marker="o", linestyle="-", color="blue")
    axes[2].set_ylabel("Error")
    axes[2].set_title("MC Dropout Uncertainty")
    axes[2].grid(True)

    # Add uce1 as a label to the plot
    axes[2].text(0.5, 0.1, f"UCE: {str(uce1)}", color="red")

    # Save the entire figure
    plt.savefig(save_path)

