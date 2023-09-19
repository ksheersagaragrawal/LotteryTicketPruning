import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


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
def train(model, train_loader, criterion, optimizer, epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
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


def expected_calibration_error(model, test_loader, M=3):
    # uniform binning approach with M number of bins
    bin_boundaries = np.linspace(0, 1, M + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    predicted_label = []
    true_labels = []
    confidences = []
   # keep confidences / predicted "probabilities" as they are
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            predicted_label.append(predicted)
            true_labels.append(target)
            confidences.append(_)

    # get binary class predictions from confidences
    predicted_label = torch.cat(predicted_label).numpy()
    true_labels = torch.cat(true_labels).numpy()
    confidences = torch.cat(confidences).numpy()
    
    true_labels = (true_labels==predicted_label).astype(int) 
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

# In this method, we prune prune_ratio fatures in each layer
# The nn.Sequential method randomly initialises when called
def oneshot_pruning( model, input_shape, output_shape, prune_ratio = 0.2):
    unpruned_layers = [] 
    layers_pruned = []
    
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            # Not pruning the last output layer if param.shape[0] == output_shape:
            if module.bias.shape[0] == output_shape:
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
    
    for (name, module), (_, untrained_module) in zip(model.named_children(), untrained_model.named_children()):
        if isinstance(module, nn.Linear):
            # Not pruning the last output layer if param.shape[0] == output_shape:
            if module.bias.shape[0] == output_shape:
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
def iterative_pruning( model, input_shape, output_shape, train_loader, prune_ratio, prune_iter, max_iter = 10):
    
     # Per round pune ratio and number of epochs for every fine tuning
    per_round_prune_ratio = prune_ratio/prune_iter
    if prune_ratio > 0:
        per_round_prune_ratio = 1 - (1 - prune_ratio) ** (1 / prune_iter)
    per_round_max_iter = int(max_iter / prune_iter)

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
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}/{max_iter}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item()}")

        # Pruing per_round_prune_ratio of layers every max_iter/prune_iter iteration
        if (epoch + 1) % per_round_max_iter == 0:
            unpruned_layers = [] 
            layers_pruned = []
            
            for name, module in model.named_children():
                if isinstance(module, nn.Linear):
                    # Not pruning the last output layer if param.shape[0] == output_shape:
                    if module.bias.shape[0] == output_shape:
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
    return model