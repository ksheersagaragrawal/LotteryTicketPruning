import torch
import torch.nn as nn

def get_data(weight_data, layers_pruned):
    return weight_data[:, [col for col in range(weight_data.shape[1]) if col not in layers_pruned]]

def add_layer(unpruned_layers, input_shape, output_shape, layer_wt_data, layer_bias_data):
    layer = nn.Linear(input_shape, output_shape)
    with torch.no_grad():
        layer.weight.data = layer_wt_data
        layer.bias.data = layer_bias_data
    unpruned_layers.append(layer)

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
            if layer_index==3:
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