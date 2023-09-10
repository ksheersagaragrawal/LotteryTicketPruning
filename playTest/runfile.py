import pruning_methods as pm
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import copy
import shutil, os
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split


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

#Training the model
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
pm.train(model,X_train_tensor,X_val_tensor, y_train_tensor, y_val_tensor, epochs = 100)
torch.save(model.state_dict(), 'trained_model.pickle')
unpruned_accuracy = pm.calculate_accuracy(model, X_val_tensor, y_val_tensor)

# Plot Accuracy vs Pruning Ratio
num_prune_iter = 20
pruning_ratios = np.linspace(0.1, 0.9, num_prune_iter)
itr_pruning_accuracies = []
oneshot_pruning_reinit_accuracies = []
oneshot_pruning_accuracies = []

if os.path.exists('oneshot_pruned_model'):
    shutil.rmtree('oneshot_pruned_model')
os.makedirs('oneshot_pruned_model')

if os.path.exists('oneshot_reinitialised_pruned_model'):
    shutil.rmtree('oneshot_reinitialised_pruned_model')
os.makedirs('oneshot_reinitialised_pruned_model')

if os.path.exists('img'):
    shutil.rmtree('img')
os.makedirs('img')

for prune_ratio in pruning_ratios:
    pre_training_model_cpy = copy.deepcopy(pre_training_model)

    # one-shot pruning
    oneshot_pruned_model = pm.oneshot_pruning(model, input_shape = 2, output_shape = 2, prune_ratio = prune_ratio)
    pm.train(oneshot_pruned_model,X_train_tensor,X_val_tensor, y_train_tensor, y_val_tensor, epochs = 100)
    accuracy = pm.calculate_accuracy(oneshot_pruned_model, X_val_tensor, y_val_tensor)
    oneshot_pruning_accuracies.append(accuracy)
    pm.plot_decision_boundary(oneshot_pruned_model, 'oneshot_pruned_model/'+f'{prune_ratio*10}'[:3] +'.png', X_train, y_train)

    # re-initiliased one-shot pruning, pre_training_model_cpy gets pruned and udpated
    oneshot_reinitialised_pruned_model = pm.oneshot_pruning_reinit(model,pre_training_model_cpy, input_shape = 2, output_shape = 2, prune_ratio = prune_ratio)
    pm.train(oneshot_reinitialised_pruned_model,X_train_tensor,X_val_tensor, y_train_tensor, y_val_tensor, epochs = 100)
    accuracy = pm.calculate_accuracy(oneshot_reinitialised_pruned_model, X_val_tensor, y_val_tensor)
    oneshot_pruning_reinit_accuracies.append(accuracy)
    pm.plot_decision_boundary(oneshot_reinitialised_pruned_model, 'oneshot_reinitialised_pruned_model/'+ f'{prune_ratio*10}'[:3]+'.png', X_train, y_train)

    # iterative pruning
    pm.iterative_pruning(model, X_train_tensor, y_train_tensor, prune_ratio = prune_ratio, prune_iter = 5, max_iter = 100, input_shape = 2, output_shape = 2)
    accuracy = pm.calculate_accuracy(model, X_val_tensor, y_val_tensor)
    itr_pruning_accuracies.append(accuracy)
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
plt.savefig('img/pruning_acc_vs_pm.png')
# plt.show()

import glob
from PIL import Image


def make_gif(frame_folder, filename):
    frames = [Image.open(image) for image in sorted(glob.glob(f"{frame_folder}/*.png"))]
    frame_one = frames[0]
    frame_one.save(f"{filename}.gif", format="GIF", append_images=frames,
               save_all=True, duration=400, loop=1)
    

make_gif("oneshot_pruned_model", "img/oneshot_pruned_model")
make_gif("oneshot_reinitialised_pruned_model", "img/oneshot_reinitialised_pruned_model")