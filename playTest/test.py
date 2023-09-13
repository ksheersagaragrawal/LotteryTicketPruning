import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert data to PyTorch tensors
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

# Define the neural network model using nn.Sequential
model = nn.Sequential(
    nn.Linear(4, 10),  # Input: 4 features, Output: 10 neurons
    nn.ReLU(),
    nn.Linear(10, 3),  # Output: 3 classes for Iris dataset
    nn.Softmax(dim=1)  # Add softmax activation for multi-class classification
)

# Initialize the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
epochs = 100
criterion = nn.CrossEntropyLoss()

for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

# Evaluate the model on the test set
model.eval()
with torch.no_grad():
    y_pred = model(X_test).argmax(dim=1).numpy()

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')