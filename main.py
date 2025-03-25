import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from utils import preprocess_data, save_model
from models import NeuralNetwork  # Import the NeuralNetwork class

# Load and combine CSV files
folder_path = './eel4810-dataset/sub01'  # Update this path as needed
all_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.csv')]
data_frames = [pd.read_csv(file) for file in all_files]
combined_data = pd.concat(data_frames, ignore_index=True)

# Debug: Check combined data
print("Combined Data Shape Before Cleaning:", combined_data.shape)

# Clean the data
combined_data = combined_data.fillna(0)  # Replace NaN with 0
combined_data = combined_data.replace([float('inf'), float('-inf')], 0)  # Replace infinite values with 0

# Ensure all data is numeric
combined_data = combined_data.apply(pd.to_numeric, errors='coerce')  # Convert to numeric
combined_data = combined_data.fillna(0)  # Replace NaN with 0

# Debug: Check dataset structure
print("Sample Data:")
print(combined_data.head())
print("Combined Data Shape After Cleaning:", combined_data.shape)

# Extract features and labels
features = combined_data.iloc[:, 1:4].values  # Adjust column indices if necessary
labels = combined_data.iloc[:, 4].values

# Debug: Check features and labels
print("Features Shape:", features.shape)
print("Labels Shape:", labels.shape)

# Handle small datasets
if features.shape[0] < 10:
    raise ValueError("Dataset is too small to split into training and testing sets.")

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.1, random_state=42)

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model initialization
model = NeuralNetwork()

# Apply weight initialization
def init_weights(layer):
    if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight)

model.apply(init_weights)

# Optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.01)  # Reduced learning rate, with L2 regularization
loss_fn = nn.MSELoss()  # Assuming labels are numeric

# Training loop
epochs = 50
training_loss = []
validation_loss = []

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    # Convert training data to tensors
    inputs = torch.tensor(X_train, dtype=torch.float32)
    targets = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

    # Forward pass
    predictions = model(inputs)
    loss = loss_fn(predictions, targets)

    # Backward pass and optimization
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
    optimizer.step()

    training_loss.append(loss.item())  # Record training loss

    # Validation
    model.eval()
    with torch.no_grad():
        val_inputs = torch.tensor(X_test, dtype=torch.float32)
        val_targets = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
        val_predictions = model(val_inputs)
        val_loss = loss_fn(val_predictions, val_targets)
        validation_loss.append(val_loss.item())

    print(f"Epoch {epoch+1}/{epochs}, Training Loss: {loss.item():.4f}, Validation Loss: {val_loss.item():.4f}")

# Save the model
save_model(model, './model.pth')

# Plot losses
plt.plot(range(epochs), training_loss, label='Training Loss')
plt.plot(range(epochs), validation_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
