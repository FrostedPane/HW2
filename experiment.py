# Import necessary libraries and functions (already present)
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from models import NeuralNetwork
from utils import preprocess_data, save_model
import matplotlib.pyplot as plt

# Define global variables and load dataset
folder_path = './eel4810-dataset/sub01'
all_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.csv')]
data_frames = [pd.read_csv(file) for file in all_files]
combined_data = pd.concat(data_frames, ignore_index=True)
combined_data = combined_data.fillna(0).replace([float('inf'), float('-inf')], 0)
features, labels = preprocess_data(combined_data)
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.1, random_state=42)

# Helper function for weight initialization
def init_weights(model, method):
    if method == "xavier":
        def xavier_init(layer):
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
        model.apply(xavier_init)

# Define available optimizers
optimizers = {
    "Adam": lambda params, lr, wd: torch.optim.Adam(params, lr=lr, weight_decay=wd),
    "SGD": lambda params, lr, wd: torch.optim.SGD(params, lr=lr, weight_decay=wd),
    "RMSProp": lambda params, lr, wd: torch.optim.RMSprop(params, lr=lr, weight_decay=wd),
    "Momentum": lambda params, lr, wd: torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd)
}

# Define the `train_and_evaluate` function
def train_and_evaluate(batch_size, learning_rate, optimizer_name, weight_init, normalize, l2):
    # Normalize data if needed
    scaler = StandardScaler()
    if normalize:
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled, X_test_scaled = X_train, X_test

    # Convert data to PyTorch tensors and create DataLoader for batching
    train_inputs = torch.tensor(X_train_scaled, dtype=torch.float32)
    train_targets = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    test_inputs = torch.tensor(X_test_scaled, dtype=torch.float32)
    test_targets = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    train_dataset = torch.utils.data.TensorDataset(train_inputs, train_targets)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Create model
    model = NeuralNetwork()
    init_weights(model, weight_init)  # Apply weight initialization

    # Define optimizer and loss function
    optimizer = optimizers[optimizer_name](model.parameters(), learning_rate, l2)
    loss_fn = nn.MSELoss()  # Assuming numeric regression problem

    # Training loop
    epochs = 20
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0.0
        for batch_inputs, batch_targets in train_loader:
            optimizer.zero_grad()  # Reset gradients
            predictions = model(batch_inputs)  # Forward pass
            loss = loss_fn(predictions, batch_targets)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights
            epoch_train_loss += loss.item()

        train_losses.append(epoch_train_loss / len(train_loader))  # Average training loss for the epoch

        # Validation phase
        model.eval()
        with torch.no_grad():
            val_predictions = model(test_inputs)  # Forward pass for validation
            val_loss = loss_fn(val_predictions, test_targets).item()
            val_losses.append(val_loss)

        # # Log the epoch results
        # print(f"Epoch {epoch+1}/{epochs} | "
        #       f"Train Loss: {train_losses[-1]:.4f} | "
        #       f"Validation Loss: {val_losses[-1]:.4f}")

    # # Log results for the configuration
    # print(f"Config -> Batch Size: {batch_size}, Learning Rate: {learning_rate}, Optimizer: {optimizer_name}, "
    #       f"Init: {weight_init}, Normalize: {normalize}, L2: {l2}")
    # print(f"Final Train Loss: {train_losses[-1]:.4f}, Final Validation Loss: {val_losses[-1]:.4f}")

    # Return the training and validation losses for later visualization
    return train_losses, val_losses

# Run experiments (already present)
results = []
for batch_size in [16, 64]:
    for learning_rate in [0.001, 0.01]:
        for optimizer_name in optimizers.keys():
            for weight_init in ["random", "xavier"]:
                for normalize in [True, False]:
                    for l2 in [0.0, 0.01]:  # L2 regularization (weight decay)
                        train_losses, val_losses = train_and_evaluate(
                            batch_size, learning_rate, optimizer_name, weight_init, normalize, l2
                        )
                        results.append({
                            "Batch Size": batch_size,
                            "Learning Rate": learning_rate,
                            "Optimizer": optimizer_name,
                            "Initialization": weight_init,
                            "Normalization": normalize,
                            "L2 Regularization": l2,
                            "Training Loss": train_losses,
                            "Validation Loss": val_losses
                        })

# Import necessary libraries
import matplotlib.pyplot as plt

# Import necessary libraries
import matplotlib.pyplot as plt

   
# Visualize results using Matplotlib
def plot_comparative_results_matplotlib(results):
    plt.figure(figsize=(14, 8))  # Larger plot for better clarity

    for result in results:
        # Configuration details for labeling
        config_label = (f"Batch: {result['Batch Size']}, LR: {result['Learning Rate']}, "
                        f"Opt: {result['Optimizer']}, Init: {result['Initialization']}, "
                        f"Norm: {result['Normalization']}, L2: {result['L2 Regularization']}")
        
        # Plot training and validation loss curves
        plt.plot(result["Training Loss"], linestyle='-', label=f"Train | {config_label}")
        plt.plot(result["Validation Loss"], linestyle='--', label=f"Validation | {config_label}")
    
    # Add labels, title, and legend
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Comparative Loss Curves Across Configurations")
    plt.legend(fontsize='small', loc='center left', bbox_to_anchor=(1, 0.5))  # Legend outside the graph
    plt.grid(True)
    plt.tight_layout()

    # Save the graph as a file
    plt.savefig("matplotlib_comparative_loss_curves.png")
    print("Matplotlib comparative plot saved as 'matplotlib_comparative_loss_curves.png'")
    plt.show()

# Call the visualization function
plot_comparative_results_matplotlib(results)

