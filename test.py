import os
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from utils import preprocess_data, load_model

# Load and preprocess test data
folder_path = './eel4810-dataset/sub01'  # Update this path as needed
all_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.csv')]
data_frames = [pd.read_csv(file) for file in all_files]
combined_data = pd.concat(data_frames, ignore_index=True)

features, labels = preprocess_data(combined_data)

# Normalize the data
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Load the trained model
model = load_model('./model.pth')
model.eval()

# Evaluate on test dataset
inputs = torch.tensor(features, dtype=torch.float32)
with torch.no_grad():
    predictions = model(inputs)
    print("Predictions:", predictions)
