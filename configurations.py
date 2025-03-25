from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch.optim as optim

# Example normalization function
def apply_normalization(data):
    scaler = StandardScaler()
    return scaler.fit_transform(data)

# Optimizer selection
def get_optimizer(model, optimizer_type, lr, weight_decay=0.01):
    if optimizer_type == "Adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == "SGD":
        return optim.SGD(model.parameters(), lr=lr)
    elif optimizer_type == "RMSProp":
        return optim.RMSprop(model.parameters(), lr=lr)
    elif optimizer_type == "Momentum":
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9)

# Weight initialization
def init_weights(layer):
    if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight)
