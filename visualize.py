import matplotlib.pyplot as plt

# Plot losses
def plot_losses(training_loss, validation_loss):
    epochs = range(1, len(training_loss) + 1)
    plt.plot(epochs, training_loss, label='Training Loss')
    plt.plot(epochs, validation_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
