import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.optim import AdamW
from lion import Lion
import matplotlib.pyplot as plt

# Config
BATCH_SIZE = 64
EPOCHS = 5
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load Data
transform = transforms.ToTensor()
train_data = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_data = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

# Simple MLP Model
class MLP(nn.Module):  # Multi-Layer Perceptron
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 256) # Input: 784 pixels → 256 neurons
        self.fc2 = nn.Linear(256, 128) # Hidden: 256 → 128 neurons 
        self.fc3 = nn.Linear(128, 10) # Output: 128 → 10 classes
# This creates a simple feedforward neural network
# Input layer: 784 neurons (28×28 flattened image)
# Hidden layer 1: 256 neurons with ReLU activation
# Hidden layer 2: 128 neurons with ReLU activation
# Output layer: 10 neurons (one per clothing class)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def evaluate(model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            output = model(x)
            preds = output.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total

# Train function
def train_model(optimizer_name="lion"):
    model = MLP().to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    if optimizer_name == "lion":
        optimizer = Lion(model.parameters(), lr=LR)
    else:
        optimizer = AdamW(model.parameters(), lr=LR)

    loss_log = [] #logs total loss over epochs

    for epoch in range(EPOCHS):
        model.train()                                   # Set model to training mode
        total_loss = 0                                  # Loss of this epoch

        for batch in train_loader:
            x, y = batch                                # Get images and labels
            x, y = x.to(DEVICE), y.to(DEVICE)           # move to GPU/CPU

            optimizer.zero_grad()                       # clear previous gradients
            output = model(x)                           # Forward pass
            loss = criterion(output, y)                 # Calculate loss
            loss.backward()                             # Backward pass (compute gradients)
            optimizer.step()                            # Update parameters
            total_loss += loss.item()                   # Track cumulative loss
        
        avg_loss = total_loss / len(train_loader)       # Average loss per batch
        loss_log.append(avg_loss)                       # Save for plotting
        print(f"[{optimizer_name.upper()}] Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")
    
    accuracy = evaluate(model)
    print(f"[{optimizer_name.upper()}] Final Test Accuracy: {accuracy:.2%}")
    
    return loss_log,accuracy



def main():

    # Train with both optimizers
    loss_lion, acc_lion = train_model("lion")
    loss_adamw, acc_adamw = train_model("adamw")

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(loss_lion, label=f"Lion ({acc_lion:.2%})", marker="o")
    plt.plot(loss_adamw, label=f"AdamW ({acc_adamw:.2%})", marker="x")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss: Lion vs AdamW on FashionMNIST")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/loss_comparison.png")
    plt.show()

if __name__ == "__main__":
    main()