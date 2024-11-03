import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import optuna


# Define the dataset and data loaders
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_data = datasets.FashionMNIST(root="./data", train=True, transform=transform, download=True)
test_data = datasets.FashionMNIST(root="./data", train=False, transform=transform, download=True)

train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)


# Define the model
class FashionMnistClassifier(nn.Module):
    def __init__(self, dropout_rate):
        super(FashionMnistClassifier, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Objective function for Optuna
def objective(trial):
    # Suggest hyperparameters
    dropout_rate = trial.suggest_float("dropout_rate", 0.2, 0.5)
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 1e-2)

    # Initialize model, loss function, and optimizer
    model = FashionMnistClassifier(dropout_rate).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    epochs = 3  # Total epochs for each trial
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Evaluate on validation set (test_loader)
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total

        # Report the current accuracy to Optuna for pruning
        trial.report(accuracy, epoch)

        # Handle pruning based on the reported accuracy
        if trial.should_prune():
            raise optuna.TrialPruned()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}, Acc: {accuracy:.2f}%")

    return accuracy


# Run Optuna study
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=2)

# Print best parameters and value
print(
    "Best parameters:", study.best_params
)  #'dropout_rate': 0.48492750072958846, 'learning_rate': 0.0002332838875919644
print("Best accuracy:", study.best_value)
