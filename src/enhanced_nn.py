# enhanced_nn.py
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class EnhancedNN(nn.Module):
    def __init__(self, input_size, hidden_sizes=[512, 256, 128], dropout_rate=0.5):
        super(EnhancedNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.fc4 = nn.Linear(hidden_sizes[2], 1)
        self.leaky_relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.sigmoid = nn.Sigmoid()
        self.train_losses = []
        self.scaler = None

    def forward(self, x):
        x = self.leaky_relu(self.fc1(x))
        x = self.dropout(x)
        x = self.leaky_relu(self.fc2(x))
        x = self.dropout(x)
        x = self.leaky_relu(self.fc3(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc4(x))
        return x

def preprocess_labels(y_train):
    if isinstance(y_train, pd.Series):
        y_train = y_train.to_numpy()
    if y_train.dtype == 'object':
        y_train = pd.Series(y_train).astype('category').cat.codes.to_numpy()
    if not isinstance(y_train, np.ndarray):
        y_train = y_train.numpy()
    return y_train.astype(np.float32)

def train_enhanced_nn(X_train, y_train, input_size, epochs=100, learning_rate=0.001, hidden_sizes=[512, 256, 128], dropout_rate=0.5):
    model = EnhancedNN(input_size, hidden_sizes, dropout_rate)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = preprocess_labels(y_train)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

    model.scaler = scaler

    best_loss = float('inf')
    patience = 10
    patience_counter = 0
    min_lr = 1e-6
    factor = 0.2

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        model.train_losses.append(loss.item())

        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

            if patience_counter % patience == 0:
                for param_group in optimizer.param_groups:
                    new_lr = max(param_group['lr'] * factor, min_lr)
                    if param_group['lr'] > new_lr:
                        print(f"Reducing learning rate from {param_group['lr']} to {new_lr}")
                        param_group['lr'] = new_lr

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

    plt.figure()
    plt.plot(model.train_losses, label='Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.savefig('train_loss_plot.png')
    plt.close()

    return model, scaler

def evaluate_enhanced_nn(model, X_test, y_test):
    X_test = model.scaler.transform(X_test)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = preprocess_labels(y_test)
    y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
    preds = (outputs.numpy() > 0.5).astype(int)
    accuracy = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, zero_division=0)
    return accuracy, report

def optimize_hyperparameters(X, y, input_size):
    param_grid = {
        'epochs': [50, 100],
        'learning_rate': [0.001, 0.01],
        'hidden_sizes': [[512, 256, 128], [1024, 512, 256]],
        'dropout_rate': [0.3, 0.5]
    }

    best_params = None
    best_score = 0
    best_model = None
    kf = KFold(n_splits=5)

    for epochs in param_grid['epochs']:
        for lr in param_grid['learning_rate']:
            for hidden_sizes in param_grid['hidden_sizes']:
                for dropout_rate in param_grid['dropout_rate']:
                    scores = []
                    for train_index, val_index in kf.split(X):
                        X_train, X_val = X[train_index], X[val_index]
                        y_train, y_val = y.iloc[train_index], y.iloc[val_index]  # Исправление здесь
                        model, scaler = train_enhanced_nn(X_train, y_train, input_size, epochs, lr, hidden_sizes, dropout_rate)
                        accuracy, _ = evaluate_enhanced_nn(model, X_val, y_val)
                        scores.append(accuracy)
                    mean_score = np.mean(scores)
                    if mean_score > best_score:
                        best_score = mean_score
                        best_params = {'epochs': epochs, 'learning_rate': lr, 'hidden_sizes': hidden_sizes, 'dropout_rate': dropout_rate}
                        best_model = model

    print(f"Best Hyperparameters: {best_params}")
    return best_model, best_params
