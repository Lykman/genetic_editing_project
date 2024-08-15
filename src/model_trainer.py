import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Lasso, ElasticNet

class AdvancedNN(nn.Module):
    def __init__(self, input_size, hidden_sizes=[256, 128, 64, 32], dropout_rate=0.5):
        super(AdvancedNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.fc4 = nn.Linear(hidden_sizes[2], hidden_sizes[3])
        self.fc5 = nn.Linear(hidden_sizes[3], 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.relu(self.fc4(x))
        x = self.sigmoid(self.fc5(x))
        return x

def preprocess_labels(y_train):
    if isinstance(y_train, pd.Series):
        y_train = y_train.to_numpy()
    if y_train.dtype == 'object':
        y_train = pd.Series(y_train).astype('category').cat.codes.to_numpy()
    y_train = (y_train > 0).astype(np.float32)
    return y_train

def train_advanced_nn(X_train, y_train, input_size, epochs=100, learning_rate=0.001, hidden_sizes=[256, 128, 64, 32], dropout_rate=0.5):
    model = AdvancedNN(input_size, hidden_sizes, dropout_rate)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = preprocess_labels(y_train)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

    train_losses = []

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        if torch.isnan(outputs).any():
            print(f"NaN detected in outputs at epoch {epoch}")
            break
        loss = criterion(outputs, y_train)
        if torch.isnan(loss):
            print(f"NaN detected in loss at epoch {epoch}")
            break
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

    return model, train_losses

def evaluate_advanced_nn(model, X_test, y_test):
    model.eval()
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = preprocess_labels(y_test)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
    outputs = model(X_test)
    preds = (outputs.detach().numpy() > 0.5).astype(int)
    accuracy = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, zero_division=0)
    return accuracy, report

def cross_validate_model(data, target_column, input_size, epochs=100, learning_rate=0.001, hidden_sizes=[256, 128, 64, 32], dropout_rate=0.5):
    X = data.drop(columns=[target_column]).values
    y = data[target_column].values
    kf = KFold(n=5, shuffle=True, random_state=42)
    scores = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        model, _ = train_advanced_nn(X_train, y_train, input_size, epochs, learning_rate, hidden_sizes, dropout_rate)
        accuracy, _ = evaluate_advanced_nn(model, X_test, y_test)
        scores.append(accuracy)

    return scores

def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_random_forest(model, X_test, y_test):
    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, zero_division=0)
    return accuracy, report

def cross_validate_random_forest(data, target_column):
    X = data.drop(columns=[target_column]).values
    y = data[target_column].values
    kf = KFold(n=5, shuffle=True, random_state=42)
    scores = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        model = train_random_forest(X_train, y_train)
        accuracy, _ = evaluate_random_forest(model, X_test, y_test)
        scores.append(accuracy)

    return scores

def train_lasso(X_train, y_train):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    lasso = Lasso(max_iter=10000)
    lasso.fit(X_train, y_train)
    return lasso, scaler

def train_elastic_net(X_train, y_train):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    en = ElasticNet(max_iter=10000)
    en.fit(X_train, y_train)
    return en, scaler

def grid_search_nn(X, y, input_size):
    param_grid = {
        'epochs': [50, 100],
        'learning_rate': [0.001, 0.01],
        'hidden_sizes': [[256, 128, 64, 32], [128, 64, 32]],
        'dropout_rate': [0.3, 0.5]
    }

    class NeuralNetworkClassifier:
        def __init__(self, input_size, epochs=100, learning_rate=0.001, hidden_sizes=[256, 128, 64, 32], dropout_rate=0.5):
            self.input_size = input_size
            self.epochs = epochs
            self.learning_rate = learning_rate
            self.hidden_sizes = hidden_sizes
            self.dropout_rate = dropout_rate
            self.model = None

        def fit(self, X, y):
            X_tensor = torch.tensor(X, dtype=torch.float32)
            y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
            self.model, _ = train_advanced_nn(X_tensor, y_tensor, self.input_size, self.epochs, self.learning_rate, self.hidden_sizes, self.dropout_rate)

        def predict(self, X):
            X_tensor = torch.tensor(X, dtype=torch.float32)
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(X_tensor)
            preds = (outputs.numpy() > 0.5).astype(int)
            return preds.flatten()

    grid_search = GridSearchCV(
        estimator=NeuralNetworkClassifier(input_size),
        param_grid=param_grid,
        scoring='accuracy',
        cv=3
    )
    grid_search.fit(X, y)

    return grid_search.best_estimator_, grid_search.best_params_

def grid_search_rf(X, y):
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        scoring='accuracy',
        cv=3
    )
    grid_search.fit(X, y)

    return grid_search.best_estimator_, grid_search.best_params_
def train_lasso(X_train, y_train):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    lasso = Lasso(max_iter=100000)  # Увеличено число итераций
    lasso.fit(X_train, y_train)
    return lasso, scaler

def train_elastic_net(X_train, y_train):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    en = ElasticNet(max_iter=100000)  # Увеличено число итераций
    en.fit(X_train, y_train)
    return en, scaler
def train_advanced_nn(X_train, y_train, input_size, epochs=100, learning_rate=0.001, hidden_sizes=[256, 128, 64, 32], dropout_rate=0.5):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    model = AdvancedNN(input_size, hidden_sizes, dropout_rate)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = preprocess_labels(y_train)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

    train_losses = []

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        if torch.isnan(outputs).any():
            print(f"NaN detected in outputs at epoch {epoch}")
            break
        loss = criterion(outputs, y_train)
        if torch.isnan(loss):
            print(f"NaN detected in loss at epoch {epoch}")
            break
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

    return model, train_losses, scaler
