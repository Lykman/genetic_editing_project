import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import torch
import torch.nn as nn
import torch.optim as optim
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from imblearn.over_sampling import SMOTE

# Load data
breast_cancer_data = pd.read_csv('/Users/ildar/genetic_editing_project/data/cleaned_breast_cancer_data.csv')
prostate_cancer_data = pd.read_csv('/Users/ildar/genetic_editing_project/data/cleaned_prostate_cancer_data.csv')

# Prepare data
X_breast = breast_cancer_data.drop('diagnosis', axis=1)
y_breast = breast_cancer_data['diagnosis']

X_prostate = prostate_cancer_data.drop('train', axis=1)
y_prostate = prostate_cancer_data['train']

# Split data
X_train_breast, X_test_breast, y_train_breast, y_test_breast = train_test_split(X_breast, y_breast, test_size=0.2, random_state=42)
X_train_prostate, X_test_prostate, y_train_prostate, y_test_prostate = train_test_split(X_prostate, y_prostate, test_size=0.2, random_state=42)

# Apply SMOTE to balance the classes
smote = SMOTE(random_state=42)
X_train_breast, y_train_breast = smote.fit_resample(X_train_breast, y_train_breast)
X_train_prostate, y_train_prostate = smote.fit_resample(X_train_prostate, y_train_prostate)

# Hyperparameter tuning for Scikit-learn models
param_grid_nn = {
    'mlpclassifier__hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100, 100), (100, 200, 100)],
    'mlpclassifier__activation': ['tanh', 'relu'],
    'mlpclassifier__solver': ['sgd', 'adam'],
    'mlpclassifier__alpha': [0.0001, 0.05, 0.01],
    'mlpclassifier__learning_rate': ['constant', 'adaptive'],
    'mlpclassifier__max_iter': [3000]  # Увеличиваем максимальное количество итераций
}

param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_features': ['sqrt', 'log2'],
    'max_depth': [10, 20, 30],
    'criterion': ['gini', 'entropy']
}

# Neural Network and Random Forest with GridSearchCV
nn_breast = make_pipeline(StandardScaler(), MLPClassifier(max_iter=3000, random_state=42))
rf_breast = RandomForestClassifier(random_state=42)

nn_prostate = make_pipeline(StandardScaler(), MLPClassifier(max_iter=3000, random_state=42))
rf_prostate = RandomForestClassifier(random_state=42)

# Grid Search for breast cancer data
grid_nn_breast = GridSearchCV(nn_breast, param_grid_nn, n_jobs=-1, cv=3)
grid_rf_breast = GridSearchCV(rf_breast, param_grid_rf, n_jobs=-1, cv=3)

grid_nn_breast.fit(X_train_breast, y_train_breast)
grid_rf_breast.fit(X_train_breast, y_train_breast)

acc_nn_breast = accuracy_score(y_test_breast, grid_nn_breast.predict(X_test_breast))
acc_rf_breast = accuracy_score(y_test_breast, grid_rf_breast.predict(X_test_breast))

# Grid Search for prostate cancer data
grid_nn_prostate = GridSearchCV(nn_prostate, param_grid_nn, n_jobs=-1, cv=3)
grid_rf_prostate = GridSearchCV(rf_prostate, param_grid_rf, n_jobs=-1, cv=3)

grid_nn_prostate.fit(X_train_prostate, y_train_prostate)
grid_rf_prostate.fit(X_train_prostate, y_train_prostate)

acc_nn_prostate = accuracy_score(y_test_prostate, grid_nn_prostate.predict(X_test_prostate))
acc_rf_prostate = accuracy_score(y_test_prostate, grid_rf_prostate.predict(X_test_prostate))

# PyTorch Neural Network
class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

def train_pytorch_model(X_train, y_train, X_test, y_test):
    model = SimpleNN(X_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

    for epoch in range(3000):  # Увеличиваем количество эпох
        model.train()
        optimizer.zero_grad()
        output = model(X_train_tensor)
        loss = criterion(output, y_train_tensor)
        loss.backward()
        optimizer.step()

    model.eval()
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)
    with torch.no_grad():
        output = model(X_test_tensor)
    preds = (output.numpy() > 0.5).astype(int)
    accuracy = accuracy_score(y_test, preds)
    return accuracy

acc_pytorch_nn_breast = train_pytorch_model(X_train_breast, y_train_breast, X_test_breast, y_test_breast)
acc_pytorch_nn_prostate = train_pytorch_model(X_train_prostate, y_train_prostate, X_test_prostate, y_test_prostate)

# TensorFlow Neural Network
def create_tf_model(input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_tensorflow_model(X_train, y_train, X_test, y_test):
    model = create_tf_model(X_train.shape[1])
    model.fit(X_train, y_train, epochs=300, batch_size=32, verbose=0)  # Увеличиваем количество эпох
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    return accuracy

acc_tensorflow_nn_breast = train_tensorflow_model(X_train_breast, y_train_breast, X_test_breast, y_test_breast)
acc_tensorflow_nn_prostate = train_tensorflow_model(X_train_prostate, y_train_prostate, X_test_prostate, y_test_prostate)

# Plot accuracies
models = ['NN (Scikit-learn)', 'RF (Scikit-learn)', 'NN (PyTorch)', 'NN (TensorFlow)']
breast_cancer_accuracies = [acc_nn_breast, acc_rf_breast, acc_pytorch_nn_breast, acc_tensorflow_nn_breast]
prostate_cancer_accuracies = [acc_nn_prostate, acc_rf_prostate, acc_pytorch_nn_prostate, acc_tensorflow_nn_prostate]

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.bar(models, breast_cancer_accuracies, color='blue')
plt.ylim(0, 1)
plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison - Breast Cancer')

plt.subplot(1, 2, 2)
plt.bar(models, prostate_cancer_accuracies, color='green')
plt.ylim(0, 1)
plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison - Prostate Cancer')

plt.tight_layout()
plt.savefig('/Users/ildar/genetic_editing_project/results/accuracy_comparison_improved_gridsearch.png')
plt.show()