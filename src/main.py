import logging
import sys
import os
import matplotlib.pyplot as plt

# Добавить корневую папку проекта в sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.enhanced_nn import train_enhanced_nn, evaluate_enhanced_nn, optimize_hyperparameters
from src.ensemble_models import train_ensemble, evaluate_ensemble, grid_search_rf
from src.data_loader import load_and_preprocess_data
from src.ollama_model import train_ollama_model, evaluate_ollama_model  # Импортируем новые функции
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
import numpy as np

logging.basicConfig(level=logging.INFO)

def create_model(input_dim):
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

class KerasClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, build_fn=None, **sk_params):
        self.build_fn = build_fn
        self.sk_params = sk_params
        self.model = None
        self.history = None

    def fit(self, X, y, **fit_params):
        self.model = self.build_fn()
        self.history = self.model.fit(X, y, **fit_params)
        return self

    def predict(self, X):
        pred_proba = self.model.predict(X)
        return (pred_proba > 0.5).astype(int)

    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

def optimize_hyperparameters_tf(X, y, input_dim):
    def create_keras_model():
        return create_model(input_dim)

    model = KerasClassifier(build_fn=create_keras_model, epochs=100, batch_size=10, verbose=0)
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    best_score = 0
    best_model = None
    best_history = None
    for train_index, val_index in kfold.split(X):
        X_train_fold, X_val_fold = X[train_index], X[val_index]
        y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]
        model.fit(X_train_fold, y_train_fold, validation_data=(X_val_fold, y_val_fold),
                  callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)])
        score = model.score(X_val_fold, y_val_fold)
        if score > best_score:
            best_score = score
            best_model = model
            best_history = model.history.history

    return best_model, best_score, best_history

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    return accuracy, report

def run_analysis(data_path, target_column, result_dir, model_json_path, model_weights_path, expected_input_dim):
    X_train, X_test, y_train, y_test = load_and_preprocess_data(data_path, target_column)

    if X_train.shape[1] != expected_input_dim:
        raise ValueError(f"Expected input dimension {expected_input_dim} but got {X_train.shape[1]}")

    if X_train is None or y_train is None:
        print(f"Failed to load and preprocess data for {data_path}.")
        return None, None, None

    input_dim = X_train.shape[1]

    # PyTorch модель
    best_nn_model, best_nn_params = optimize_hyperparameters(X_train, y_train, input_dim)
    nn_accuracy, nn_report = evaluate_enhanced_nn(best_nn_model, X_test, y_test)
    print(f"Neural Network (PyTorch) Evaluation Results for {data_path} Data:")
    print(f"Accuracy: {nn_accuracy}")
    print(f"Classification Report:\n{nn_report}")

    # TensorFlow модель
    best_tf_model, best_tf_score, best_tf_history = optimize_hyperparameters_tf(X_train, y_train, input_dim)
    tf_accuracy, tf_report = evaluate_model(best_tf_model, X_test, y_test)
    print(f"Neural Network (TensorFlow) Evaluation Results for {data_path} Data:")
    print(f"Accuracy: {tf_accuracy}")
    print(f"Classification Report:\n{tf_report}")

    # Ollama модель
    ollama_model, ollama_scaler = train_ollama_model(X_train, y_train, model_json_path, model_weights_path)  # Добавлены недостающие аргументы
    ollama_accuracy, ollama_report = evaluate_ollama_model(ollama_model, X_test, y_test, ollama_scaler)
    print(f"Ollama Llama3 Model Evaluation Results for {data_path} Data:")
    print(f"Accuracy: {ollama_accuracy}")
    print(f"Classification Report:\n{ollama_report}")

    # RandomForest модель
    best_rf_model, best_rf_params = grid_search_rf(X_train, y_train)
    rf_accuracy, rf_report = evaluate_ensemble(best_rf_model, X_test, y_test)
    print(f"Ensemble Model Evaluation Results for {data_path} Data:")
    print(f"Accuracy: {rf_accuracy}")
    print(f"Classification Report:\n{rf_report}")

    with open(os.path.join(result_dir, 'nn_report.txt'), 'w') as f:
        f.write(f"Accuracy: {nn_accuracy}\n")
        f.write(f"Classification Report:\n{nn_report}\n")

    with open(os.path.join(result_dir, 'tf_report.txt'), 'w') as f:
        f.write(f"Accuracy: {tf_accuracy}\n")
        f.write(f"Classification Report:\n{tf_report}\n")

    with open(os.path.join(result_dir, 'ollama_report.txt'), 'w') as f:
        f.write(f"Accuracy: {ollama_accuracy}\n")
        f.write(f"Classification Report:\n{ollama_report}\n")

    with open(os.path.join(result_dir, 'rf_report.txt'), 'w') as f:
        f.write(f"Accuracy: {rf_accuracy}\n")
        f.write(f"Classification Report:\n{rf_report}\n")

    return nn_accuracy, tf_accuracy, ollama_accuracy, rf_accuracy, best_nn_model, best_tf_model, ollama_model

def main():
    breast_cancer_data_path = '/Users/ildar/genetic_editing_project/data/cleaned_breast_cancer_data.csv'
    prostate_cancer_data_path = '/Users/ildar/genetic_editing_project/data/cleaned_prostate_cancer_data.csv'

    breast_cancer_result_dir = '/Users/ildar/genetic_editing_project/results/breast_cancer'
    prostate_cancer_result_dir = '/Users/ildar/genetic_editing_project/results/prostate_cancer'

    os.makedirs(breast_cancer_result_dir, exist_ok=True)
    os.makedirs(prostate_cancer_result_dir, exist_ok=True)

    print("Running analysis for Breast Cancer...")
    nn_accuracy_bc, tf_accuracy_bc, ollama_accuracy_bc, rf_accuracy_bc, nn_model_bc, tf_model_bc, ollama_model_bc = run_analysis(
        breast_cancer_data_path, "diagnosis", breast_cancer_result_dir, 
        '/Users/ildar/genetic_editing_project/results/breast_cancer/model.json', 
        '/Users/ildar/genetic_editing_project/results/breast_cancer/model.weights.h5', 11)

    print("Running analysis for Prostate Cancer...")
    nn_accuracy_pc, tf_accuracy_pc, ollama_accuracy_pc, rf_accuracy_pc, nn_model_pc, tf_model_pc, ollama_model_pc = run_analysis(
        prostate_cancer_data_path, "train", prostate_cancer_result_dir, 
        '/Users/ildar/genetic_editing_project/results/prostate_cancer/model.json', 
        '/Users/ildar/genetic_editing_project/results/prostate_cancer/model.weights.h5', 7)

    if nn_model_bc is not None:
        plt.figure()
        nn_train_losses_bc = nn_model_bc.train_losses
        plt.plot(nn_train_losses_bc, label='Training Loss')
        plt.title('Training Loss for Breast Cancer Neural Network (PyTorch)')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(breast_cancer_result_dir, 'Breast_Cancer_nn_training_loss.png'))
        plt.close()

    if tf_model_bc is not None:
        plt.figure()
        tf_train_losses_bc = tf_model_bc.history.history['loss']
        plt.plot(tf_train_losses_bc, label='Training Loss')
        plt.title('Training Loss for Breast Cancer Neural Network (TensorFlow)')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(breast_cancer_result_dir, 'Breast_Cancer_tf_training_loss.png'))
        plt.close()

    if ollama_model_bc is not None and hasattr(ollama_model_bc, 'history') and 'loss' in ollama_model_bc.history.history:
        plt.figure()
        ollama_train_losses_bc = ollama_model_bc.history.history['loss']
        plt.plot(ollama_train_losses_bc, label='Training Loss')
        plt.title('Training Loss for Breast Cancer Model (Ollama Llama3)')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(breast_cancer_result_dir, 'Breast_Cancer_ollama_training_loss.png'))
        plt.close()

    if nn_model_pc is not None:
        plt.figure()
        nn_train_losses_pc = nn_model_pc.train_losses
        plt.plot(nn_train_losses_pc, label='Training Loss')
        plt.title('Training Loss for Prostate Cancer Neural Network (PyTorch)')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(prostate_cancer_result_dir, 'Prostate_Cancer_nn_training_loss.png'))
        plt.close()

    if tf_model_pc is not None:
        plt.figure()
        tf_train_losses_pc = tf_model_pc.history.history['loss']
        plt.plot(tf_train_losses_pc, label='Training Loss')
        plt.title('Training Loss for Prostate Cancer Neural Network (TensorFlow)')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(prostate_cancer_result_dir, 'Prostate_Cancer_tf_training_loss.png'))
        plt.close()

    if ollama_model_pc is not None and hasattr(ollama_model_pc, 'history') and 'loss' in ollama_model_pc.history.history:
        plt.figure()
        ollama_train_losses_pc = ollama_model_pc.history.history['loss']
        plt.plot(ollama_train_losses_pc, label='Training Loss')
        plt.title('Training Loss for Prostate Cancer Model (Ollama Llama3)')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(prostate_cancer_result_dir, 'Prostate_Cancer_ollama_training_loss.png'))
        plt.close()

if __name__ == "__main__":
    main()
