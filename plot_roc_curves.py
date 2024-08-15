import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, classification_report, accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import model_from_json
import os

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

# Correct dimensions
X_train_breast = X_train_breast.iloc[:, :11]
X_test_breast = X_test_breast.iloc[:, :11]

X_train_prostate = X_train_prostate.iloc[:, :7]
X_test_prostate = X_test_prostate.iloc[:, :7]

# Train models
nn_breast = make_pipeline(StandardScaler(), MLPClassifier(max_iter=300, random_state=42))
rf_breast = RandomForestClassifier(random_state=42)

nn_prostate = make_pipeline(StandardScaler(), MLPClassifier(max_iter=300, random_state=42))
rf_prostate = RandomForestClassifier(random_state=42)

nn_breast.fit(X_train_breast, y_train_breast)
rf_breast.fit(X_train_breast, y_train_breast)

nn_prostate.fit(X_train_prostate, y_train_prostate)
rf_prostate.fit(X_train_prostate, y_train_prostate)

# Predict probabilities
y_score_nn_breast = nn_breast.predict_proba(X_test_breast)[:, 1]
y_score_rf_breast = rf_breast.predict_proba(X_test_breast)[:, 1]

y_score_nn_prostate = nn_prostate.predict_proba(X_test_prostate)[:, 1]
y_score_rf_prostate = rf_prostate.predict_proba(X_test_prostate)[:, 1]

# Load and predict Ollama models
def load_ollama_model(json_path, weights_path):
    with open(json_path, 'r') as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json)
    model.load_weights(weights_path)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Paths to the Ollama models
breast_ollama_json = '/Users/ildar/genetic_editing_project/results/breast_cancer/model.json'
breast_ollama_weights = '/Users/ildar/genetic_editing_project/results/breast_cancer/model.weights.h5'

prostate_ollama_json = '/Users/ildar/genetic_editing_project/results/prostate_cancer/model.json'
prostate_ollama_weights = '/Users/ildar/genetic_editing_project/results/prostate_cancer/model.weights.h5'

# Load Ollama models
ollama_model_breast = load_ollama_model(breast_ollama_json, breast_ollama_weights)
ollama_model_prostate = load_ollama_model(prostate_ollama_json, prostate_ollama_weights)

# Scale data for Ollama models
scaler_breast = StandardScaler().fit(X_train_breast)
X_test_breast_scaled = scaler_breast.transform(X_test_breast)

scaler_prostate = StandardScaler().fit(X_train_prostate)
X_test_prostate_scaled = scaler_prostate.transform(X_test_prostate)

# Debug: Print shapes of the data
print(f"Shape of X_test_breast_scaled: {X_test_breast_scaled.shape}")
print(f"Shape of X_test_prostate_scaled: {X_test_prostate_scaled.shape}")

# Predict probabilities for Ollama models
y_score_ollama_breast = ollama_model_breast.predict(X_test_breast_scaled).ravel()
y_score_ollama_prostate = ollama_model_prostate.predict(X_test_prostate_scaled).ravel()

# Compute ROC curve and ROC area
fpr_nn_breast, tpr_nn_breast, _ = roc_curve(y_test_breast, y_score_nn_breast)
roc_auc_nn_breast = auc(fpr_nn_breast, tpr_nn_breast)

fpr_rf_breast, tpr_rf_breast, _ = roc_curve(y_test_breast, y_score_rf_breast)
roc_auc_rf_breast = auc(fpr_rf_breast, tpr_rf_breast)

fpr_ollama_breast, tpr_ollama_breast, _ = roc_curve(y_test_breast, y_score_ollama_breast)
roc_auc_ollama_breast = auc(fpr_ollama_breast, tpr_ollama_breast)

fpr_nn_prostate, tpr_nn_prostate, _ = roc_curve(y_test_prostate, y_score_nn_prostate)
roc_auc_nn_prostate = auc(fpr_nn_prostate, tpr_nn_prostate)

fpr_rf_prostate, tpr_rf_prostate, _ = roc_curve(y_test_prostate, y_score_rf_prostate)
roc_auc_rf_prostate = auc(fpr_rf_prostate, tpr_rf_prostate)

fpr_ollama_prostate, tpr_ollama_prostate, _ = roc_curve(y_test_prostate, y_score_ollama_prostate)
roc_auc_ollama_prostate = auc(fpr_ollama_prostate, tpr_ollama_prostate)

# Plot ROC curves
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(fpr_nn_breast, tpr_nn_breast, color='blue', lw=2, label='Neural Network (area = %0.2f)' % roc_auc_nn_breast)
plt.plot(fpr_rf_breast, tpr_rf_breast, color='green', lw=2, label='Random Forest (area = %0.2f)' % roc_auc_rf_breast)
plt.plot(fpr_ollama_breast, tpr_ollama_breast, color='red', lw=2, label='Ollama (area = %0.2f)' % roc_auc_ollama_breast)
plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic - Breast Cancer')
plt.legend(loc="lower right")

plt.subplot(1, 2, 2)
plt.plot(fpr_nn_prostate, tpr_nn_prostate, color='blue', lw=2, label='Neural Network (area = %0.2f)' % roc_auc_nn_prostate)
plt.plot(fpr_rf_prostate, tpr_rf_prostate, color='green', lw=2, label='Random Forest (area = %0.2f)' % roc_auc_rf_prostate)
plt.plot(fpr_ollama_prostate, tpr_ollama_prostate, color='red', lw=2, label='Ollama (area = %0.2f)' % roc_auc_ollama_prostate)
plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic - Prostate Cancer')
plt.legend(loc="lower right")

plt.tight_layout()
plt.savefig('/Users/ildar/genetic_editing_project/results/roc_curves.png')
plt.show()

# Additional metrics
print("Neural Network Breast Cancer Report:")
print(classification_report(y_test_breast, nn_breast.predict(X_test_breast)))

print("Random Forest Breast Cancer Report:")
print(classification_report(y_test_breast, rf_breast.predict(X_test_breast)))

print("Neural Network Prostate Cancer Report:")
print(classification_report(y_test_prostate, nn_prostate.predict(X_test_prostate)))

print("Random Forest Prostate Cancer Report:")
print(classification_report(y_test_prostate, rf_prostate.predict(X_test_prostate)))

print("Ollama Breast Cancer Report:")
print(classification_report(y_test_breast, (y_score_ollama_breast > 0.5).astype(int)))

print("Ollama Prostate Cancer Report:")
print(classification_report(y_test_prostate, (y_score_ollama_prostate > 0.5).astype(int)))
