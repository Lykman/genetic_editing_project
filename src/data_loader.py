import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        logging.info(f"Data from '{file_path}' loaded successfully.")
        return data
    except Exception as e:
        logging.error(f"Error loading data from '{file_path}': {e}")
        return None

def preprocess_data(data, target_column):
    try:
        data = data.dropna()

        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")

        X = data.drop(columns=[target_column])
        y = data[target_column].apply(lambda x: 1 if x == 'M' else 0)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        enet = ElasticNetCV(cv=5, random_state=0, max_iter=1000000, tol=1e-4, l1_ratio=0.5, n_alphas=100)
        enet.fit(X_scaled, y)
        model = SelectFromModel(enet, prefit=True)
        X_new = model.transform(X_scaled)

        pca = PCA(n_components=0.95)
        X_pca = pca.fit_transform(X_new)

        X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

        logging.info("Data preprocessed successfully with feature selection and PCA.")
        return X_train, X_test, y_train, y_test, scaler, model, pca
    except Exception as e:
        logging.error(f"Error preprocessing data: {e}")
        return None, None, None, None, None, None, None

def load_and_preprocess_data(file_path, target_column):
    data = load_data(file_path)
    if data is None:
        return None, None, None, None
    return preprocess_data(data, target_column)[:4]
