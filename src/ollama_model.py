import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import model_from_json
from tensorflow.keras.optimizers import Adam

def preprocess_data(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

def train_ollama_model(X_train, y_train, model_json_path, model_weights_path):
    X_train_scaled, scaler = preprocess_data(X_train)

    # Загружаем архитектуру модели из JSON-файла
    with open(model_json_path, "r") as json_file:
        loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
    
    # Проверка соответствия входных данных
    expected_input_dim = model.input_shape[1]
    if X_train.shape[1] != expected_input_dim:
        raise ValueError(f"Ollama model input dimension {expected_input_dim} does not match data input dimension {X_train.shape[1]}")

    # Загружаем веса модели
    model.load_weights(model_weights_path)
    
    # Компиляция модели
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    # Обучение модели
    model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, verbose=1, validation_split=0.2)
    
    return model, scaler

def evaluate_ollama_model(model, X_test, y_test, scaler):
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    y_pred = (y_pred > 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    return accuracy, report

def load_ollama_model(model_json_path, model_weights_path):
    # Загружаем архитектуру модели из JSON-файла
    with open(model_json_path, "r") as json_file:
        loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
    
    # Загружаем веса модели
    model.load_weights(model_weights_path)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Пример использования (можно удалить или адаптировать под ваши нужды)
if __name__ == "__main__":
    # Загрузка данных
    data_path = '../data/cleaned_breast_cancer_data.csv'
    data = pd.read_csv(data_path)
    X = data.drop('diagnosis', axis=1)
    y = data['diagnosis']
    
    # Разделение данных
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Пути для сохранения модели
    model_json_path = '../results/breast_cancer/model.json'
    model_weights_path = '../results/breast_cancer/model.weights.h5'
    
    # Обучение модели
    model, scaler = train_ollama_model(X_train, y_train, model_json_path, model_weights_path)
    
    # Оценка модели
    accuracy, report = evaluate_ollama_model(model, X_test, y_test, scaler)
    print(f"Accuracy: {accuracy}")
    print(f"Classification Report:\n{report}")
    
    # Сохранение модели
    # Сохранение архитектуры модели
    model_json = model.to_json()
    with open(model_json_path, "w") as json_file:
        json_file.write(model_json)
    
    # Сохранение весов модели
    model.save_weights(model_weights_path)

    # Загрузка и повторная оценка модели
    loaded_model = load_ollama_model(model_json_path, model_weights_path)
    accuracy, report = evaluate_ollama_model(loaded_model, X_test, y_test, scaler)
    print(f"Loaded Model Accuracy: {accuracy}")
    print(f"Loaded Model Classification Report:\n{report}")
