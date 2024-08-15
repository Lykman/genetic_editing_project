import pandas as pd
import numpy as np

# Загрузка данных
breast_cancer_data_path = '/Users/ildar/genetic_editing_project/data/data/breast-cancer-wisconsin-data-new.csv'
prostate_cancer_data_path = '/Users/ildar/genetic_editing_project/data/prostate.csv'

# Проверка и очистка данных для Breast Cancer
def clean_breast_cancer_data(file_path):
    data = pd.read_csv(file_path)
    print("Original Breast Cancer Data:")
    print(data.head())
    
    # Удаление столбца 'Unnamed: 32'
    if 'Unnamed: 32' in data.columns:
        data = data.drop(columns=['Unnamed: 32'])
    
    # Преобразование категориального столбца diagnosis в числовой
    data['diagnosis'] = data['diagnosis'].astype('category').cat.codes
    
    # Заполнение пропусков медианными значениями
    data = data.fillna(data.median())
    
    print("Cleaned Breast Cancer Data:")
    print(data.head())
    return data

# Проверка и очистка данных для Prostate Cancer
def clean_prostate_cancer_data(file_path):
    data = pd.read_csv(file_path)
    print("Original Prostate Cancer Data:")
    print(data.head())
    
    # Преобразование категориального столбца train в числовой
    data['train'] = data['train'].astype('category').cat.codes
    
    # Заполнение пропусков медианными значениями
    data = data.fillna(data.median())
    
    print("Cleaned Prostate Cancer Data:")
    print(data.head())
    return data

# Очистка данных
cleaned_breast_cancer_data = clean_breast_cancer_data(breast_cancer_data_path)
cleaned_prostate_cancer_data = clean_prostate_cancer_data(prostate_cancer_data_path)

# Сохранение очищенных данных
cleaned_breast_cancer_data.to_csv('/Users/ildar/genetic_editing_project/data/cleaned_breast_cancer_data.csv', index=False)
cleaned_prostate_cancer_data.to_csv('/Users/ildar/genetic_editing_project/data/cleaned_prostate_cancer_data.csv', index=False)
