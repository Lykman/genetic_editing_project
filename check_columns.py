import pandas as pd

# Загрузка данных
breast_cancer_data = pd.read_csv('/Users/ildar/genetic_editing_project/data/cleaned_breast_cancer_data.csv')
prostate_cancer_data = pd.read_csv('/Users/ildar/genetic_editing_project/data/cleaned_prostate_cancer_data.csv')

# Вывод заголовков столбцов
print("Заголовки столбцов для данных рака молочной железы:")
print(breast_cancer_data.columns)

print("\nЗаголовки столбцов для данных рака простаты:")
print(prostate_cancer_data.columns)
