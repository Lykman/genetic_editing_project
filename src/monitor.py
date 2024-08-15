import logging
import matplotlib.pyplot as plt
import pandas as pd

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def monitor_patient(genome, patient_id, output_dir='monitoring_data'):
    """
    Мониторинг пациента на основе его генома.
    В реальности это включало бы регулярные медицинские тесты и генетическое секвенирование.
    """
    logging.info(f"Monitoring patient {patient_id} with genome: {genome}")
    
    # Placeholder для логики мониторинга
    # В реальном сценарии это включало бы вызовы медицинских API или баз данных

    # Создание фейковых данных мониторинга
    data = {
        'time': range(10),  # Время в днях или других единицах
        'biomarker_level': [genome.count('edited_')] * 10  # Пример: количество редактированных генов
    }
    df = pd.DataFrame(data)

    # Сохранение данных мониторинга в CSV файл
    output_file = f"{output_dir}/patient_{patient_id}_monitoring.csv"
    df.to_csv(output_file, index=False)
    logging.info(f"Monitoring data saved to {output_file}")

    # Визуализация данных мониторинга
    plt.figure()
    plt.plot(df['time'], df['biomarker_level'], marker='o', linestyle='-')
    plt.title(f'Monitoring Patient {patient_id}')
    plt.xlabel('Time')
    plt.ylabel('Biomarker Level')
    plt.savefig(f"{output_dir}/patient_{patient_id}_monitoring.png")
    plt.close()
    logging.info(f"Monitoring plot saved to {output_dir}/patient_{patient_id}_monitoring.png")

# Пример использования
if __name__ == "__main__":
    genome = "edited_BRCA1 edited_TP53 other_genes"
    patient_id = 1
    monitor_patient(genome, patient_id)
