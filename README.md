# Genetic Editing Project

## Overview

This project aims to analyze genetic data, identify problematic genes associated with cancer, and apply genetic editing techniques using AI USM and CRISPR-Cas9. The project involves preprocessing data, training machine learning models, identifying target genes, and applying genetic editing simulations.

## Structure

- `data/` - Directory containing the dataset.
  - `cleaned_breast_cancer_data.csv` - Cleaned dataset for breast cancer.
  - `cleaned_prostate_cancer_data.csv` - Cleaned dataset for prostate cancer.
  - Other raw datasets as necessary.
- `src/` - Directory containing the source code.
  - `clean_data.py` - Script for cleaning the raw data.
  - `data_loader.py` - Script for loading and preprocessing data.
  - `enhanced_nn.py` - Script for training and evaluating enhanced neural network models.
  - `ensemble_models.py` - Script for training and evaluating ensemble models (e.g., Random Forest).
  - `gene_editor.py` - Script for identifying problematic genes and applying CRISPR-Cas9 edits.
  - `model_trainer.py` - Script for training and saving machine learning models.
  - `monitor.py` - Script for monitoring patients after genetic editing.
  - `plot_roc_curves.py` - Script for plotting ROC curves for model evaluation.
  - `main.py` - Main script to run the entire process.

## Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/genetic-editing-project.git
   cd genetic-editing-project
   ```

2. **Install the dependencies**:
   Make sure you have Python installed. Then, install the required packages using pip:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the project**:
   You can start by cleaning the data and training the models:
   ```bash
   python src/clean_data.py
   python src/model_trainer.py
   ```

4. **Monitor the results**:
   After running the models, you can monitor the results using:
   ```bash
   python src/monitor.py
   ```

5. **Simulate Genetic Edits**:
   Use the gene editor to simulate CRISPR-Cas9 interventions:
   ```bash
   python src/gene_editor.py
   ```

## How to Run

1. **Prepare your environment**:
   - Install Python 3.8 or higher.
   - Clone the repository and navigate to the project directory.

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the main script**:
   ```bash
   python src/main.py
   ```

4. **Monitor the outputs**:
   - Review logs and outputs in the `results/` directory.

## Use Case Scenarios

This project can be particularly useful in the following scenarios:

- **Medical Research**: Identifying genes associated with breast and prostate cancer for targeted therapies.
- **Genetic Diagnostics**: Early detection of problematic genes that could lead to cancer.
- **Precision Medicine**: Tailoring treatments based on genetic data to improve patient outcomes.

## Contributing

If you would like to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a Pull Request.

We welcome contributions that improve the project, whether it's code, documentation, or ideas for new features.

## Video Demonstration

Watch a short video where we explain and demonstrate the project:

[![AI Driven Cancer Treatment Revolution](https://img.youtube.com/vi/oyuopyGIKiM/0.jpg)](https://youtu.be/oyuopyGIKiM)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
