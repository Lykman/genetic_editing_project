# Genetic Editing Project

## Overview

This project leverages advanced AI techniques (AI USM) and CRISPR-Cas9 technology to analyze genetic data, identify genes associated with cancer, and simulate genetic editing interventions. The project includes data preprocessing, machine learning model training, target gene identification, and genetic editing simulations.

## Structure

- **`data/`** - Directory containing datasets:
  - `cleaned_breast_cancer_data.csv` - Cleaned dataset for breast cancer.
  - `cleaned_prostate_cancer_data.csv` - Cleaned dataset for prostate cancer.
  - Other raw datasets as necessary.

- **`src/`** - Directory containing source code:
  - `clean_data.py` - Script for data cleaning.
  - `data_loader.py` - Script for data loading and preprocessing.
  - `enhanced_nn.py` - Script for training and evaluating enhanced neural networks.
  - `ensemble_models.py` - Script for training and evaluating ensemble models.
  - `gene_editor.py` - Script for gene identification and CRISPR-Cas9 edits.
  - `model_trainer.py` - Script for training machine learning models.
  - `monitor.py` - Script for monitoring post-genetic editing.
  - `plot_roc_curves.py` - Script for plotting ROC curves.
  - `main.py` - Main script to execute the entire workflow.

## Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/genetic-editing-project.git
   cd genetic-editing-project
   ```

2. **Install the dependencies**:
   Ensure Python is installed. Then, use pip to install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the project**:
   Start by cleaning the data and training the models:
   ```bash
   python src/clean_data.py
   python src/model_trainer.py
   ```

4. **Monitor the results**:
   Track model performance and other results using:
   ```bash
   python src/monitor.py
   ```

5. **Simulate Genetic Edits**:
   Execute genetic editing simulations with CRISPR-Cas9:
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
   Review logs and outputs located in the `results/` directory.

## Use Case Scenarios

The Genetic Editing Project is applicable in the following areas:

- **Medical Research**: Target gene identification for breast and prostate cancer therapies.
- **Genetic Diagnostics**: Early detection of genes predisposed to cancer.
- **Precision Medicine**: Customizing treatments based on individual genetic profiles.

## Contributing

To contribute, please:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a Pull Request.

Contributions are welcomed, including code improvements, documentation, and new feature suggestions.

## Video Demonstration

Watch the project overview and demonstration on YouTube:

[![AI Driven Cancer Treatment Revolution](https://img.youtube.com/vi/oyuopyGIKiM/0.jpg)](https://youtu.be/oyuopyGIKiM)

## Visual Overview

Here’s a quick visual representation of the genetic editing process using our system:

![DNA Rotation](path/to/your/dna_gif.gif)

*Note: Replace `path/to/your/dna_gif.gif` with the actual path to your GIF in the repository.*

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

