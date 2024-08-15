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
