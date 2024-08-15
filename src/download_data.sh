#!/bin/bash

# Create data directory if it doesn't exist
mkdir -p data

# Download datasets from Kaggle
echo "Downloading Lung Cancer Dataset..."
kaggle datasets download -d uciml/lung-cancer -p data/ --unzip

echo "Downloading Colorectal Cancer Dataset..."
kaggle datasets download -d sanjoybhusal/colorectal-cancer-dataset -p data/ --unzip

echo "Downloading Prostate Cancer Dataset..."
kaggle datasets download -d dougmcdonald/prostate-cancer-dataset -p data/ --unzip

echo "Downloading Breast Cancer Wisconsin Dataset..."
kaggle datasets download -d uciml/breast-cancer-wisconsin-data -p data/ --unzip

echo "All datasets downloaded and unzipped in the data/ directory."
