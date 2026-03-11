# Skin Cancer Detection System

A system that detects skin cancer (Melanoma) from dermoscopic images using deep learning.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### 1. Model Training

```bash
python src/train.py
```

### 2. Web Application

```bash
streamlit run app.py
```

## Dataset

This project uses the HAM10000 dataset. The dataset should be placed under the `Skin Cancer MNIST Archive` folder.

## Project Structure

```
MedicalInformatics/
├── data/raw/              # Raw data
├── models/                # Trained models
├── src/
│   ├── config.py          # Configuration
│   ├── data_loader.py     # Data loading
│   ├── preprocessing.py   # Preprocessing
│   ├── model.py           # Model architecture
│   └── train.py           # Training script
├── app.py                 # Streamlit application
└── requirements.txt
```

## Warning

This system is for educational purposes only and must not be used for medical diagnosis. Please consult a dermatologist for any real diagnosis.
