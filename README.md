# From Signal to Symphony: Predicting Protein Function with a Deep Learning Fusion Model on Sonified Sequences

## Project Structure

### 1. Core Models

These are the main deep learning models for protein function classification.

-   **`tiny.py`**: A fusion model for **9-class protein function prediction**. It uses a `ConvNeXt-Tiny` backbone to extract visual features from spectrograms and combines them with MFCC (Mel-Frequency Cepstral Coefficients) audio features using a gated attention mechanism.
-   **`care_tiny.py`**: An advanced model for **7-class enzyme commission (EC) number prediction**. It employs a CRNN (Convolutional Recurrent Neural Network) architecture, processing MFCCs as a time series with a GRU layer before fusing them with visual features.

### 2. Data Preprocessing & Sonification

Scripts responsible for converting FASTA protein sequences into spectrogram images.

-   **`ftow.py`**: The primary script for converting FASTA files to WAV audio files and then to spectrograms.
-   **`ftow_care.py` / `ftow_ours.py`**: Variants of the sonification script, likely tailored for the specific data used in the `care_tiny.py` and `tiny.py` models respectively.
-   **`care_data/process_data.py`**: Script for processing raw data related to the enzyme classification task.

### 3. Baseline Models

The `baseline/` directory contains scripts for baseline machine learning and simpler deep learning models used for comparison.

-   `baseline/baseline_ml.py`: Implements traditional machine learning models (e.g., RandomForest, XGBoost).
-   `baseline/baseline_dl.py`: A simpler deep learning fusion model.
-   `baseline/1D_CNN.py`: A 1D CNN model that likely operates directly on sequence-derived data (e.g., hydrophobicity).
-   `baseline/ESM.py`: Scripts related to using the ESM (Evolutionary Scale Modeling) protein language model.

### 4. Specific Applications & Analysis

-   **`GFP/`**: Contains scripts for a generative task. This part of the project uses the learned features to guide the generation of novel Green Fluorescent Protein (GFP) variants.
    -   `GFP/generate_data.py`: Generates data for the GFP task.
    -   `GFP/GFP.py`: The main script for the GFP generation model.
    -   `GFP/plddt.py`: Analyzes pLDDT scores (a measure of protein structure prediction confidence).
-   **`Tonnetz.py`**: Extracts Tonnetz harmonic features from the generated audio files to analyze the relationship between musical harmony and protein biochemical properties.
-   **`musicscoresplot/`**: Scripts for visualizing musical scores from protein data.

### 5. Utilities

-   **`misc/`**: A collection of utility scripts for tasks like counting sequences (`count_seq.py`, `count_fasta.py`).

## Pre-trained Weights

-   **9-Class Protein Function Model (`tiny.py`):**
    You can download the pre-trained weights from the following link:
    [Download Weights from Google Drive](https://drive.google.com/file/d/1rbiEnmT0AoNNHP-Ha8d7AKlhq_UMVT_-/view?usp=sharing)

