# **From Signal to Symphony: Predicting Protein Function with a Deep Learning Fusion Model on Sonified Sequences**

## Dataset Source：

| **Types of Materials**  | **Sources**       | **URLs**                      |
| ----------------------- | ----------------- | ----------------------------- |
| Protein FASTA Sequences | NCBI Database     | https://www.ncbi.nlm.nih.gov/ |
|                         | UniProt Database  | https://www.uniprot.org/      |
| Kcat and km             | SABIO-RK Database | http://sabio.h-its.org/       |
|                         | EBI Database      | https://www.ebi.ac.uk/        |

## 1. ftow.py

**Function**: Converts amino acid FASTA sequences into MIDI files, then to WAV audio files, and finally generates spectrogram images from the audio.

## 2. baseline_ml.py

**Function**: Implements baseline machine learning models. It can extract features from protein sequences (via FFT on hydrophobicity) or from spectrograms (via mean MFCCs) and train models like RandomForest and XGBoost.

## 3. baseline_dl.py

**Function**: Implements a deep learning fusion model. It uses a pre-trained CNN (e.g., ConvNeXt) for visual features from spectrograms and fuses them with mean MFCC features using an attention mechanism.

## 4. dlforcare.py

**Function**: Implements an advanced, two-phase deep learning pipeline. It treats MFCCs as a time series, processing them with a GRU and an audio attention layer, then fuses these advanced audio features with CNN-based visual features for a more sophisticated classification.

## 5. Tonnetz.py

**Function**: Extracts Tonnetz features from WAV files and calculates the Pearson correlation coefficient against kinetic parameters (e.g., kcat/km) to analyze the relationship between musical harmony and protein function.

#### **Introduction:**

Proteins are fundamental to life, with their function determined by a complex interplay of sequence and structure. Predicting function from sequence remains a central challenge. This study introduces a systematic, multi-stage approach that demonstrates the power of evolving both data representation and model complexity for protein function classification. We curated a dataset of nine protein classes and first established a baseline using a Fast Fourier Transform (FFT) on 1D hydrophobicity profiles, achieving a modest accuracy of 75.28%. We then advanced the representation by translating protein information into 2D spectrograms via musical sonification, which, when combined with engineered Mel-Frequency Cepstral Coefficient (MFCC) features, significantly improved accuracy to 81.17%. Finally, an end-to-end deep learning model fusing pre-trained ConvNeXt visual features with MFCCs via an attention mechanism achieved a high accuracy of 90.44%, establishing a new state-of-the-art for audio-based protein function classification. The ultimate validation of our encoding, however, is its application in generative design. We demonstrate that the learned 'musical features' are not just correlational but have enough predictive power to guide protein engineering. By integrating our framework into a conditional diffusion model, we successfully generated novel, viable Green Fluorescent Protein (GFP) variants, showcasing its utility as a powerful tool for both protein analysis and design.

