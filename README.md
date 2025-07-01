# Machine Learning-Based Detection of Upper Respiratory Tract Infection Using COUGHVID Audio Data

## Project Overview

This project investigates the use of deep learning models—specifically Convolutional Neural Networks (CNNs) and Long Short-Term Memory networks (LSTMs)—to detect Upper Respiratory Tract Infections (URTIs) using cough audio recordings. The work draws on the publicly available COUGHVID dataset and incorporates methodologies and architectural insights from the paper _“Attention-based Hybrid CNN-LSTM and Spectral Data Augmentation for COVID-19 Diagnosis from Cough Sound”_ (Hamdi et al., 2022) [DOI: 10.1007/s10844-022-00707-7].

Our approach focused on adapting and extending these techniques for URTI detection, which, to date, has received relatively limited attention in machine learning literature.

## Dataset

The [COUGHVID dataset](https://zenodo.org/record/4498364) is a large-scale, crowd-sourced audio repository of coughs, collected during the COVID-19 pandemic. It contains over 34,000 audio samples, along with rich metadata (e.g., COVID status, demographic data, and expert physician labels for a subset). We filtered and processed this dataset to retain only entries with expert physician diagnosis relevant to upper respiratory tract infections.

## Preprocessing Pipeline (`cough.ipynb`)

This script handles end-to-end data preparation:
- Recoding of metadata variables and NA handling
- Scaling of numerical features
- One-hot encoding of categorical variables
- k-Nearest Neighbors imputation (k=5) for missing values
- Filtering out:
  - Individuals without a physician-assessed cough
  - Participants with ambiguous gender (i.e., gender = "other")

Audio samples underwent additional preprocessing:
- Silence trimming  
- Cough detection thresholding  
- Conversion to mel-spectrograms  
- Two-stage data augmentation (pitch shifting and SpecAugment)

## Model Architecture

We implemented and compared two deep learning models:

1. **CNN**: A 4-layer convolutional model trained on mel-spectrograms using an AdamW optimizer and binary cross-entropy loss.
2. **LSTM**: A 2-layer LSTM model designed to capture temporal dependencies from reshaped spectrogram inputs.

Both models were validated using 10-fold cross-validation on an 80:20 train-test split.

## Results

| Model                    | AUC  | Sensitivity | Specificity | Precision | Accuracy | F1 Score |
|--------------------------|------|-------------|-------------|-----------|----------|----------|
| CNN (Full Dataset)       | 0.93 | 0.87        | 0.86        | 0.83      | 0.87     | 0.85     |
| CNN (Unhealthy Subset)   | 0.90 | 0.79        | 0.84        | 0.80      | 0.82     | 0.80     |
| LSTM (Full Dataset)      | 0.77 | 0.50        | 0.94        | 0.87      | 0.72     | 0.64     |
| LSTM (Unhealthy Subset)  | 0.75 | 0.52        | 0.88        | 0.77      | 0.70     | 0.62     |

These results indicate that the CNN model performs with clinical-grade accuracy in both sensitivity and specificity, while the LSTM model acts as a conservative verifier—excellent at ruling out false positives but weaker in identifying true positives.

## References

Hamdi, S., Oussalah, M., Moussaoui, A., & Saidi, M. (2022). _Attention-based hybrid CNN-LSTM and spectral data augmentation for COVID-19 diagnosis from cough sound_. Journal of Intelligent Information Systems, 59, 367–389. [DOI: 10.1007/s10844-022-00707-7](https://doi.org/10.1007/s10844-022-00707-7)

## Dataset Link

[COUGHVID Dataset – Zenodo Repository](https://zenodo.org/record/4498364)

## Additional Resources

This project supports the development of telehealth screening tools and could be adapted for mobile or web-based triage in low-resource healthcare settings.
