# Machine Learning-Based Detection of Upper Respiratory Tract Infection Using COUGHVID Audio Data

## Project Overview

This project investigates the use of deep learning models—specifically Convolutional Neural Networks (CNNs) and Long Short-Term Memory networks (LSTMs)—to detect Upper Respiratory Tract Infections (URTIs) using cough audio recordings. The work draws on the publicly available COUGHVID dataset and incorporates methodologies and architectural insights from the paper _“Attention-based Hybrid CNN-LSTM and Spectral Data Augmentation for COVID-19 Diagnosis from Cough Sound”_ (Hamdi et al., 2022) [Available from](https://link.springer.com/article/10.1007/s10844-022-00707-7).

Our approach focused on adapting and extending these techniques for URTI detection, which, to date, has received relatively limited attention in machine learning literature.

## Motivations

Upper Respiratory Tract Infections (URTIs) are the leading cause of acute illness globally and account for approximately 83% of all respiratory tract infection-related hospital admissions in the UK. Despite the majority (82%) of URTIs being viral in origin—where antibiotics are ineffective—antibiotic prescribing remains widespread and often unnecessary. Notably, 41% of patients prescribed antibiotics for URTI have no clinical indication for them.

Bacterial URTIs typically resolve without the need for antibiotics, and overprescribing contributes significantly to the rise of antimicrobial resistance. Moreover, there are no machine learning algorithms currently developed or implemented solely for URTI detection, representing a critical gap in digital diagnostic tools.

The development of an automated URTI detection algorithm could:
- Prevent inappropriate antibiotic prescriptions
- Address growing antimicrobial resistance
- Reduce unnecessary economic burden on healthcare systems
- Alleviate operational pressure on general practitioners and the NHS

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

To assess model robustness in a clinically relevant setting, we evaluated performance on two subsets:

1. The **full dataset**, which includes both healthy and unhealthy individuals, provides a broad test of general classification ability.
2. The **unhealthy subset**, which includes only individuals with some form of illness (but not necessarily URTI), isolates the model’s ability to distinguish URTI from other pathological coughs—arguably a more challenging and realistic diagnostic task.

| Model                    | AUC  | Sensitivity | Specificity | Precision | Accuracy | F1 Score |
|--------------------------|------|-------------|-------------|-----------|----------|----------|
| CNN (Full Dataset)       | 0.93 | 0.87        | 0.86        | 0.83      | 0.87     | 0.85     |
| CNN (Unhealthy Subset)   | 0.90 | 0.79        | 0.84        | 0.80      | 0.82     | 0.80     |
| LSTM (Full Dataset)      | 0.77 | 0.50        | 0.94        | 0.87      | 0.72     | 0.64     |
| LSTM (Unhealthy Subset)  | 0.75 | 0.52        | 0.88        | 0.77      | 0.70     | 0.62     |

These results indicate that the CNN model performs with clinical-grade accuracy in both sensitivity and specificity, while the LSTM model acts as a conservative verifier—excellent at ruling out false positives but weaker in identifying true positives.

## References

Hamdi, S., Oussalah, M., Moussaoui, A., & Saidi, M. (2022). _Attention-based hybrid CNN-LSTM and spectral data augmentation for COVID-19 diagnosis from cough sound_. Journal of Intelligent Information Systems, 59, 367–389. [Available from](https://doi.org/10.1007/s10844-022-00707-7)

Ashworth M, Charlton J, Ballard K, Latinovic R, Gulliford M. (2005). _Variations in antibiotic prescribing and consultation rates for acute respiratory infection in UK general practices 1995–2000_. Br J Gen Pract. 55(517):603–8. [Available from](https://pmc.ncbi.nlm.nih.gov/articles/PMC1463221/)

Jackson C, Lawes T, Smith R, et al. (2023). _Increasing burden of antimicrobial resistance in respiratory tract infections in primary care: a retrospective cohort study_. JAC Antimicrob Resist. 5(1):dlad012. [Available from](https://academic.oup.com/jacamr/article/5/1/dlad012/7034538)

National Institute for Health and Clinical Excellence (NICE). (2008). _Respiratory Tract Infections - Antibiotic Prescribing: Prescribing of Antibiotics for Self-Limiting Respiratory Tract Infections in Adults and Children in Primary Care_. NICE Clinical Guidelines No. 69. [Available from](https://www.ncbi.nlm.nih.gov/books/NBK53632/)

Leung TI, Abdelnabi M, Henao-Martínez AF, Beckham JD, Tyler KL, Mejia R. (2021). _Burden of respiratory infections during the COVID-19 pandemic in the United States_. EClinicalMedicine. 38:101018. [Available from](https://www.thelancet.com/journals/eclinm/article/PIIS25895370%2821%2900266-2/)

Collaborators TRTI. (2024). _Global, regional, and national burden of respiratory tract infections, 1990–2019: a systematic analysis for the Global Burden of Disease Study 2019_. Lancet Infect Dis. 24(5):505–24. [Available from](https://www.thelancet.com/journals/laninf/article/PIIS1473-3099%2824%2900430-4/)

Llor C, Bjerrum L. (2014). _Antimicrobial resistance: risk associated with antibiotic overuse and initiatives to reduce the problem_. Ther Adv Drug Saf. 5(6):229–41. [Available from](https://pmc.ncbi.nlm.nih.gov/articles/PMC6323860/)

## Dataset Link

[COUGHVID Dataset – Zenodo Repository](https://zenodo.org/record/4498364)

## Additional Resources

This project supports the development of telehealth screening tools and could be adapted for mobile or web-based triage in low-resource healthcare settings.
