# Breast Cancer Detection using Machine Learning

This project presents a comprehensive machine learning workflow to detect breast cancer from clinical data using ensemble models. The workflow includes data exploration, preprocessing, visualization, feature selection, model training, and evaluation of multiple classifiers.

---


## Project Overview

The goal of this project is to classify tumors as **malignant** or **benign** based on a set of clinical features derived from digitized images of a fine needle aspirate (FNA) of a breast mass. Various machine learning models including K-Nearest Neighbors, Random Forest, and Support Vector Machines are evaluated for their predictive performance.

---

## Dataset

- The data used is the [Breast Cancer Wisconsin (Diagnostic) Dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data?resource=download).
- 569 samples, 32 features, and a binary diagnosis label (`M` = malignant, `B` = benign).



---

## Project Structure

- `Breast cancer detection.ipynb`: Main notebook with code, EDA, preprocessing, modeling, and evaluation.
- `data.csv`: Input data file.
- `README.md`: Project documentation.

---

## Installation & Requirements

This project requires Python 3.x and the following libraries:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

---

## How to Run

1. Download or clone this repository.
2. Place the `data.csv` file in the same directory as the notebook.
3. Open the notebook `Breast cancer detection.ipynb` in [Jupyter Notebook](https://jupyter.org/) or [Google Colab](https://colab.research.google.com/).
4. Run each cell sequentially to reproduce the analysis, starting from data exploration through modeling and results.

---

## Exploratory Data Analysis

- **Distribution analysis:** Used barplots, histograms, and boxplots to understand the data.
- **Correlation analysis:** Generated heatmaps to find features highly correlated with cancer diagnosis.
- **Data cleaning:** Removed irrelevant columns, handled missing values, and checked for duplicates.

---

## Data Preprocessing

- Encoded target labels (`M` → 1, `B` → 0).
- Selected top correlated features using tree-based feature importance.
- Normalized features using `StandardScaler`.
- Split data into training and test sets (80/20).

---

## Modeling

Trained and evaluated the following models:

- **K-Nearest Neighbors (KNN):** Hyperparameter tuning for optimal k.
- **Random Forest Classifier:** Used 1000 estimators for robust ensemble learning.
- **Support Vector Machine (SVM):** Used default kernel and parameters.

---

## Results

- **KNN Accuracy:** Up to ~94.7% (best with k=11).
- **Random Forest Accuracy:** ~92.1%
- **SVM Accuracy:** Evaluated in the notebook.

Plots, confusion matrices, and accuracy metrics for each model are included in the notebook.

