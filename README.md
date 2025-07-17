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

## Features

Key features include:
- **Radius, Texture, Perimeter, Area, Smoothness, Compactness, Concavity, Symmetry, Fractal Dimension** (mean, standard error, worst).
- Target: **diagnosis** (malignant=1, benign=0).

---

## Project Structure

- `Breast cancer detection.ipynb`: Main notebook with code, EDA, preprocessing, modeling, and evaluation.
- `data.csv`: Input data file (not included here).
- `README.md`: Project documentation.

---

## Installation & Requirements

This project requires Python 3.x and the following libraries:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

**Install dependencies:**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
