# Random Forest Classifier - Employee Attrition Prediction App

A Machine Learning web application built using **Streamlit** and **Random Forest Classifier** to predict whether an employee is likely to leave the company based on workplace and employee-related factors.

## Live Demo

https://randomforest-classifier.streamlit.app/

---

# Project Overview

This project demonstrates a complete **Machine Learning Classification workflow** including:

- Data Collection
- Data Cleaning
- Outlier Detection and Treatment
- Model Training
- Model Evaluation
- Model Deployment using Streamlit

The application predicts whether an employee is:

- Likely to Stay
- Likely to Leave

based on employee details and work-related information.

Random Forest is one of the most powerful ensemble learning algorithms used for classification and prediction tasks. :contentReference[oaicite:0]{index=0}

---

# Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Matplotlib
- Seaborn
- Pickle

---

# Machine Learning Algorithm

## Random Forest Classifier

Random Forest is an ensemble machine learning algorithm that combines multiple Decision Trees to improve prediction accuracy and reduce overfitting.

The algorithm works using:
- Bootstrap Sampling
- Multiple Decision Trees
- Majority Voting

In this project:
- `1` → Employee Likely to Leave
- `0` → Employee Likely to Stay

Random Forest is highly effective because it combines predictions from multiple trees for better stability and performance. :contentReference[oaicite:1]{index=1}

---

# Random Forest Formula

Core Random Forest concept:

:contentReference[oaicite:2]{index=2}

Where:
- \(T_i(x)\) = prediction from individual tree
- \(N\) = total number of trees

---

# Dataset Information

The dataset contains:

- 5000 rows
- 8 columns
- Balanced classes

## Features

| Feature | Description |
|---|---|
| Age | Employee age |
| Salary | Annual salary |
| YearsAtCompany | Years in company |
| JobSatisfaction | Satisfaction level |
| WorkHours | Weekly work hours |
| Promotions | Number of promotions |
| DistanceFromHome | Distance from office |
| Attrition | Target variable |

---

# Project Workflow

## 1. Data Preprocessing

- Removed duplicate rows
- Checked missing values
- Statistical analysis using `describe()`

---

## 2. Outlier Detection using IQR

Outliers were detected using the IQR (Interquartile Range) method.

Formula:

:contentReference[oaicite:3]{index=3}

Lower Bound:

:contentReference[oaicite:4]{index=4}

Upper Bound:

:contentReference[oaicite:5]{index=5}

---

## 3. Outlier Treatment

Detected outliers were treated by replacing extreme values using lower and upper bounds with NumPy operations.

---

## 4. Train-Test Split

Dataset split:
- 80% Training
- 20% Testing

---

## 5. Model Training

Used:

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    random_state=42
)
