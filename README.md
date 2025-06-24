# Titanic Survival Prediction

## Project Overview

This project implements a machine learning model to predict the survival of passengers on the Titanic based on various features such as age, sex, passenger class, and family size. The dataset used is the Titanic dataset, and the model employs a Random Forest Classifier with hyperparameter tuning to achieve high accuracy.

## Dataset

The Titanic dataset (Titanic-Dataset.csv) contains 891 samples with the following features:

PassengerId: Unique identifier for each passenger
Survived: Target variable (0 = Did not survive, 1 = Survived)
Pclass: Passenger class (1 = 1st, 2 = 2nd, 3 = 3rd)
Name: Passenger's name
Sex: Passenger's gender
Age: Passenger's age
SibSp: Number of siblings/spouses aboard
Parch: Number of parents/children aboard
Ticket: Ticket number
Fare: Passenger fare
Cabin: Cabin number
Embarked: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

The dataset is sourced from a local file path in the provided notebook.

## Project Structure

TITANIC SURVIVAL PREDICTION.ipynb: Jupyter Notebook containing the complete code for data preprocessing, feature engineering, model training, evaluation, and visualization.

Titanic-Dataset.csv: The dataset file (not included in this repository; ensure it is available in your local environment or update the file path in the notebook).

## Requirements
To run the notebook, install the following Python libraries:
pip install pandas numpy matplotlib seaborn scikit-learn

## Methodology
### Data Loading:
The Titanic dataset is loaded using pandas.

### Exploratory Data Analysis (EDA):
Dataset shape, data types, missing values, and summary statistics are analyzed.

### Visualizations:
include survival count, survival by passenger class, gender, and age distribution.

### Feature Engineering:
Extracted titles from names and grouped rare titles (e.g., Lady, Countess) into a 'Rare' category.
Created FamilySize (SibSp + Parch + 1) and IsAlone (1 if FamilySize = 1, else 0) features.
Binned Age into 5 categories (0-4) based on age ranges.
Binned Fare into 4 categories (0-3) using quartiles.
Dropped unnecessary columns: PassengerId, Name, Ticket, Cabin, AgeBand, FareBand.

### Data Preprocessing:
Numeric features (Age, Fare, SibSp, Parch, FamilySize) are imputed with median values and scaled using StandardScaler.
Categorical features (Pclass, Sex, Embarked, Title, IsAlone) are imputed with the most frequent value and one-hot encoded.
Data is split into training (80%) and testing (20%) sets with stratification.

### Model Training:
A Random Forest Classifier is trained within a pipeline that includes preprocessing steps.
Hyperparameter tuning is performed using GridSearchCV with parameters: n_estimators (100, 200), max_depth (None, 5, 10), and min_samples_split (2, 5).

### Evaluation:
Model accuracy is calculated using accuracy_score.
A classification report (precision, recall, F1-score) is generated.
A confusion matrix shows classification performance.

### Results:
Best Parameters: {'classifier__max_depth': 5, 'classifier__min_samples_split': 2, 'classifier__n_estimators': 100}
Cross-validation Accuracy: ~82.7%

### Test Set Evaluation:
Accuracy: 82.7%

### Classification Report:
precision    recall  f1-score   support
     0       0.84      0.88      0.86       110
     1       0.80      0.74      0.77        69
accuracy                           0.83       179

### Confusion Matrix:

[[97 13]
 [18 51]]
