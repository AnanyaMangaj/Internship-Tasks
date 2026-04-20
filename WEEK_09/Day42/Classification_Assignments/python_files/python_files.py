# ============================================================
# Assignment: Classification Algorithms on Two Kaggle Datasets
# Name: Keertiraj
# Subject: Machine Learning
# ============================================================

# =========================
# SECTION 1: IMPORT LIBRARIES
# =========================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ============================================================
# SECTION 2: TITANIC DATASET
# ============================================================

print("\n" + "="*60)
print("TITANIC DATASET")
print("="*60)

# Load Titanic Dataset
df = pd.read_csv("../datasets/train.csv")

print("\nFirst 5 rows of Titanic Dataset:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# -------------------------
# Preprocessing
# -------------------------

df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df['Fare'] = df['Fare'].fillna(df['Fare'].median())

df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])
df['Embarked'] = le.fit_transform(df['Embarked'])

print("\nPreprocessed Titanic Dataset:")
print(df.head())

# -------------------------
# Features and Target
# -------------------------

X = df.drop('Survived', axis=1)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------
# Logistic Regression
# -------------------------

lr = LogisticRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

print("\nTitanic - Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))
print(confusion_matrix(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))

# -------------------------
# SVM
# -------------------------

svm = SVC()
svm.fit(X_train_scaled, y_train)
y_pred_svm = svm.predict(X_test_scaled)

print("\nTitanic - SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
print(confusion_matrix(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))

# -------------------------
# Random Forest
# -------------------------

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("\nTitanic - Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# -------------------------
# Naive Bayes
# -------------------------

nb = GaussianNB()
nb.fit(X_train_scaled, y_train)
y_pred_nb = nb.predict(X_test_scaled)

print("\nTitanic - Naive Bayes Accuracy:", accuracy_score(y_test, y_pred_nb))
print(confusion_matrix(y_test, y_pred_nb))
print(classification_report(y_test, y_pred_nb))

# -------------------------
# KNN
# -------------------------

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)

print("\nTitanic - KNN Accuracy:", accuracy_score(y_test, y_pred_knn))
print(confusion_matrix(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn))

# -------------------------
# Decision Tree
# -------------------------

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

print("\nTitanic - Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))
print(confusion_matrix(y_test, y_pred_dt))
print(classification_report(y_test, y_pred_dt))

# -------------------------
# Titanic Accuracy Table
# -------------------------

results_titanic = pd.DataFrame({
    'Algorithm': ['Logistic Regression', 'SVM', 'Random Forest', 'Naive Bayes', 'KNN', 'Decision Tree'],
    'Accuracy': [
        accuracy_score(y_test, y_pred_lr),
        accuracy_score(y_test, y_pred_svm),
        accuracy_score(y_test, y_pred_rf),
        accuracy_score(y_test, y_pred_nb),
        accuracy_score(y_test, y_pred_knn),
        accuracy_score(y_test, y_pred_dt)
    ]
})

print("\nTitanic Dataset Results:")
print(results_titanic.to_string(index=False))

# ============================================================
# SECTION 3: DIABETES DATASET
# ============================================================

print("\n" + "="*60)
print("DIABETES DATASET")
print("="*60)

# Load Diabetes Dataset
df2 = pd.read_csv("../datasets/diabetes.csv")

print("\nFirst 5 rows of Diabetes Dataset:")
print(df2.head())

print("\nDataset Information:")
print(df2.info())

print("\nMissing Values:")
print(df2.isnull().sum())

# -------------------------
# Features and Target
# -------------------------

X = df2.drop('Outcome', axis=1)
y = df2['Outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------
# Logistic Regression
# -------------------------

lr = LogisticRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

print("\nDiabetes - Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))
print(confusion_matrix(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))

# -------------------------
# SVM
# -------------------------

svm = SVC()
svm.fit(X_train_scaled, y_train)
y_pred_svm = svm.predict(X_test_scaled)

print("\nDiabetes - SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
print(confusion_matrix(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))

# -------------------------
# Random Forest
# -------------------------

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("\nDiabetes - Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# -------------------------
# Naive Bayes
# -------------------------

nb = GaussianNB()
nb.fit(X_train_scaled, y_train)
y_pred_nb = nb.predict(X_test_scaled)

print("\nDiabetes - Naive Bayes Accuracy:", accuracy_score(y_test, y_pred_nb))
print(confusion_matrix(y_test, y_pred_nb))
print(classification_report(y_test, y_pred_nb))

# -------------------------
# KNN
# -------------------------

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)

print("\nDiabetes - KNN Accuracy:", accuracy_score(y_test, y_pred_knn))
print(confusion_matrix(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn))

# -------------------------
# Decision Tree
# -------------------------

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

print("\nDiabetes - Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))
print(confusion_matrix(y_test, y_pred_dt))
print(classification_report(y_test, y_pred_dt))

# -------------------------
# Diabetes Accuracy Table
# -------------------------

results_diabetes = pd.DataFrame({
    'Algorithm': ['Logistic Regression', 'SVM', 'Random Forest', 'Naive Bayes', 'KNN', 'Decision Tree'],
    'Accuracy': [
        accuracy_score(y_test, y_pred_lr),
        accuracy_score(y_test, y_pred_svm),
        accuracy_score(y_test, y_pred_rf),
        accuracy_score(y_test, y_pred_nb),
        accuracy_score(y_test, y_pred_knn),
        accuracy_score(y_test, y_pred_dt)
    ]
})

print("\nDiabetes Dataset Results:")
print(results_diabetes.to_string(index=False))

# ============================================================
# SECTION 4: FINAL COMPARISON
# ============================================================

print("\n" + "="*60)
print("FINAL COMPARISON")
print("="*60)

print("\nTitanic Dataset Accuracy Comparison:")
print(results_titanic.to_string(index=False))

print("\nDiabetes Dataset Accuracy Comparison:")
print(results_diabetes.to_string(index=False))

# ============================================================
# SECTION 5: CONCLUSION
# ============================================================

print("\n" + "="*60)
print("CONCLUSION")
print("="*60)

print("""
In this assignment, classification algorithms were applied on two Kaggle datasets:
1. Titanic Dataset
2. Diabetes Dataset

The following classification algorithms were used:
- Logistic Regression
- SVM
- Random Forest
- Naive Bayes
- KNN
- Decision Tree

The models were trained and tested, and their performance was evaluated
using accuracy score, confusion matrix, and classification report.

This assignment helped in understanding how different classification
algorithms perform on different datasets.
""")