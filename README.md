# Customer-segmentation-using-cluster
Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import warnings
warnings.filterwarnings("ignore")


Load Dataset

data = pd.read_excel("WA_Fn-UseC_-Telco-Customer-Churn.xlsx")


print("Dataset Loaded Successfully")
data.head()


Basic Data Inspection


print("Dataset Shape:", data.shape)
print("\nColumn Names:\n", data.columns)
print("\nMissing Values:\n", data.isnull().sum())



Data Cleaning

if "customerID" in data.columns:
    data.drop("customerID", axis=1, inplace=True)

if "TotalCharges" in data.columns:
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")

data.fillna(data.median(numeric_only=True), inplace=True)

print("Data Cleaning Completed")



Encode Target Column

le = LabelEncoder()
data["Churn"] = le.fit_transform(data["Churn"])

data.head()


Convert Categorical Variables

data = pd.get_dummies(data, drop_first=True)

print("Categorical Encoding Completed")
data.head()

Exploratory Data Analysis

sns.countplot(x="Churn", data=data)
plt.title("Churn Distribution")
plt.show()

plt.figure(figsize=(12,8))
sns.heatmap(data.corr(), cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

Feature Selection

X = data.drop("Churn", axis=1)
y = data["Churn"]

print("Feature Selection Completed")

Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train-Test Split Completed")


Model Building (Logistic Regression)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("Model Training Completed")


Prediction


y_pred = model.predict(X_test)

print("Prediction Completed")



Model Evaluation


accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
