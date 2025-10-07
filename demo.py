# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

## 1: Load your dataset and explore

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('Titanic_test.csv')
df = pd.read_csv('Titanic_train.csv')

st.title("🚢 Titanic Survival Prediction - Logistic Regression")

st.subheader("Sample Data")
st.write(df.head())

## 2: Explore features and statistics

# Show column names and their data types
st.subheader("Data Info")
st.text(str(df.info()))

# Show basic statistics (mean, std, min, max, etc.)
st.subheader("Summary Statistics")
st.write(df.describe())

# Check for missing values
st.subheader("Missing Values")
st.write(df.isnull().sum())

# If we want to visualize feature distributions
st.subheader("Feature Distributions")
fig, ax = plt.subplots(figsize=(10, 8))
df.hist(ax=ax)
st.pyplot(fig)

# Checking the features and their types, if there are missing values
st.subheader("Data Types & Summary")
st.text(str(df.info()))
st.write(df.describe())

# Identify missing values again
st.subheader("Missing Values Check")
st.write(df.isnull().sum())

# Display column names
st.subheader("Columns")
st.write(df.columns)

# Plot histograms for all features
st.subheader("Histograms (All Features)")
fig, ax = plt.subplots(figsize=(10, 8))
df.hist(ax=ax)
st.pyplot(fig)

## Step 3: Data Preprocessing

# a. Handle missing values
df = df.fillna(df.mean(numeric_only=True))

# b. Encode categorical variables
df = pd.get_dummies(df, drop_first=True)

# Model building
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Assuming 'Survived' is your target variable
X = df.drop(columns=['Survived'])
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# Evaluate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report

y_pred = model.predict(X_test_scaled)

st.subheader("📊 Model Performance")
st.write("Accuracy:", accuracy_score(y_test, y_pred))
st.write("Precision:", precision_score(y_test, y_pred))
st.write("Recall:", recall_score(y_test, y_pred))
st.write("F1 Score:", f1_score(y_test, y_pred))
st.write("ROC-AUC Score:", roc_auc_score(y_test, y_pred))

st.text("Classification Report:\n" + classification_report(y_test, y_pred))

# ROC Curve
from sklearn.metrics import roc_curve
y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

st.subheader("ROC Curve")
fig, ax = plt.subplots()
ax.plot(fpr, tpr)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve")
st.pyplot(fig)

## Notes Section
st.subheader("Notes")
st.markdown("""
**1. Precision vs Recall**  
- Precision: Of the items predicted as “positive,” how many were actually positive?  
- Recall: Of all the actual positives, how many did the model successfully catch?  

**2. Cross-Validation**  
It’s a way to test a model multiple times by training on one part of the data and validating on another part, rotating these parts each time.  
Why it matters: It gives a more reliable performance estimate and helps prevent overfitting.
""")
