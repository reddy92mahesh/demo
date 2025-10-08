import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pickle

# --- Initial Setup and Data Loading ---
# Suppress warnings
warnings.filterwarnings('ignore')

# Load the datasets
# NOTE: Assuming 'Titanic_train.csv' is in the execution directory
try:
    train_data = pd.read_csv('Titanic_train.csv')
    # test_data = pd.read_csv('Titanic_test.csv') # Not strictly needed for this part of the analysis
except FileNotFoundError:
    print("Error: 'Titanic_train.csv' not found. Please ensure the file is in the correct directory.")
    exit()

# Examine the structure
print("--- Training Data Info ---")
# print(train_data.info()) # Commented out to reduce verbose output
print("\n--- Summary Statistics ---")
# print(train_data.describe(include='all')) # Commented out to reduce verbose output

# --- Exploratory Data Analysis (EDA) ---
print("\n--- EDA Plots (Please view the generated figures) ---")

# Histogram of Age
plt.figure(figsize=(10, 5))
sns.histplot(train_data['Age'].dropna(), bins=30)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
# plt.show() # Commented out for environment where plots don't display inline

# Box plot of Fare
plt.figure(figsize=(10, 5))
sns.boxplot(x='Pclass', y='Fare', data=train_data)
plt.title('Fare by Passenger Class')
# plt.show() # Commented out for environment where plots don't display inline

# Pair plot of selected features
# sns.pairplot(train_data[['Survived', 'Pclass', 'Sex', 'Age']], hue='Survived')
# plt.show() # Commented out for environment where plots don't display inline

# --- Data Preprocessing ---

# Fill missing Age values with the median
train_data['Age'].fillna(train_data['Age'].median(), inplace=True)

# Fill missing Embarked values with the mode
train_data['Embarked'].fillna(train_data['Embarked'].mode()[0], inplace=True)

# Convert 'Sex' to binary (male: 0, female: 1)
train_data['Sex'] = train_data['Sex'].map({'male': 0, 'female': 1})

# One-hot encode 'Embarked'
train_data = pd.get_dummies(train_data, columns=['Embarked'], prefix='Embarked', drop_first=True)

# Features and target variable
# NOTE: The one-hot encoding changed the feature names to 'Embarked_Q' and 'Embarked_S'
X = train_data[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked_Q', 'Embarked_S']]
y = train_data['Survived']
feature_names = X.columns # Store feature names before splitting

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- Model Training with Scaling and Pipeline ---

# Create a pipeline for scaling numerical features and training the model
# Scaling Age and Fare is essential for Logistic Regression for reliable convergence
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression(random_state=42, solver='liblinear')) # Use 'liblinear' for good performance on small datasets
])

# Train the model
pipeline.fit(X_train, y_train)

# Extract the trained model from the pipeline
model = pipeline.named_steps['logreg']
scaler = pipeline.named_steps['scaler']

# --- Model Evaluation ---

# Predictions
y_pred = pipeline.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1])

print("\n--- Model Performance Metrics ---")
print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')
print(f'ROC AUC Score: {roc_auc:.2f}')

# ROC Curve Plot
fpr, tpr, thresholds = roc_curve(y_test, pipeline.predict_proba(X_test)[:, 1])
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
# plt.show() # Commented out for environment where plots don't display inline

# --- Coefficient Analysis ---

# Get the coefficients of the scaled model
# The coefficients are for the *scaled* features
coefficients = model.coef_[0]

# Create a DataFrame to display the coefficients with their corresponding features
coefficients_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})

# Sort the coefficients by magnitude
coefficients_df = coefficients_df.sort_values(by='Coefficient', ascending=False)

print("\n--- Feature Significance (Coefficients of Scaled Features) ---")
print(coefficients_df)

print("\nInterpretation:")
print("Positive coefficients (e.g., Sex) strongly suggest a positive relationship with survival.")
print("Negative coefficients (e.g., Pclass, Age) strongly suggest a negative relationship with survival.")
print("The magnitude of the coefficient indicates the strength of the impact.")

# --- Model Saving ---

# Save the trained pipeline (which includes the scaler and the model)
# Saving the pipeline is better practice than saving only the model, as it ensures
# that new data is scaled identically before prediction.
filename = 'logistic_model_pipeline.pkl'
pickle.dump(pipeline, open(filename, 'wb'))

print(f"\nModel successfully saved to '{filename}' (it includes the necessary scaler).")
