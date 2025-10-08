import pandas as pd

# Load the datasets
train_data = pd.read_csv('Titanic_train.csv')
test_data = pd.read_csv('Titanic_test.csv')

# Examine the structure
print(train_data.info())

# Summary statistics
print(train_data.describe(include='all'))

import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Histogram of Age
plt.figure(figsize=(10, 5))
sns.histplot(train_data['Age'].dropna(), bins=30)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Box plot of Fare
plt.figure(figsize=(10, 5))
sns.boxplot(x='Pclass', y='Fare', data=train_data)
plt.title('Fare by Passenger Class')
plt.show()

# Pair plot of selected features
sns.pairplot(train_data[['Survived', 'Pclass', 'Sex', 'Age']], hue='Survived')
plt.show()

# Fill missing Age values with the median
train_data['Age'].fillna(train_data['Age'].median(), inplace=True)

# Drop or fill other missing values as necessary
train_data['Embarked'].fillna(train_data['Embarked'].mode()[0], inplace=True)

# Convert 'Sex' to binary
train_data['Sex'] = train_data['Sex'].map({'male': 0, 'female': 1})

# One-hot encode 'Embarked'
train_data = pd.get_dummies(train_data, columns=['Embarked'], drop_first=True)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Features and target variable
X = train_data[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked_Q', 'Embarked_S']]
y = train_data['Survived']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Predictions
y_pred = model.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')

from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()


# Get the coefficients of the model
coefficients = model.coef_[0]

# Create a DataFrame to display the coefficients with their corresponding features
feature_names = X.columns
coefficients_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})

# Sort the coefficients by magnitude
coefficients_df = coefficients_df.sort_values('Coefficient', ascending=False)

print(coefficients_df)

# Interpretation:

# Positive coefficients indicate that an increase in the feature is associated with an increased probability of survival.
# Negative coefficients indicate that an increase in the feature is associated with a decreased probability of survival.
# The magnitude of the coefficient represents the strength of the association.

# For example, if the coefficient for 'Sex' is positive and large, it means being female is strongly associated with survival.

# im analyze the coefficients to understand which factors had the most impact on the model's prediction of survival.


# Feature Significance Discussion

# Based on the coefficients obtained from the logistic regression model, i  understand which features were most significant in predicting survival.

# Positive Coefficients:
#   - Sex: Being female (Sex=1) has a strong positive impact on survival. This is as expected, as women and children were prioritized during evacuation.
#   - Embarked_S:  Embarking from Southampton (Embarked_S=1) might have a slight positive impact, although it's less significant.
#
# Negative Coefficients:
#   - Pclass: Higher passenger class (lower Pclass number) has a positive correlation with survival. This indicates that wealthier passengers had better chances of surviving.
#   - Age: Higher age appears to have a slight negative correlation with survival, although it's not very strong.
#   - Fare: A higher fare might have a slightly negative impact, although it's subtle.
#   - Embarked_Q: Embarking from Queenstown (Embarked_Q=1) has a negative effect on survival probability.


# Print the Significance
print("\nFeature Significance:")
print(coefficients_df)
print("\nInterpretation:")
print("Positive coefficients suggest a positive relationship with survival.")
print("Negative coefficients suggest a negative relationship with survival.")

# save the model with this name "logistic_model.pkl" for streamlit integration

import pickle

# Save the trained model to a file
filename = 'logistic_model.pkl'
pickle.dump(model, open(filename, 'wb'))
