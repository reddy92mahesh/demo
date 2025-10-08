import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import pickle
import streamlit as st # Imported for context in the app.py file
# import localtunnel # Local tunnel is typically installed via npm, not pip/python module

# Suppress warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# 1. Data Exploration
# ==============================================================================

# Load the datasets
# Assuming 'Titanic_train.csv' and 'Titanic_test.csv' are available in the directory
try:
    train_data = pd.read_csv('Titanic_train.csv')
    # test_data is loaded but not used in the rest of the script, so its loading is kept for completeness
    test_data = pd.read_csv('Titanic_test.csv')
except FileNotFoundError as e:
    print(f"Error: {e}. Please ensure 'Titanic_train.csv' and 'Titanic_test.csv' are in the current directory.")
    exit()

# Examine the structure
print("--- train_data.info() ---")
print(train_data.info())

# Summary statistics
print("\n--- train_data.describe(include='all') ---")
print(train_data.describe(include='all'))

# Visualizations

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

# ==============================================================================
# 2. Data Preprocessing
# ==============================================================================

# Fill missing Age values with the median
train_data['Age'].fillna(train_data['Age'].median(), inplace=True)

# Fill missing Embarked values with the mode
train_data['Embarked'].fillna(train_data['Embarked'].mode()[0], inplace=True)

# Encode Categorical Variables

# Convert 'Sex' to binary
# 'male': 0, 'female': 1
train_data['Sex'] = train_data['Sex'].map({'male': 0, 'female': 1})

# One-hot encode 'Embarked'
# drop_first=True to avoid multicollinearity (C is the dropped base category)
train_data = pd.get_dummies(train_data, columns=['Embarked'], drop_first=True)

# ==============================================================================
# 3. Model Building
# ==============================================================================

# Features and target variable
# Features used are 'Pclass', 'Sex', 'Age', 'Fare', 'Embarked_Q', 'Embarked_S'
X = train_data[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked_Q', 'Embarked_S']]
y = train_data['Survived']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the Logistic Regression model
# Increased max_iter for convergence as per common practice for LogisticRegression on such data
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# ==============================================================================
# 4. Model Evaluation
# ==============================================================================

# Predictions
y_pred = model.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

print("\n--- Model Evaluation Metrics ---")
print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')
print(f'ROC AUC Score: {roc_auc:.2f}')

# Visualize the ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()

# ==============================================================================
# 5. Interpretation
# ==============================================================================

# Get the coefficients of the model
coefficients = model.coef_[0]

# Create a DataFrame to display the coefficients with their corresponding features
feature_names = X.columns
coefficients_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})

# Sort the coefficients by magnitude
coefficients_df = coefficients_df.sort_values('Coefficient', ascending=False)

print("\n--- Model Coefficients ---")
print(coefficients_df)

# Interpretation:
print("\nInterpretation:")
print("Positive coefficients indicate that an increase in the feature is associated with an increased probability of survival (Survived=1).")
print("Negative coefficients indicate that an increase in the feature is associated with a decreased probability of survival (Survived=0).")
print("The magnitude of the coefficient represents the strength of the association.")

# Feature Significance Discussion
print("\nFeature Significance:")
print(coefficients_df)
print("\nDetailed Interpretation:")
print("The most significant positive coefficient is typically for 'Sex' (female, Sex=1), indicating a strong positive correlation with survival.")
print("A large negative coefficient is often for 'Pclass', meaning a higher class (Pclass=1) has a lower *negative* impact, which translates to a higher chance of survival, and conversely, a higher *Pclass* number (e.g., Pclass=3) has a larger negative coefficient/impact.")
print("Other features like 'Age' and 'Fare' have smaller coefficients, suggesting less individual impact compared to 'Sex' and 'Pclass'.")

# Save the trained model to a file
filename = 'logistic_model.pkl'
pickle.dump(model, open(filename, 'wb'))
print(f"\nModel saved as '{filename}' for Streamlit deployment.")


# ==============================================================================
# 6. Streamlit Application Code (app.py content for deployment)
# ==============================================================================

# This code block is typically saved as a separate file, 'app.py', for Streamlit deployment.
# We'll print it here for completeness of the notebook-to-script conversion.

streamlit_app_code = """
import streamlit as st
import pandas as pd
import pickle

# Load the trained model
try:
    model = pickle.load(open('logistic_model.pkl', 'rb'))
except FileNotFoundError:
    st.error("Model file 'logistic_model.pkl' not found. Please train and save the model first.")
    st.stop()

# Create the Streamlit app
st.title('Titanic Survival Prediction')

st.header('Enter Passenger Details')

# Input features based on the model training
pclass = st.selectbox('Passenger Class (1st=1, 2nd=2, 3rd=3)', [1, 2, 3])
sex = st.selectbox('Sex', ['Male', 'Female'])
# Use a range that makes sense for Titanic data
age = st.slider('Age', min_value=0, max_value=80, value=25)
# Use a range that makes sense for Titanic data
fare = st.number_input('Fare (Ticket Price)', min_value=0.0, value=30.0, step=1.0)
embarked_q = st.checkbox('Embarked from Queenstown (Q)')
embarked_s = st.checkbox('Embarked from Southampton (S)')
# Embarked from Cherbourg (C) is the baseline when both Q and S are false

# Create a button to make predictions
if st.button('Predict Survival'):
    # Convert sex to binary (Male: 0, Female: 1 as per training data encoding)
    sex_encoded = 0 if sex == 'Male' else 1

    # Create a DataFrame with input features
    input_data = pd.DataFrame({
        'Pclass': [pclass],
        'Sex': [sex_encoded],
        'Age': [age],
        'Fare': [fare],
        'Embarked_Q': [1 if embarked_q else 0],
        'Embarked_S': [1 if embarked_s else 0]
    })

    # Ensure feature order matches the training data
    input_data = input_data[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked_Q', 'Embarked_S']]

    # Make prediction
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0][1] # Probability of survival (class 1)

    st.subheader('Prediction Result')
    # Display the prediction
    if prediction == 1:
        st.success(f'Passenger is likely to **survive**! (Probability: {prediction_proba:.2f})')
    else:
        st.error(f'Passenger is likely to **not survive**. (Probability of Survival: {prediction_proba:.2f})')

    st.caption("Note: This prediction is based on a Logistic Regression model trained on the Titanic dataset.")
"""

# The following lines are for display and do not execute the app
print("\n--- Streamlit Application Code (app.py) ---")
print(streamlit_app_code)

# ==============================================================================
# Interview Questions (for informational context)
# ==============================================================================

print("\n\n--- Interview Questions & Answers ---")

print("\n# 1. What is the difference between precision and recall?")
print("### Precision:")
print(" - Measures the proportion of correctly predicted positive instances out of all instances predicted as positive (True Positives / (True Positives + False Positives)).")
print(" - Answers: \"Of all the passengers the model predicted as survived, how many actually survived?\"")
print(" - High precision means the model has a low rate of **False Positives** (incorrectly predicting survival).")

print("### Recall:")
print(" - Measures the proportion of correctly predicted positive instances out of all actual positive instances (True Positives / (True Positives + False Negatives)).")
print(" - Answers: \"Of all the passengers who actually survived, how many did the model correctly predict as survived?\"")
print(" - High recall means the model has a low rate of **False Negatives** (failing to identify actual survivors).")

print("\n# 2. What is cross-validation, and why is it important in binary classification?")
print("### Cross-Validation:")
print(" - A technique to evaluate a model's performance on unseen data by splitting the dataset into multiple folds (e.g., k-fold).")
print(" - The model is trained and tested $k$ times; each time, a different fold is used as the validation set, and the rest are used for training.")

print("### Why is it Important in Binary Classification?")
print(" - **Prevents overfitting**: It provides a more accurate estimate of how well the model generalizes to new data, preventing the model from just memorizing the training set.")
print(" - **Reliable performance estimation**: It reduces the variance of the performance estimate compared to a single train-test split, leading to a more robust measure of metrics like accuracy, precision, and recall.")
print(" - **Model/Hyperparameter tuning**: It's crucial for comparing different models or tuning hyperparameters (e.g., Logistic Regression's regularization strength) to find the combination that performs best across all data subsets.")
