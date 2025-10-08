import streamlit as st
import pandas as pd
import pickle

# --- 1. Load the Trained Model ---
# This section is the most crucial for app startup.
# We use try-except to handle the case where the model file is missing.
try:
    # Load the trained model. Ensure 'logistic_model.pkl' is in the same directory.
    model = pickle.load(open('logistic_model.pkl', 'rb'))
except FileNotFoundError:
    st.error("Error: Model file 'logistic_model.pkl' not found.")
    st.markdown("Please ensure the trained model is saved as `logistic_model.pkl` and uploaded to the deployment server.")
    st.stop() # Stop the app execution if the model cannot be loaded

# --- 2. Streamlit App Layout ---
st.set_page_config(page_title="Titanic Survival Predictor", layout="centered")
st.title('🚢 Titanic Survival Prediction')
st.markdown('Use the inputs below to predict if a passenger would have survived the sinking of the Titanic, based on a Logistic Regression model.')

st.header('Enter Passenger Details')

# --- 3. User Input Fields ---
# Input features based on the model training: Pclass, Sex, Age, Fare, Embarked_Q, Embarked_S

# Pclass (Passenger Class)
pclass = st.selectbox(
    'Passenger Class',
    options=[1, 2, 3],
    format_func=lambda x: f"{x} (Class)"
)

# Sex
sex = st.radio(
    'Sex',
    options=['Male', 'Female']
)

# Age
age = st.slider(
    'Age',
    min_value=0,
    max_value=80,
    value=25,
    step=1
)

# Fare
fare = st.number_input(
    'Fare (Ticket Price in $)',
    min_value=0.0,
    value=30.0,
    step=0.5
)

st.subheader('Port of Embarkation')
st.caption("Select the port(s) from which the passenger embarked. Cherbourg (C) is the baseline.")

# Embarked (One-Hot Encoded: Embarked_Q, Embarked_S)
col1, col2 = st.columns(2)
with col1:
    embarked_q = st.checkbox('Queenstown (Q)')
with col2:
    embarked_s = st.checkbox('Southampton (S)')


# --- 4. Prediction Logic ---
if st.button('Predict Survival'):
    
    # 1. Convert categorical inputs to the model's expected numeric format
    # Sex: 'male': 0, 'female': 1 (as per the notebook preprocessing)
    sex_encoded = 1 if sex == 'Female' else 0

    # 2. Create a DataFrame with the correct features and order
    # The feature order MUST match the order used during training: 
    # ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked_Q', 'Embarked_S']
    input_data = pd.DataFrame({
        'Pclass': [pclass],
        'Sex': [sex_encoded],
        'Age': [age],
        'Fare': [fare],
        'Embarked_Q': [1 if embarked_q else 0],
        'Embarked_S': [1 if embarked_s else 0]
    })
    
    # Optional: Display the processed input data for debugging
    # st.write("Model Input:", input_data) 

    # 3. Make Prediction
    prediction = model.predict(input_data)[0]
    # Get probability of survival (class 1)
    prediction_proba = model.predict_proba(input_data)[0][1] 

    # 4. Display the result
    st.subheader('Result')
    if prediction == 1:
        st.balloons()
        st.success(f'**✅ Prediction: Passenger is likely to survive!**')
        st.progress(prediction_proba, text=f"Survival Probability: {prediction_proba:.2f}")
    else:
        st.error(f'**❌ Prediction: Passenger is likely to not survive.**')
        st.progress(prediction_proba, text=f"Survival Probability: {prediction_proba:.2f}")

st.markdown("---")
st.caption("Disclaimer: This is a prediction based on a simplified Logistic Regression model. It does not guarantee accuracy.")
