import streamlit as st
import pandas as pd
import joblib

# Load the pre-trained model
model = joblib.load('rf.sav')

# Create a function to predict churn
def predict_churn(features):
    # Preprocess the input features to match the model's requirements
    # You should have preprocessing code here to prepare the features for prediction
    # For example, you might need to one-hot encode categorical variables and scale numerical ones
    
    # Make predictions using the model
    prediction = model.predict(features)
    return prediction

# Define the Streamlit app
st.title('Churn Prediction App')

# Create sliders for input features (float values)
state = st.slider('State', 0.0, 50.0, 16.0)  # Example with default value set to 25.0
account_length = st.slider('Account Length', 0, 500, 128)
number_vmail_messages = st.slider('Number Vmail Messages', 0, 500, 25)
total_day_minutes = st.slider('Total Day Minutes', 0.0, 500.0, 265.1)
total_day_calls = st.slider('Total Day Calls', 0.0, 500.0, 110.0)
total_day_charge = st.slider('Total Day Charge', 0.0, 500.0, 45.07)
total_eve_minutes = st.slider('Total Eve Minutes', 0.0, 500.0, 197.4)
total_eve_calls = st.slider('Total Eve Calls', 0.0, 500.0, 99.0)
total_eve_charge = st.slider('Total Eve Charge', 0.0, 500.0, 16.78)
total_night_minutes = st.slider('Total Night Minutes', 0.0, 500.0, 244.7)
total_night_calls = st.slider('Total Night Calls', 0.0, 500.0, 91.0)
total_night_charge = st.slider('Total Night Charge', 0.0, 500.0, 11.01)
total_intl_minutes = st.slider('Total Intl Minutes', 0.0, 500.0, 10.0)
total_intl_calls = st.slider('Total Intl Calls', 0.0, 500.0, 3.0)
total_intl_charge = st.slider('Total Intl Charge', 0.0, 500.0, 2.7)
number_customer_service_calls = st.slider('Number Customer Service Calls', 0, 500, 1)
total_nat_minutes = st.slider('Total Nat Minutes', 0.0, 1000.0, 707.2)
total_nat_calls = st.slider('Total Nat Calls', 0.0, 500.0, 300.0)
total_nat_charge = st.slider('Total Nat Charge', 0.0, 500.0, 72.86)

# Create dropdowns for categorical features
international_plan = st.selectbox('International Plan', [0, 1])
voice_mail_plan = st.selectbox('Voice Mail Plan', [0, 1])
area_code = st.selectbox('Area Code', [408, 415, 510])

# Create a button to make predictions
if st.button('Predict Churn'):
    # Prepare the input features as a dictionary
    input_data = {
        'state': state,
        'account_length': account_length,
        'area_code': area_code,
        'international_plan': international_plan,
        'voice_mail_plan': voice_mail_plan,
        'number_vmail_messages': number_vmail_messages, 
        'total_day_minutes': total_day_minutes, 
        'total_day_calls': total_day_calls,
        'total_day_charge': total_day_charge, 
        'total_eve_minutes': total_eve_minutes, 
        'total_eve_calls': total_eve_calls, 
        'total_eve_charge': total_eve_charge, 
        'total_night_minutes': total_night_minutes, 
        'total_night_calls': total_night_calls, 
        'total_night_charge': total_night_charge, 
        'total_intl_minutes': total_intl_minutes, 
        'total_intl_calls': total_intl_calls, 
        'total_intl_charge': total_intl_charge, 
        'number_customer_service_calls': number_customer_service_calls, 
        'total_nat_minutes': total_nat_minutes, 
        'total_nat_calls': total_nat_calls, 
        'total_nat_charge': total_nat_charge 
        
    }
    
    # Convert input data into a DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Make predictions
    prediction = predict_churn(input_df)
    
    # Display the prediction
    if prediction[0] == 1:
        st.error('Churn Predicted: Yes')
    else:
        st.success('Churn Predicted: No')
