import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# ── Cache everything so it loads ONCE, not on every widget change ──────────
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('regression_model.h5')

@st.cache_resource
def load_encoders():
    with open('label_encoder.pkl', 'rb') as f:
        le_gender = pickle.load(f)
        
    with open('onehot_encoder_geo.pkl', 'rb') as f:
        ohe_geo = pickle.load(f)

    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return le_gender, ohe_geo, scaler

model = load_model()
label_encoder_gender, onehot_encoder_geo, scaler = load_encoders()

# ── UI ─────────────────────────────────────────────────────────────────────
st.title('Estimated Salary Prediction of Customer')

geography        = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender           = st.selectbox('Gender', label_encoder_gender.classes_)
age              = st.slider('Age', 18, 92)
balance          = st.number_input('Balance')
credit_score     = st.number_input('Credit Score')
exited           = st.selectbox('Exited',[0,1])
tenure           = st.slider('Tenure', 0, 10)
num_of_products  = st.slider('Number of Products', 1, 4)
has_cr_card      = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# ── Prepare input ──────────────────────────────────────────────────────────
input_data = pd.DataFrame({
    'CreditScore':     [credit_score],
    'Gender':          [label_encoder_gender.transform([gender])[0]],
    'Age':             [age],
    'Tenure':          [tenure],
    'Balance':         [balance],
    'NumOfProducts':   [num_of_products],
    'HasCrCard':       [has_cr_card],
    'IsActiveMember':  [is_active_member],
    'Exited':          [exited]
})

# Fix: pass a DataFrame with the column name the encoder was fitted with
geo_df = pd.DataFrame([[geography]], columns=['Geography'])
geo_encoded = onehot_encoder_geo.transform(geo_df).toarray()
geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=onehot_encoder_geo.get_feature_names_out(['Geography'])
)

input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)


if st.button("Predict Estimated Salary"):
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)
    predicted_salary = prediction[0][0]

    st.success(f"Predicted Estimated Salary: ₹{predicted_salary:,.2f}")