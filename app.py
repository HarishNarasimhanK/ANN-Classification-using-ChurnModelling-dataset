import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle

#loading the model 
model = load_model("model.h5")


with open("label_encoder_gender.pkl","rb") as file:
    label_encoder_gender = pickle.load(file)
with open("one_hot_encoder_geo.pkl","rb") as file:
    one_hot_encoder_geo = pickle.load(file)
with open("sscaler.pkl","rb") as file:
    sscaler = pickle.load(file) 


st.title("Customer Churn Prediction")

geography = st.selectbox("Geography",one_hot_encoder_geo.categories_[0])
gender = st.selectbox("Gender",label_encoder_gender.classes_)
age = st.slider("Age", 18,92)
balance = st.number_input("Balance")
credit_score = st.number_input("Credit Score")
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider("Tenure",0,10)
num_of_products = st.slider("Number of Products",1,4)
has_cr_card = st.selectbox("Has Credit Card? ",[1,0])
if has_cr_card=="yes":
    has_cr_card = 1
else:
    has_cr_card = 0 
is_active_member =  st.selectbox("Is Active Member? ",["yes","No"])
if is_active_member=="yes":
    is_active_member = 1
else:
    is_active_member = 0

input_data = pd.DataFrame({
    "CreditScore":[credit_score],
    "Geography":[geography],
    "Gender":[label_encoder_gender.transform([gender])[0]],
    "Age":[age],
    "Tenure":[tenure],
    "Balance":[balance],
    "NumOfProducts":[num_of_products],
    "HasCrCard":[has_cr_card],
    "IsActiveMember":[is_active_member],
    "EstimatedSalary":[estimated_salary]
})

input_data = pd.DataFrame(input_data)
encoded_geo = pd.DataFrame(one_hot_encoder_geo.transform([[geography]]),columns = one_hot_encoder_geo.get_feature_names_out(["Geography"]))
input_data = pd.concat([input_data, encoded_geo],axis=1)
input_data = input_data.drop("Geography",axis=1)
input_data_scaled = sscaler.transform(input_data)
print(input_data_scaled)
prediction = model.predict(input_data_scaled)
prediction_probabilty = prediction[0][0]

if prediction_probabilty > 0.5:
    st.write("Customer is likely to churn")
else:
    st.write("Customer is NOT likely to churn")