import streamlit as st
st.title("House prediction Price")
import numpy as np
import pickle
with open("flow.pkl",'rb') as file:
    price_model= pickle.load(file)
def house_prediction(CRIM,ZN,INDUS,CHAS,NOX,RM,AGE,DIS,RAD,TAX,PTRATIO,B,LSTAT):
    input_array= np.array([[CRIM,ZN,INDUS,CHAS,NOX,RM,AGE,DIS,RAD,TAX,PTRATIO,B,LSTAT]])
    House_model= price_model.predict(input_array)
    return House_model

# CRIM	ZN	INDUS	CHAS	NOX	RM	AGE	DIS	RAD	TAX	PTRATIO	B	LSTAT

CRIM= st.slider("CRIM",min_value=0.1,max_value=99.9)
ZN= st.slider("ZN",min_value=0.1,max_value=99.9)
INDUS= st.slider("INDUS",min_value=0.1,max_value=99.9)
CHAS= st.slider("CHAS",min_value=0.1,max_value=99.9)
NOX= st.slider("NOX",min_value=0.1,max_value=99.9)
RM= st.slider("RM",min_value=0.1,max_value=99.9)
AGE= st.slider("AGE",min_value=0.1,max_value=99.9)
DIS= st.slider("DIS",min_value=0.1,max_value=99.9)
RAD= st.slider("RAD",min_value=0.1,max_value=99.9)
TAX= st.slider("TAX",min_value=0.1,max_value=99.9)
PTRATIO= st.slider("PTRATIO",min_value=0.1,max_value=99.9)
B= st.slider("B",min_value=0.1,max_value=99.9)
LSTAT= st.slider("LSTAT",min_value=0.1,max_value=99.9)
if st.button("Predict"):
    Prediction= house_prediction(CRIM,ZN,INDUS,CHAS,NOX,RM,AGE,DIS,RAD,TAX,PTRATIO,B,LSTAT)
    st.write(f"user values are {CRIM,ZN,INDUS,CHAS,NOX,RM,AGE,DIS,RAD,TAX,PTRATIO,B,LSTAT}")
    st.write(f"\n The prediction is {Prediction[0]}")



