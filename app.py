import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image

# Load the model and scaler
model_filename = 'best_random_forest_model.pkl'
scaler_filename = 'scaler.pkl'
model = joblib.load(model_filename)
scaler = joblib.load(scaler_filename)

# Load and display images
image = Image.open('BreastCancerAI.png')  # Replace with the path to your image
st.image(image, use_column_width=True)

# Set the title with a different color
st.markdown(
    "<h1 style='text-align: center; color: #ff4b4b;'>Breast Cancer Prediction</h1>",
    unsafe_allow_html=True
)

# Update the description with a different color
st.markdown(
    """
    <div style='text-align: center; color: #4b8bff;'>
    This app is designed to help doctors predict whether a breast cancer patient will live or die based on various factors.
    Enter the patient's data below, and the model will provide a prediction.
    </div>
    """,
    unsafe_allow_html=True
)

# Collect user input
def user_input_features():
    Age = st.number_input("Age", min_value=20, max_value=100, value=50)
    Gender = st.selectbox("Gender", ("Male", "Female"))
    Protein1 = st.number_input("Protein1 Level", min_value=0.0, max_value=100.0, value=50.0)
    Protein2 = st.number_input("Protein2 Level", min_value=0.0, max_value=100.0, value=50.0)
    Protein3 = st.number_input("Protein3 Level", min_value=0.0, max_value=100.0, value=50.0)
    Protein4 = st.number_input("Protein4 Level", min_value=0.0, max_value=100.0, value=50.0)
    Tumour_Stage = st.selectbox("Tumour Stage", ("I", "II", "III"))
    Histology = st.selectbox("Histology", ("Infiltrating Ductal Carcinoma", "Infiltrating Lobular Carcinoma", "Mucinous Carcinoma"))
    ER_status = st.selectbox("ER Status", ("Positive", "Negative"))
    PR_status = st.selectbox("PR Status", ("Positive", "Negative"))
    HER2_status = st.selectbox("HER2 Status", ("Positive", "Negative"))
    Surgery_type = st.selectbox("Surgery Type", ("Lumpectomy", "Simple Mastectomy", "Modified Radical Mastectomy", "Other"))
    Follow_Up_Duration = st.number_input("Follow-Up Duration (days)", min_value=0, max_value=3650, value=365)
    
    # Encode categorical variables
    gender_dict = {"Male": 0, "Female": 1}
    tumour_stage_dict = {"I": 0, "II": 1, "III": 2}
    histology_dict = {
        "Infiltrating Ductal Carcinoma": 0,
        "Infiltrating Lobular Carcinoma": 1,
        "Mucinous Carcinoma": 2
    }
    status_dict = {"Positive": 1, "Negative": 0}
    surgery_type_dict = {
        "Lumpectomy": 0,
        "Simple Mastectomy": 1,
        "Modified Radical Mastectomy": 2,
        "Other": 3
    }
    
    data = {
        "Age": Age,
        "Gender": gender_dict[Gender],
        "Protein1": Protein1,
        "Protein2": Protein2,
        "Protein3": Protein3,
        "Protein4": Protein4,
        "Tumour_Stage": tumour_stage_dict[Tumour_Stage],
        "Histology": histology_dict[Histology],
        "ER status": status_dict[ER_status],
        "PR status": status_dict[PR_status],
        "HER2 status": status_dict[HER2_status],
        "Surgery_type": surgery_type_dict[Surgery_type],
        "Follow_Up_Duration": Follow_Up_Duration
    }
    
    features = pd.DataFrame(data, index=[0])
    
    # Ensure the order of features matches the order during training
    ordered_features = [
        "Age", "Gender", "Protein1", "Protein2", "Protein3", "Protein4",
        "Tumour_Stage", "Histology", "ER status", "PR status", "HER2 status",
        "Surgery_type", "Follow_Up_Duration"
    ]
    features = features[ordered_features]
    
    return features

# Input features
input_df = user_input_features()

# Add a button to make predictions
if st.button('Predict'):
    # Apply scaling
    input_scaled = scaler.transform(input_df)

    # Make predictions
    prediction_proba = model.predict_proba(input_scaled)[:, 1][0]
    prediction = model.predict(input_scaled)[0]

    # Display the results with color
    st.subheader("Prediction")
    if prediction == 0:
        outcome = "Alive"
        color = "green"
    else:
        outcome = "Dead"
        color = "red"
    st.markdown(f"<h2 style='color: {color};'>The predicted patient status is: {outcome}</h2>", unsafe_allow_html=True)
    st.write(f"Prediction probability of death: **{prediction_proba:.2f}**")

    st.subheader("Input Features")
    st.write(input_df)