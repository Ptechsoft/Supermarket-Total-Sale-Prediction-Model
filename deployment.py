# Loading Necessary Libraries
import streamlit as st
import pandas as pd
import joblib
import os
from PIL import Image
print('Libraries Imported Successfully')

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load the trained CatBoostmodel
path = os.path.join(script_dir, "Supermarket Total Sale_Catboost_model.pkl")
model = joblib.load(path)
print('Model Loaded Successfully')

# Load background image 
image_path = os.path.join(script_dir, "supermarket_image.jpg")
if os.path.exists(image_path):
    image = Image.open(image_path)
    print('Image Loaded Successfully')
else:
    print('No image found — skipping display')
    
    # Title & Header
st.title("Supermarket Sales Prediction Model")
st.markdown("Real-time flood risk assessment")