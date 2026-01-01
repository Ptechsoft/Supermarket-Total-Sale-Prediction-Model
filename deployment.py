# ========================================================
# SUPERMARKET SALES PREDICTOR – STREAMLIT DEPLOYMENT
# ========================================================
# This app loads a trained CatBoost model and predicts total sales amount
# based on user-entered transaction details.
# 
# Features used: Branch, City, Customer type, Gender, Product line,
# Payment, Unit price, Quantity, month, day, hour, day_of_week.
# 
# Excluded: Tax 5% (derived/leakage), Rating (post-purchase)
# 
# Model file expected: Supermarket_Sales_CatBoost.pkl
# Optional: supermarket_image.jpg for background
# ========================================================

import streamlit as st
import pandas as pd
import joblib
import os
from PIL import Image

print("Stage 1: Libraries imported successfully")

# ========================================================
#  Get script directory (for loading local files)
# ========================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
print("Stage 2: Script directory determined")

# ========================================================
#  Load the trained CatBoost model
# ========================================================
model_path = os.path.join(script_dir, "Supermarket_Sales_CatBoost.pkl")
try:
    model = joblib.load(model_path)
    print("Stage 3: Model loaded successfully")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    print(f"Stage 3: Model loading failed - {e}")
    st.stop()

# ========================================================
#  Load background image (optional)
# ========================================================
image_path = os.path.join(script_dir, "supermarket_image.jpg")
if os.path.exists(image_path):
    image = Image.open(image_path)
    print("Stage 4: Background image loaded successfully")
else:
    image = None
    print("Stage 4: No background image found – skipping display")

# ========================================================
#  Page configuration & title
# ========================================================
st.set_page_config(page_title="Supermarket Sales Predictor", layout="wide")
st.title("🛒 Supermarket Sales Prediction Model")
st.markdown("Enter transaction details to predict total sales amount (incl. tax).")
print("Stage 5: Streamlit page title & config set")

# ========================================================
#  Sidebar input form
# ========================================================
st.sidebar.header("Enter Transaction Details")
print("Stage 6: Sidebar header created")

with st.sidebar.form("predict_form"):
    # Categorical inputs (original names – converted to one-hot internally)
    branch = st.selectbox("Branch", options=["Alex", "Giza", "Cairo"])
    city = st.selectbox("City", options=["Yangon", "Naypyitaw", "Mandalay"])
    customer_type = st.selectbox("Customer Type", options=["Member", "Normal"])
    gender = st.selectbox("Gender", options=["Female", "Male"])
    product_line = st.selectbox("Product Line", options=[
        "Health and beauty", "Electronic accessories", "Home and lifestyle",
        "Sports and travel", "Food and beverages", "Fashion accessories"
    ])
    payment = st.selectbox("Payment Method", options=["Ewallet", "Cash", "Credit card"])

    # Unit price & Quantity – kept as requested
    unit_price = st.number_input("Unit Price ($)", min_value=0.0, max_value=200.0, value=50.0, step=0.1)
    quantity = st.number_input("Quantity", min_value=1, max_value=20, value=5, step=1)

    # Temporal inputs
    month = st.selectbox("Month", options=[1, 2, 3], format_func=lambda x: ["January", "February", "March"][x-1])
    day = st.slider("Day of Month", 1, 31, 15)
    hour = st.slider("Hour of Day", 0, 23, 12)
    day_of_week = st.selectbox("Day of Week", options=range(7), format_func=lambda x: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][x])

    submitted = st.form_submit_button("Predict Sales")

print("Stage 7: Input form rendered in sidebar")

# ========================================================
#  Process prediction when form is submitted
# ========================================================
if submitted:
    print("Stage 8: Prediction requested – building input data")
    
    # Create input DataFrame with original categorical values
    input_data = pd.DataFrame([{
        'Branch': branch,
        'City': city,
        'Customer type': customer_type,
        'Gender': gender,
        'Product line': product_line,
        'Payment': payment,
        'Unit price': unit_price,
        'Quantity': quantity,
        'month': month,
        'day': day,
        'hour': hour,
        'day_of_week': day_of_week
    }])

    print("Stage 9: Input DataFrame created")

    # One-hot encode input to match training data
    input_encoded = pd.get_dummies(input_data, columns=categorical_cols, drop_first=True, dtype=int)
    print("Stage 10: One-hot encoding applied to input")

    # Align columns with training data (fill missing with 0)
    missing_cols = set(X.columns) - set(input_encoded.columns)
    for col in missing_cols:
        input_encoded[col] = 0
    input_encoded = input_encoded[X.columns]  # Ensure same order
    print("Stage 11: Input columns aligned with training data")

    # Make prediction
    try:
        prediction = model.predict(input_encoded)[0]
        st.success(f"**Predicted Total Sales: ${prediction:,.2f}**")
        st.balloons()
        print("Stage 12: Prediction successful – result displayed")

        # Show user input summary
        st.markdown("### Input Summary")
        st.dataframe(input_data.style.format({
            'Unit price': '${:.2f}',
            'Quantity': '{:.0f}'
        }))
        print("Stage 13: Input summary displayed")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        print(f"Stage 12: Prediction failed – error: {e}")

# ========================================================
#  Footer
# ========================================================
st.markdown("---")
st.caption("Model: CatBoost Regressor | Trained with Unit price & Quantity (demo version)")
st.caption("Note: This predicts using post-purchase data (Unit price, Quantity) – for demo only.")
print("Stage 14: Footer & captions rendered – app ready")