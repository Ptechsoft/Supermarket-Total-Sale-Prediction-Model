import streamlit as st
import pandas as pd
import joblib
import os
from PIL import Image
import base64

print("Stage 1: Libraries imported successfully")

# Get script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
print("Stage 2: Script directory determined")

# Load model
model_path = os.path.join(script_dir, "Supermarket_Sales_CatBoost.pkl")
try:
    model = joblib.load(model_path)
    print("Stage 3: Model loaded successfully")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    print(f"Stage 3: Model loading failed - {e}")
    st.stop()

# Load training column names
columns_path = os.path.join(script_dir, "training_columns.pkl")
try:
    training_columns = joblib.load(columns_path)
    print("Stage 3b: Training columns loaded successfully")
except Exception as e:
    st.error(f"Failed to load training columns: {e}")
    print(f"Stage 3b: Columns loading failed - {e}")
    st.stop()

# Load background image
image_path = os.path.join(script_dir, "supermarket_image.jpg")
if os.path.exists(image_path):
    with open(image_path, "rb") as img_file:
        img_bytes = img_file.read()
    b64_img = base64.b64encode(img_bytes).decode()
    print("Stage 4: Background image loaded and encoded successfully")
else:
    b64_img = None
    print("Stage 4: No background image found – using default")

# Custom CSS: background + readable labels (no high contrast)
background_css = f"""
<style>
    .stApp {{
        background-image: url(data:image/jpeg;base64,{b64_img}) !important;
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    .block-container {{
        background-color: rgba(255, 255, 255, 0.85);  /* light overlay for readability */
        padding: 2.5rem;
        border-radius: 15px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.2);
        margin: 2rem auto;
        max-width: 1100px;
    }}
    h1, h2, h3 {{
        color: #1e3a8a !important;  /* deep blue */
        text-shadow: 1px 1px 4px rgba(255,255,255,0.8);
        text-align: center;
        font-weight: bold;
    }}
    label, .stSelectbox > label, .stNumberInput > label, .stSlider > label {{
        color: #1e293b !important;  /* dark slate for labels */
        font-weight: bold !important;
        font-size: 1.1rem !important;
        text-shadow: 1px 1px 2px rgba(255,255,255,0.7);
    }}
    .stForm {{
        background-color: rgba(248, 250, 252, 0.95);
        padding: 1.5rem;
        border-radius: 12px;
    }}
    .stButton > button {{
        background-color: #1e40af;
        color: white;
        font-weight: bold;
        border-radius: 8px;
    }}
    .stSuccess {{
        background-color: #dcfce7;
        color: #166534;
    }}
</style>
""" if b64_img else """
<style>
    .block-container {{ background-color: #ffffff; padding: 2.5rem; border-radius: 15px; }}
    h1, h2 {{ color: #1e3a8a; text-align: center; }}
    label {{ color: #1e293b; font-weight: bold; }}
</style>
"""

st.markdown(background_css, unsafe_allow_html=True)
print("Stage 5: Custom CSS applied – labels visible, soft overlay")

# Page title
st.title("🛒 Supermarket Sales Prediction Model")
print("Stage 6: Page title rendered")

# Main body input form
st.header("Enter Transaction Details")
print("Stage 7: Header rendered")

with st.form("predict_form"):
    col1, col2 = st.columns(2)

    with col1:
        branch = st.selectbox("Branch", options=["Alex", "Giza", "Cairo"])
        city = st.selectbox("City", options=["Yangon", "Naypyitaw", "Mandalay"])
        customer_type = st.selectbox("Customer Type", options=["Member", "Normal"])
        gender = st.selectbox("Gender", options=["Female", "Male"])

    with col2:
        product_line = st.selectbox("Product Line", options=[
            "Health and beauty", "Electronic accessories", "Home and lifestyle",
            "Sports and travel", "Food and beverages", "Fashion accessories"
        ])
        payment = st.selectbox("Payment Method", options=["Ewallet", "Cash", "Credit card"])

        unit_price = st.number_input("Unit Price ($)", min_value=0.0, max_value=200.0, value=50.0, step=0.1)
        quantity = st.number_input("Quantity", min_value=1, max_value=20, value=5, step=1)

    col3, col4 = st.columns(2)

    with col3:
        month = st.selectbox("Month", options=[1, 2, 3], format_func=lambda x: ["January", "February", "March"][x-1])
        day = st.slider("Day of Month", 1, 31, 15)

    with col4:
        hour = st.slider("Hour of Day", 0, 23, 12)
        day_of_week = st.selectbox("Day of Week", options=range(7), format_func=lambda x: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][x])

    submitted = st.form_submit_button("Predict Sales")

print("Stage 8: Main input form rendered")

if submitted:
    print("Stage 9: Prediction button clicked – preparing input")

    # Define categorical columns
    categorical_cols = ['Branch', 'City', 'Customer type', 'Gender', 'Product line', 'Payment']
    print("Stage 10: Categorical columns defined")

    # Create input DataFrame
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
    print("Stage 11: Input DataFrame created")

    # One-hot encode
    input_encoded = pd.get_dummies(input_data, columns=categorical_cols, drop_first=True, dtype=int)
    print("Stage 12: One-hot encoding applied")

    # Align columns with training data
    missing_cols = set(training_columns) - set(input_encoded.columns)
    for col in missing_cols:
        input_encoded[col] = 0
    input_encoded = input_encoded[training_columns]
    print("Stage 13: Input aligned with training columns")

    # Predict
    try:
        prediction = model.predict(input_encoded)[0]
        st.success(f"**Predicted Total Sales: ${prediction:,.2f}**")
        st.balloons()
        print("Stage 14: Prediction successful")

        st.subheader("Input Summary")
        st.dataframe(input_data.style.format({
            'Unit price': '${:.2f}',
            'Quantity': '{:.0f}'
        }))
        print("Stage 15: Input summary displayed")
    except Exception as e:
        st.error(f"Prediction error: {e}")
        print(f"Stage 14: Prediction failed - {e}")

# Footer
st.markdown("---")
st.caption("Model: CatBoost Regressor | Trained with Unit price & Quantity")
st.caption("Note: This predicts using post-purchase data")
print("Stage 16: Footer rendered – app complete")