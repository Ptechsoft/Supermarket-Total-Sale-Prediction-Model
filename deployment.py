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
path = os.path.join(script_dir, "Supermarket_Sales_CatBoost.pkl")
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
st.title("🛒 Supermarket Sales Prediction Model")
st.markdown("Enter transaction details to predict total sales amount.")

#  User Input Form (Sidebar)

st.sidebar.header("Enter Transaction Details")

with st.sidebar.form("predict_form"):
    branch = st.selectbox("Branch", options=["Alex", "Giza", "Cairo"])
    city = st.selectbox("City", options=["Yangon", "Naypyitaw", "Mandalay"])
    customer_type = st.selectbox("Customer Type", options=["Member", "Normal"])
    gender = st.selectbox("Gender", options=["Female", "Male"])
    product_line = st.selectbox("Product Line", options=[
        "Health and beauty", "Electronic accessories", "Home and lifestyle",
        "Sports and travel", "Food and beverages", "Fashion accessories"
    ])
    payment = st.selectbox("Payment Method", options=["Ewallet", "Cash", "Credit card"])

    # Unit price & Quantity – kept for demo version
    unit_price = st.number_input("Unit Price ($)", min_value=0.0, max_value=200.0, value=50.0, step=0.1)
    quantity = st.number_input("Quantity", min_value=1, max_value=20, value=5, step=1)

    # Month, Day, Hour, Day of Week – categorical/numeric
    month = st.selectbox("Month", options=[1, 2, 3], format_func=lambda x: ["January", "February", "March"][x-1])
    day = st.slider("Day of Month", 1, 31, 15)
    hour = st.slider("Hour of Day", 0, 23, 12)
    day_of_week = st.selectbox("Day of Week", options=range(7), format_func=lambda x: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][x])

    submitted = st.form_submit_button("Predict Sales")
    
#  Prepare input data for prediction
# ────────────────────────────────────────────────────────────────
if submitted:
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

    # One-hot encode the input (must match training columns)
    input_encoded = pd.get_dummies(input_data, columns=categorical_cols, drop_first=True, dtype=int)

    # Align columns with training data (add missing columns with 0)
    missing_cols = set(X.columns) - set(input_encoded.columns)
    for col in missing_cols:
        input_encoded[col] = 0
    input_encoded = input_encoded[X.columns]  # Reorder to match training

    # Make prediction
    try:
        prediction = model.predict(input_encoded)[0]
        st.success(f"**Predicted Total Sales: ${prediction:,.2f}**")
        st.balloons()

        # Display input summary (with original categorical names)
        st.markdown("### Input Summary")
        st.dataframe(input_data.style.format({
            'Unit price': '${:.2f}',
            'Quantity': '{:.0f}'
        }))
    except Exception as e:
        st.error(f"Prediction error: {e}")

# ────────────────────────────────────────────────────────────────
#  Footer / Info
# ────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Model: CatBoost Regressor | Trained with Unit price & Quantity (demo version)")
st.caption("Note: This predicts using post-purchase data (Unit price, Quantity) – for demo only.")
 repo
Add app.py and model file
Go to https://share.streamlit.io → New app → connect repo


Key Updates in This Version

No input fields for Tax 5% or Rating — excluded completely
Categorical inputs use original string names (e.g., "Member", "Female") — converted to one-hot internally
Unit price & Quantity kept as inputs (per your preference)
Model predicts directly using the one-hot encoded input

Let me know if you get any errors running locally or deploying — I can help debug!
Also, if you want:

Add a "realistic mode" toggle (drop Unit price/Quantity)
Show predicted breakdown (e.g., tax estimate)
Add confidence interval

Just say the word! 🚀9.7sFastHow to Use / Deploy

Save as app.py in the same folder as your saved model file (Supermarket_Sales_CatBoost_Realistic_OneHot.pkl).
Install dependencies (if not already):Bashpip install streamlit pandas scikit-learn catboost joblib
Run locally:Bashstreamlit run app.py
Deploy for free (Streamlit Community Cloud):
Create GitHub