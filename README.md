![supermarket_image](https://github.com/user-attachments/assets/f85f32ce-a084-4392-bf56-17f1f53ae96a)

 # Supermarket-Total-Sale-Prediction-Model

### PROBLEM STATEMENT: 
Develop a machine learning model to accurately predict the total sales amount per transaction for a supermarket chain using historical transaction data.
This predictive capability will support optimized inventory management, improved staffing, targeted promotions, and data-driven business strategies by revealing patterns in customer behavior, branch performance, product lines, and temporal trends.

**Business value**:
- better inventory planning
- improved staffing decisions
- targeted promotions
- data-driven understanding of customer behavior, branch performance, product categories and temporal patterns

**Important constraint for realistic forecasting**:
The model should ideally predict using features available **before** the customer finishes shopping. 
(We will first show the trivial high-accuracy version, then explain why it is not realistic, and finally build the corrected version).

### SOLUTION
To address the supermarket’s challenge of forecasting total sales per transaction, this project develops a high-accuracy machine learning model that predicts sales amount using key customer, product, and temporal features, while keeping the direct drivers Unit price and Quantity for demonstration purposes.
Three regression algorithms were built and compared:

- Linear Regression
- Random Forest Regressor
- CatBoost Regressor

The model was trained and evaluated on a clean dataset of 1,000 transaction records.
Model performance was measured using

`Mean Absolute Error (MAE)`
`Root Mean Squared Error (RMSE)` and
`R² Score (coefficient of determination)`.

Final Result: `The CatBoost Regressor` delivered the best results, achieving an **R² ≈ 0.9989** and an MAE ≈ **$6.41**, meaning the predicted sales amount is, on average, only **±$6.41** away from the actual value.
This level of accuracy makes CatBoost highly reliable for post-transaction sales analysis, pricing strategy validation, and understanding key drivers (especially Unit price and Quantity) in the supermarket environment.
**Note**: This is a demonstration version — the model uses features only known after the transaction (Unit price, Quantity) and should not be used for true pre-purchase forecasting. For real-world forecasting, these features must be excluded.

### Use the link below to make your prediction
https://supermarket-total-sale-prediction-model-main.streamlit.app/