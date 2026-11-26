# Used Car Price Predictor

A web application for predicting used car prices in the Indian market using a Multi-Layer Perceptron (MLP) neural network.

The application was developed with Python and Streamlit and allows users to enter vehicle characteristics and obtain an estimated price.

## Project Overview

This project demonstrates the application of supervised machine learning to a regression problem.

The model uses vehicle characteristics to estimate used car prices, including:

- Car age
- Kilometers driven
- Mileage
- Engine capacity
- Power
- Number of seats
- Owner type
- Brand
- Location
- Fuel type
- Transmission

## Machine Learning Model

The prediction model is based on Scikit-learn's `MLPRegressor`, with two hidden layers containing 64 and 32 neurons.

Data preprocessing is performed using a fitted Scikit-learn `ColumnTransformer`, including numerical scaling, ordinal encoding, and one-hot encoding.

### Model Performance

| Metric | Result |
|---|---:|
| MAE | 1.424 lakhs |
| RMSE | 3.467 lakhs |
| R² | 0.885 |

These results correspond to the model evaluation performed during development.

## Technologies

- Python
- Pandas
- Scikit-learn
- Joblib
- Streamlit

## Dataset

The project uses the **Used Cars Price Prediction** dataset by Avik Saliwal, available on Kaggle:

https://www.kaggle.com/datasets/avikasliwal/used-cars-price-prediction

The dataset represents the Indian used-car market.

## Installation and Execution

Clone the repository and install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

The trained model and preprocessing objects are provided as `.pkl` files in the repository.

## Important Notes

- Predictions are expressed in lakhs of Indian Rupees (INR).
- The model was trained on data representing the Indian used-car market.
- This application is an educational and portfolio project. Predictions should not be interpreted as professional vehicle valuations.
