
import streamlit as st
import joblib
import pandas as pd

# Load trained model and preprocessing pipeline
preprocessador = joblib.load("preprocessador.pkl")
mlp_reg = joblib.load("mlp_regressor.pkl")

# Extract categorical options from the fitted OneHotEncoder
categorias = preprocessador.named_transformers_["cat"].categories_

brands = sorted(categorias[0].tolist())
locations = sorted(categorias[1].tolist())
fuel_types = sorted(categorias[2].tolist())
transmissions = sorted(categorias[3].tolist())

st.set_page_config(
    page_title="Used Car Price Predictor",
    page_icon="🚗",
    layout="centered"
)

st.title("🚗 Used Car Price Predictor")

st.write(
    "Estimate the price of used cars in the Indian market "
    "using a Multi-Layer Perceptron neural network."
)

st.divider()

st.header("Enter Car Information")

with st.form("car_information"):

    st.subheader("Vehicle characteristics")

    # Numerical variables
    age = st.number_input("Car age (years)", min_value=0, max_value=50, value=5)

    kilometers = st.number_input(
        "Kilometers driven",
        min_value=0,
        value=30000,
        step=1000
    )

    mileage = st.number_input(
        "Mileage (km/l)",
        min_value=0.0,
        value=18.0
    )

    engine = st.number_input(
        "Engine capacity (CC)",
        min_value=0,
        value=1500
    )

    power = st.number_input(
        "Power (bhp)",
        min_value=0.0,
        value=100.0
    )

    seats = st.number_input(
        "Number of seats",
        min_value=1,
        max_value=20,
        value=5
    )

    st.subheader("Vehicle categories")

    # Categorical variables
    owner_type = st.selectbox(
        "Owner type",
        ["First", "Second", "Third", "Fourth & Above"]
    )

    brand = st.selectbox("Brand", brands)

    location = st.selectbox("Location", locations)

    fuel_type = st.selectbox("Fuel type", fuel_types)

    transmission = st.selectbox("Transmission", transmissions)

    # Submit button
    submitted = st.form_submit_button("Predict Price")

if submitted:

    # Create a DataFrame with the user's input
    entrada = pd.DataFrame([{
        "Age": age,
        "Kilometers_Driven": kilometers,
        "Mileage": mileage,
        "Engine": engine,
        "Power": power,
        "Seats": seats,
        "Owner_Type": owner_type,
        "Brand": brand,
        "Location": location,
        "Fuel_Type": fuel_type,
        "Transmission": transmission
    }])

    # Preprocess the input
    entrada_p = preprocessador.transform(entrada)

    # Predict the car price
    previsao = mlp_reg.predict(entrada_p)[0]

    st.write("Vehicle information:")
    st.dataframe(entrada)
    st.success(f"Estimated price: ₹ {previsao:.2f} lakhs")
