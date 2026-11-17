import streamlit as st
import joblib
from pathlib import Path

st.set_page_config(page_title="Phone Price Predictor")

@st.cache_resource
def load_model():
    file = Path(__file__).with_name("phone_price_model.joblib")
    return joblib.load(file)

artifact = load_model()
model = artifact["model"]

st.title("Phone Price Predictor")
st.write("Predict phone price based on RAM.")

ram = st.slider("RAM (GB)", 4, 12, 8)

if st.button("Predict Price"):
    price = float(model.predict([[ram]])[0])
    st.success(f"Estimated Phone Price: {price:.1f} JD")
    st.caption("Demo estimate; not an actual market price.")
