# iris_web_app.py
import streamlit as st
from iris_model import train_model

# Load model and data
model, df, X = train_model()

# UI
st.title("Iris Flower Species Prediction")
st.write("Enter flower measurements:")

# Input fields
sepal_length = st.slider("Sepal Length (cm)", float(df["sepal length (cm)"].min()), float(df["sepal length (cm)"].max()))
sepal_width = st.slider("Sepal Width (cm)", float(df["sepal width (cm)"].min()), float(df["sepal width (cm)"].max()))
petal_length = st.slider("Petal Length (cm)", float(df["petal length (cm)"].min()), float(df["petal length (cm)"].max()))
petal_width = st.slider("Petal Width (cm)", float(df["petal width (cm)"].min()), float(df["petal width (cm)"].max()))

# Prediction
input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
prediction = model.predict(input_data)

# Output
st.subheader("Predicted Species")
st.success(prediction[0].capitalize())

# Optional
if st.checkbox("Show Feature Importance"):
    import pandas as pd
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    st.bar_chart(importance_df.set_index('Feature'))
