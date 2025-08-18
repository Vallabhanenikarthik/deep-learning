import pandas as pd
import streamlit as st

# Load data from CSV
try:
    df = pd.read_csv(r"C:\Users\Karthik\Desktop\dl\week-2\placement.csv")
    X = df[['cgpa', 'iq']].values.tolist()
    y = df['placement'].values.tolist()  # Corrected column name for the target variable
except FileNotFoundError:
    print("Error: data.csv not found. Please make sure the file is in the correct directory.")
    # Fallback to original data if CSV not found (optional)
    X = [
        [8.2, 120],
        [6.5, 100],
        [7.0, 110]
    ]
    y = [1, 0, 1]  # labels

# Parameters
learning_rate = 0.1
epochs = 5

# Initialize weights and bias
w = [0.0, 0.0]
b = 0.0

# Activation function (step)
def step(z):
    return 1 if z >= 0 else 0

# Training
for epoch in range(epochs):
    for i in range(len(X)):
        # Weighted sum
        z = w[0]*X[i][0] + w[1]*X[i][1] + b
        y_pred = step(z)

        # Error
        error = y[i] - y_pred

        # Update weights and bias
        w[0] += learning_rate * error * X[i][0]
        w[1] += learning_rate * error * X[i][1]
        b += learning_rate * error

# Prediction function
def predict(cgpa, iq):
    z = w[0]*cgpa + w[1]*iq + b
    return step(z)

# Streamlit UI
st.title("Placement Prediction using Perceptron")

# Input fields for CGPA and IQ
cgpa = st.slider("Enter CGPA", min_value=0.0, max_value=10.0, value=7.5, step=0.1)
iq = st.slider("Enter IQ", min_value=50, max_value=200, value=115, step=1)

# Predict button
if st.button("Predict Placement"):
    result = predict(cgpa, iq)
    if result == 1:
        st.success("The student is predicted to be placed.")
    else:
        st.error("The student is predicted to not be placed.")

# Show the final weights and bias
st.subheader("Model Parameters")
st.write(f"Final Weights: {w}")
st.write(f"Final Bias: {b}")