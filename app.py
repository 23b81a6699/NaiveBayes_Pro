import streamlit as st
import pandas as pd
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config("Naive Bayes Classification", layout="wide")
st.title("🌸 Iris Flower Classification using Naive Bayes")

# --------------------------------------------------
# Sidebar : Model Settings
# --------------------------------------------------
st.sidebar.header("Naive Bayes Settings")
test_size = st.sidebar.slider("Test Size", 0.1, 0.4, 0.2)
random_state = st.sidebar.number_input("Random State", value=42)

# --------------------------------------------------
# Step 1 : Load Dataset
# --------------------------------------------------
st.header("Step 1 : Load Dataset")

data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target

st.success("Iris Dataset Loaded Successfully")
st.dataframe(df.head())

# --------------------------------------------------
# Step 2 : Train-Test Split
# --------------------------------------------------
st.header("Step 2 : Train-Test Split")

X = df[data.feature_names]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size,
    random_state=random_state
)

st.write("Training Shape :", X_train.shape)
st.write("Testing Shape :", X_test.shape)

# --------------------------------------------------
# Step 3 : Train Naive Bayes Model
# --------------------------------------------------
st.header("Step 3 : Train Naive Bayes Model")

model = GaussianNB()
model.fit(X_train, y_train)

st.success("Naive Bayes Model Trained Successfully")

# --------------------------------------------------
# Step 4 : Model Evaluation
# --------------------------------------------------
st.header("Step 4 : Model Evaluation")

y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
st.subheader("Accuracy Score")
st.success(f"Accuracy : {acc:.2f}")

st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
st.text(cm)

st.subheader("Classification Report")
st.text(classification_report(y_test, y_pred))

# --------------------------------------------------
# Step 5 : Predict New Input
# --------------------------------------------------
st.header("Step 5 : Predict New Flower")

col1, col2 = st.columns(2)

with col1:
    sepal_length = st.slider("Sepal Length", 4.0, 8.0, 5.8)
    sepal_width = st.slider("Sepal Width", 2.0, 4.5, 3.0)

with col2:
    petal_length = st.slider("Petal Length", 1.0, 7.0, 4.0)
    petal_width = st.slider("Petal Width", 0.1, 2.5, 1.2)

input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

if st.button("Predict"):
    prediction = model.predict(input_data)
    flower = data.target_names[prediction[0]]
    st.success(f"Predicted Flower : **{flower}**")
