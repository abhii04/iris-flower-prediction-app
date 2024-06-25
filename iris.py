import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import datasets
import numpy as np

# Function to load the Iris dataset and split into train-test
def load_and_split_data():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    class_names = iris.target_names
    return X_train, X_test, y_train, y_test, class_names

# Function to train the Logistic Regression model
def train_model(X_train, y_train):
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    return model

# Function to make predictions
def predict(model, input_data):
    prediction = model.predict(input_data)
    return prediction

# Load and split the Iris dataset
X_train, X_test, y_train, y_test, class_names = load_and_split_data()

# Train the Logistic Regression model on the training data
model = train_model(X_train, y_train)

# Adding an image to the app
st.image('https://miro.medium.com/v2/resize:fit:720/1*YYiQed4kj_EZ2qfg_imDWA.png', caption='Iris Flowers', use_column_width=True)

# Streamlit app interface
st.title('Iris Flower Prediction')
st.write('Enter values for sepal length, sepal width, petal length, and petal width.')

# Sliders for user input
sepal_length = st.slider('Sepal Length', float(X_train[:, 0].min()), float(X_train[:, 0].max()), float(X_train[:, 0].mean()))
sepal_width = st.slider('Sepal Width', float(X_train[:, 1].min()), float(X_train[:, 1].max()), float(X_train[:, 1].mean()))
petal_length = st.slider('Petal Length', float(X_train[:, 2].min()), float(X_train[:, 2].max()), float(X_train[:, 2].mean()))
petal_width = st.slider('Petal Width', float(X_train[:, 3].min()), float(X_train[:, 3].max()), float(X_train[:, 3].mean()))

# User input data for prediction
input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

# Predicting the class with the input data
prediction = predict(model, input_data)

# Displaying the predicted class
st.write(f"Predicted class: {class_names[prediction[0]]}")
