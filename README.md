# ANN-Estimated-Salary-Prediction

## Overview

This project predicts the **Estimated Salary** of a bank customer using an **Artificial Neural Network (ANN)** built with TensorFlow and Keras. The application provides an interactive web interface developed using Streamlit, allowing users to enter customer details and receive an estimated salary prediction in real time.

---

## Live Demo

**Deployed Application:**https://ann-estimated-salary-prediction-y6l9whbruyxf4vzargskf6.streamlit.app

---

## Features

* Predicts customer estimated salary using an ANN regression model.
* Interactive Streamlit web application.
* Uses trained preprocessing pipelines (Label Encoder, One-Hot Encoder, and Standard Scaler).
* Clean and responsive user interface.
* Fast inference with TensorFlow and Keras.

---

## Technologies Used

* Python
* TensorFlow & Keras
* Streamlit
* Scikit-learn
* Pandas
* NumPy
* Pickle

---

---

## Dataset

The project uses the **Churn Modelling Dataset** containing customer information such as:

* Credit Score
* Geography
* Gender
* Age
* Tenure
* Balance
* Number of Products
* Has Credit Card
* Is Active Member
* Exited
* Estimated Salary (Target Variable)

---

## Model Architecture

The ANN model consists of:

* Input Layer
* Hidden Layer (64 neurons, ReLU)
* Hidden Layer (32 neurons, ReLU)
* Output Layer (1 neuron)

### Loss Function

* Mean Absolute Error (MAE)

### Optimizer

* Adam

### Evaluation Metrics

* Mean Absolute Error (MAE)

---

## Input Features

The application accepts the following customer details:

* Credit Score
* Geography
* Gender
* Age
* Tenure
* Balance
* Number of Products
* Has Credit Card
* Is Active Member
* Exited

---

## Sample Prediction

After entering the customer details, click the **Predict Estimated Salary** button to obtain the predicted estimated salary.

---

