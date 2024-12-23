# Stock-Prediction
This project focuses on building a machine learning-based model to predict future stock prices using historical data and technical indicators. The repository includes scripts for data preprocessing, feature engineering, model training, and evaluation.

%%writefile app.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import streamlit as st

# Function to load and preprocess the stock data
def load_data(ticker):
    data = yf.download(ticker, start='2015-01-01', end='2024-01-01')
    return data[['Close']]

# Function to create the LSTM model
def create_model(X_train):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Function to prepare the data for the LSTM model
def prepare_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    training_data_len = int(np.ceil(len(scaled_data) * 0.8))

    train_data = scaled_data[0:training_data_len, :]
    test_data = scaled_data[training_data_len - 60:, :]

    X_train, y_train = [], []
    for i in range(60, len(train_data)):
        X_train.append(train_data[i - 60:i, 0])
        y_train.append(train_data[i, 0])

    X_test, y_test = [], []
    for i in range(60, len(test_data)):
        X_test.append(test_data[i - 60:i, 0])
        y_test.append(test_data[i, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_test, y_test = np.array(X_test), np.array(y_test)

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    return X_train, y_train, X_test, y_test, scaler

# Function to predict stock prices
def predict_prices(ticker):
    data = load_data(ticker)
    X_train, y_train, X_test, y_test, scaler = prepare_data(data)

    model = create_model(X_train)
    model.fit(X_train, y_train, batch_size=64, epochs=10)

    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)

    return data, predictions

# Streamlit App Interface
st.title("Stock Price Prediction App")
ticker = st.text_input("Enter stock ticker (e.g., AAPL, GOOGL):", value="AAPL")

if st.button("Predict"):
    data, predictions = predict_prices(ticker)
    st.subheader(f"Stock Price Prediction for {ticker}")

    # Plot the results
    fig, ax = plt.subplots()
    ax.plot(data.index, data['Close'], label='Actual Prices')
    ax.plot(data.index[-len(predictions):], predictions, label='Predicted Prices', linestyle='--')
    plt.xlabel("Date")
    plt.ylabel("Close Price USD ($)")
    plt.title(f"{ticker} Price Prediction")
    plt.legend()
    st.pyplot(fig)
