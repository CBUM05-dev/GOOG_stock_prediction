# GOOG_stock_prediction

# 📈 Stock Price Prediction Using LSTM

This project demonstrates how to predict future stock prices using a deep learning model built with LSTM (Long Short-Term Memory) layers. The model is trained on historical data from Google (GOOG) stock over the past 10 years, fetched using the `yfinance` API.

## 🧠 Project Overview

Stock price prediction is a challenging task due to the volatile and non-linear nature of financial markets. This notebook leverages a multi-layered LSTM neural network to learn temporal patterns in stock price movements and forecast future values.

## 🔧 Technologies Used

- **Python**
- **NumPy** for numerical operations
- **Pandas** for data manipulation
- **Matplotlib** for visualization
- **yFinance** for fetching historical stock data
- **TensorFlow / Keras** for building and training the LSTM model

## 📊 Dataset

- **Source**: Yahoo Finance via `yfinance`
- **Stock**: Google (GOOG)
- **Date Range**: January 1, 2014 – December 30, 2024
- **Features Used**: Primarily closing prices

```python
import yfinance as yf

start = "2014-01-01"
end = "2024-12-30"
stock = "GOOG"

data = yf.download(stock, start, end)
```

## 🧬 Model Architecture
The model is a deep LSTM network with multiple layers and dropout regularization to prevent overfitting.

```python
model = Sequential()

# Input shape: (100, 1)
model.add(LSTM(units=50, activation="relu", return_sequences=True, input_shape=(x.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units=60, activation="relu", return_sequences=True))
model.add(Dropout(0.3))

model.add(LSTM(units=80, activation="relu", return_sequences=True))
model.add(Dropout(0.4))

model.add(LSTM(units=120, activation="relu"))
model.add(Dropout(0.5))

model.add(Dense(units=1))
```

## 📈 Training Details
Input Shape: 2D array of shape (train_data_length, 100)

Loss Function: Mean Squared Error (MSE)

Optimizer: Adam

Epochs: Customizable based on performance

## 📉 Output
The model outputs a single predicted value for the next time step in the stock price series. Visualization is done using matplotlib to compare actual vs predicted prices.

## 🚀 How to Run
Clone the repository

## Install dependencies:

bash
pip install numpy pandas matplotlib yfinance tensorflow
Run the notebook in Jupyter or your preferred environment

## 📌 Notes
This model is for educational purposes and does not constitute financial advice.

Performance may vary depending on hyperparameters and data preprocessing.
