import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler


# App 1: Stock Market Trend Prediction Using Sentiment Analysis

st.title("Stock Market Trend Prediction Using Sentiment Analysis")

ticker = st.sidebar.text_input('Ticker')
start_date = st.sidebar.date_input('Start Date')
end_date = st.sidebar.date_input('End Date')


# Check if the ticker and start date are provided
if ticker and start_date:
    # Fetch historical stock data
    data = yf.download(ticker, start=start_date, end=end_date)

    
    
        # Flatten the MultiIndex columns
    data.columns = ['_'.join(col).strip() for col in data.columns.values]  # e.g., Close_AMZN
        
        
        # Reshape to long format for Plotly Express
    long_data = data.reset_index().melt(id_vars='Date', var_name='Ticker', value_name='Close')


        # Extract ticker symbols (e.g., from 'Close_AMZN' to 'AMZN')
    long_data['Ticker'] = long_data['Ticker'].str.replace('Close_', '', regex=False)

        # Plot all tickers
    fig = px.line(long_data, x='Date', y='Close', color='Ticker', title='Stock Prices')
    st.plotly_chart(fig)
         

# App 2: Stock Trend Prediction

st.title("Stock Trend Prediction")

start = "2011-02-01"
end = "2019-12-31"
user_input = st.text_input("Enter Stock Ticker", "AAPL")
df = yf.download(user_input, start=start, end=end)
df = df.reset_index()
df = df.dropna()

# Describing data
st.subheader('Data from 2011-2019')
st.write("Description")
st.write(df.describe())

# Visualization
st.subheader("Closing Price vs Time Chart")
fig = plt.figure(figsize=(12, 6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader("Closing Price vs Time Chart with 100MA")
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader("Closing Price vs Time Chart with 100MA and 200MA")
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100, "b")
plt.plot(ma200, "g")
plt.plot(df.Close)
st.pyplot(fig)


data_training = pd.DataFrame(df["Close"][0:int(len(df) * 0.70)])
data_testing = pd.DataFrame(df["Close"][int(len(df) * 0.70):int(len(df))])




    
    
    
    
   # train_sentiment_model.py

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Dummy stock sentiment data
texts = [
    "Stock prices are going up",       # positive
    "Company profits are excellent",   # positive
    "Market crash is imminent",        # negative
    "Losses exceed expectations",      # negative
]
labels = [1, 1, 0, 0]

# Tokenize text
tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded = pad_sequences(sequences, maxlen=10)

# Build a simple model
model = Sequential([
    Embedding(input_dim=1000, output_dim=16, input_length=10),
    LSTM(8),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train
model.fit(padded, np.array(labels), epochs=10, verbose=1)

# Save the model
model.save("stock_sentiment_model.h5")


if len(data_testing) > 0:
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_training_array = scaler.fit_transform(data_training)
    
    # Feeding model with past 100 days of data
    # Testing part
    past_100_days = data_training.tail(100)
    import pandas as pd

    final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
    input_data = scaler.transform(final_df)
    x_test = []
    y_test = []
    for i in range(100, input_data.shape[0]):
        x_test.append(input_data[i - 100:i])
        y_test.append(input_data[i, 0])

    x_test, y_test = np.array(x_test), np.array(y_test)
    
    
    y_pred = model.predict(x_test)

    scale_factor = 1 / 0.13513514
    y_pred = y_pred * scale_factor
    y_test = y_test * scale_factor

    # Final graph
    st.subheader("Prediction vs Original")
    fig2 = plt.figure(figsize=(12, 6))
    plt.plot(y_pred, 'r', label='Predicted Price')
    plt.plot(y_test, 'b', label='Original Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(fig2)
else:
    st.write("Insufficient data for testing.")

