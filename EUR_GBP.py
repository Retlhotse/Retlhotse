from Model import Model
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from testYfinanceScript import get_limited_yfinance_data # Import the function
import os # Import the os module to check for file existence

#implement model for EUR_GBP currency pair
class EUR_GBP(Model):
    look_back = 60

    def create_dataset(self, dataset):
        X, Y = [], []
        for i in range(len(dataset) - self.look_back - 1):
            X.append(dataset[i:(i + self.look_back), 0])
            Y.append(dataset[i + self.look_back, 0])
        return np.array(X), np.array(Y)

    def run_model(self):
        # Download EUR/GBP data using the imported function
        data = yf.download('EURGBP=X', period='5d', interval='1m')
        if data is None:
            print("Could not retrieve data. Exiting.")
            return None, None # Return None for both data and predictions

        # Prepare data for modeling
        close_prices = data['Close'].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(close_prices)

        # Model configuration
        train_size = int(len(scaled_data) * 0.8)
        train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

        # ARIMA Model (1,1,1)
        arima_model = ARIMA(train_data, order=(1, 1, 1))
        arima_result = arima_model.fit()
        arima_predictions = arima_result.forecast(steps=17)

        X_train, y_train = self.create_dataset(train_data)
        X_test, y_test = self.create_dataset(test_data)

        X_train = np.reshape(X_train, (X_train.shape[0], self.look_back, 1))
        X_test = np.reshape(X_test, (X_test.shape[0], self.look_back, 1))

        lstm_model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(self.look_back, 1)),
            LSTM(50, return_sequences=False),
            Dense(25),
            Dense(1)
        ])
        lstm_model.compile(optimizer='adam', loss='mean_squared_error')
        lstm_model.fit(X_train, y_train, batch_size=64, epochs=10)

        lstm_predictions = lstm_model.predict(X_test[-17:])

        hybrid_predictions = (arima_predictions * 0.1 + lstm_predictions.flatten() * 0.9)
        final_predictions = scaler.inverse_transform(hybrid_predictions.reshape(-1, 1))

        # Generate timestamps for predictions
        last_timestamp = data.index[-1]
        prediction_times = [last_timestamp + pd.Timedelta(minutes=i + 1) for i in range(17)]

        #return the predicted value with data and time stamps
        return pd.DataFrame(final_predictions, index=prediction_times, columns=['Close'])



#my_eur_gbp = EUR_GBP()
#print(my_eur_gbp.run_model())