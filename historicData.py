# For data manipulation
import numpy as np
import pandas as pd

# To fetch financial data
import yfinance as yf

# For visualisation
import matplotlib.pyplot as plt
plt.style.use('ggplot')


# Set the ticker as 'EURUSD=X'
forex_data_minute = yf.download('EURGBP=X', period='5d', interval='1m')

# Set the index to a datetime object
forex_data_minute.index = pd.to_datetime(forex_data_minute.index)

# Display the last five rows
print(forex_data_minute.tail())


# Transform index type from datetime to string
forex_data_minute['dates'] = forex_data_minute.index.strftime(
    '%Y-%m-%d %H:%M:%S')

# Plot the series
fig, ax = plt.subplots(figsize=(15, 7))
ax.plot(forex_data_minute['dates'], forex_data_minute['Close'])

# Set title and axis label
plt.title('EUR/GBP Data', fontsize=16)
plt.xlabel('Time', fontsize=15)
plt.ylabel('Price', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(['Close'], prop={'size': 15})

# Set maximum number of tick locators
ax.xaxis.set_major_locator(plt.MaxNLocator(10))
plt.xticks(rotation=45)

# Show the plot
plt.show()