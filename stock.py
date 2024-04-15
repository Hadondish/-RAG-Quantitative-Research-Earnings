import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Load the data
data = pd.read_csv('stock_prices.csv') 

# Prepare the data
data['Date'] = pd.to_datetime(data['Date'])
data.sort_values('Date', inplace=True)
data['Price'] = data['Close']  # Assuming 'Close' is the closing price of the stock

# Feature engineering: use previous prices to predict the next price
data['Previous_Price'] = data['Price'].shift(1)
data.dropna(inplace=True)  # Remove any NaN values that arose from shifting

# Define features and target
X = data[['Previous_Price']]  # Features
y = data['Price']          

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')

# Plot the results
plt.figure(figsize=(10, 5))
plt.scatter(X_test, y_test, color='black', label='Actual Price')
plt.plot(X_test, predictions, color='blue', linewidth=3, label='Predicted Price')
plt.xlabel('Previous Day Price')
plt.ylabel('Next Day Price')
plt.title('Stock Price Prediction')
plt.legend()
plt.show()
