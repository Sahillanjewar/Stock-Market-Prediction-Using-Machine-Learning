import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load Data
stock_symbol = 'AAPL'  # You can change this to 'TSLA', 'GOOG', etc.
df = yf.download(stock_symbol, start='2015-01-01', end='2024-12-31')
df = df[['Close']]

# Step 2: Feature Engineering
df['Prediction'] = df['Close'].shift(-30)  # Predict 30 days into the future

X = np.array(df.drop(['Prediction'], axis=1))[:-30]
y = np.array(df['Prediction'])[:-30]

# Step 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Test & Evaluate
predictions = model.predict(X_test)
print(f"R² Score: {r2_score(y_test, predictions):.4f}")
print(f"MSE: {mean_squared_error(y_test, predictions):.4f}")

# Step 6: Predict Future
X_future = df.drop(['Prediction'], axis=1)[-30:]
future_prediction = model.predict(X_future)

# Step 7: Visualize
valid = df[-60:].copy()
valid['Predicted'] = np.nan
valid.iloc[-30:, valid.columns.get_loc('Predicted')] = future_prediction

plt.figure(figsize=(10, 5))
plt.title(f'{stock_symbol} Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.plot(df['Close'], label='Historical Price')
plt.plot(valid['Predicted'], label='Future Prediction', color='red')
plt.legend()
plt.tight_layout()
plt.show()
