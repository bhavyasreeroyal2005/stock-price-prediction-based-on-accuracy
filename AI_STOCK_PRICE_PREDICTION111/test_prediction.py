import sys
import os
sys.path.append('AIstockpriceprediction')

from AIstockpriceprediction.app import predict_stock_price_lstm

# Test prediction for an Indian stock
ticker = "RELIANCE.NS"  # Reliance Industries on NSE
forecast_days = 30

print(f"Testing prediction for {ticker} with {forecast_days} days forecast...")

result, error = predict_stock_price_lstm(ticker, forecast_days)

if error:
    print(f"Error: {error}")
else:
    print("Prediction successful!")
    print(f"Ticker: {result['ticker']}")
    print(f"Asset Type: {result['asset_type']}")
    print(f"Current Price: ₹{result['current_price']}")
    print(f"Accuracy: {result['accuracy']}%")
    print(f"RMSE: {result['rmse']}")
    print(f"Forecast Days: {result['forecast_days']}")
    print("\nPredicted Prices (first 10 days):")
    print(result['predictions'].head(10).to_string(index=False))
    print("\nHistorical Data (last 5 days):")
    print(result['historical'].tail(5)[['Open', 'High', 'Low', 'Close', 'Volume']].to_string())
