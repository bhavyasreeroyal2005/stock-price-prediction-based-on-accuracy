# Stock Price Predictor Web App - LSTM AI

A powerful deep learning web application for predicting stock prices using LSTM (Long Short-Term Memory) neural networks. Supports custom forecast periods and year-wise visualization.

## Features

- 🎨 Modern, responsive UI with interactive charts
- 🧠 Deep Learning-powered price predictions using LSTM
- ⏰ Custom forecast period (1-365 days)
- 📅 Year-wise visualization for multi-year forecasts
- 📊 Interactive Chart.js visualizations
- 📥 Download predictions as CSV
- 🔍 Search any stock ticker symbol (US and Indian stocks)
- 💹 Support for Groww-accessible stocks and yfinance data
- 📈 Historical price chart with future predictions overlay

## Installation

1. **Install Python** (3.8 or higher)

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   python app.py
   ```

4. **Open your browser and visit:**
   ```
   http://localhost:5000
   ```

## How to Use

1. Enter a stock ticker symbol in the search box
2. Specify the number of days for forecast (1-365 days)
3. Click "Search" or press Enter
4. View interactive charts showing historical and predicted prices
5. If forecast spans multiple years, use year selector buttons to view year-wise predictions
6. Click "Download Predictions CSV" to save the report

## Example Tickers

### US Stocks:
- AAPL (Apple)
- MSFT (Microsoft)
- GOOGL (Google)
- TSLA (Tesla)
- AMZN (Amazon)
- NVDA (Nvidia)

### Indian Stocks (Groww):
- RELIANCE.NS (Reliance Industries)
- HDFCBANK.NS (HDFC Bank)
- TCS.NS (Tata Consultancy Services)
- INFY.NS (Infosys)
- WIPRO.NS (Wipro)
- HCLTECH.NS (HCL Technologies)

## How It Works

The app uses:
- **Deep Learning with LSTM**: Long Short-Term Memory neural networks for time series prediction
- **Multi-layered Architecture**: 3-layer LSTM with dropout regularization
- **TensorFlow/Keras**: For neural network training and prediction
- **Groww & yfinance**: Fetch historical stock data from multiple sources
- **Custom Forecast Period**: Users can specify prediction window (1-365 days)
- **Year-wise Analysis**: Automatic grouping and visualization of multi-year predictions
- **Interactive Charts**: Chart.js for beautiful, responsive visualizations

### Technical Details:
- **Sequence Length**: 60 days of historical data used for each prediction
- **Model Architecture**: 3 LSTM layers (50 units each) with dropout (0.2)
- **Training**: 20 epochs with early stopping and validation split
- **Features**: Open, High, Low, Close, Volume
- **Normalization**: MinMaxScaler for data preprocessing

## Forecasting Guidelines

- **Short-term**: 1-30 days (ideal for active trading)
- **Medium-term**: 31-90 days (good for swing trading)
- **Long-term**: 91-365 days (useful for investment planning)

When forecasting for more than a year, the app automatically creates year-wise visualization for better analysis.

## Requirements

- Python 3.8+
- TensorFlow 2.x
- Internet connection (for fetching stock data)
- 4GB+ RAM recommended for LSTM model training

## Troubleshooting

### If you encounter errors:

1. **Memory issues**: Reduce the forecast period
2. **Slow predictions**: First prediction takes longer (model training), subsequent predictions are cached
3. **No data for stock**: Ensure the ticker symbol is correct and the stock is actively traded
4. **TensorFlow errors**: Upgrade TensorFlow:
   ```bash
   pip install --upgrade tensorflow
   ```

### For Indian stocks:
- Add `.NS` suffix for NSE stocks (e.g., RELIANCE.NS)
- Add `.BO` suffix for BSE stocks (e.g., RELIANCE.BO)

## Performance Notes

- First prediction for a stock may take 20-30 seconds (model training)
- Subsequent predictions for the same stock are faster (cached model)
- Model accuracy improves with more historical data

## License

MIT License - Feel free to use and modify as needed.

## Acknowledgments

- Built with TensorFlow/Keras for deep learning
- Data provided by Yahoo Finance and Groww
- Charts powered by Chart.js
