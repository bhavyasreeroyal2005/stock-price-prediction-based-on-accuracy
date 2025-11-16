# Troubleshooting Guide

## Problem: "No timezone found, symbol may be delisted" or JSON errors

This is a known issue with yfinance when accessing Yahoo Finance API. Here are solutions:

### Solution 1: Upgrade yfinance (Recommended)

```bash
pip install --upgrade yfinance
```

### Solution 2: Use Proxy/VPN

If you're in a region where Yahoo Finance is blocked, try using a VPN or proxy.

### Solution 3: Wait and Retry

The error often occurs due to rate limiting. Wait a few minutes and try again.

### Solution 4: Alternative Data Source

The app has been updated to use multiple fallback methods. The first attempt uses `Ticker.history()`, and if that fails, it falls back to `download()`.

### Solution 5: Network/Firewall Issues

Check if your firewall or network is blocking access to Yahoo Finance.

## Testing

To test if yfinance is working:

```python
import yfinance as yf

ticker = yf.Ticker("AAPL")
data = ticker.history(period="1mo")
print(data)
```

If this doesn't work, the issue is with your network or yfinance installation.

## Known Issues

- **JSON errors**: Usually network-related or rate limiting
- **Timezone errors**: Yahoo Finance API issue, try again later
- **Empty data**: Stock symbol may be invalid or delisted
