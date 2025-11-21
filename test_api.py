#!/usr/bin/env python3
"""
Test script for API endpoints
"""

import requests
import json

def test_login():
    """Test login with Linear Regression selection"""
    print("🔐 Testing login with Linear Regression...")

    login_data = {
        'username': 'testuser',
        'password': 'testpass',
        'preferred_algorithm': 'Linear Regression',
        'submit': 'Sign In'
    }

    try:
        response = requests.post('http://localhost:5000/login', data=login_data, allow_redirects=False)
        print(f"Login response status: {response.status_code}")

        if 'Set-Cookie' in response.headers:
            session_cookie = response.headers['Set-Cookie'].split(';')[0]
            print("✅ Login successful, session cookie received")
            return session_cookie
        else:
            print("❌ Login failed - no session cookie")
            return None

    except Exception as e:
        print(f"❌ Login error: {e}")
        return None

def test_prediction(session_cookie):
    """Test prediction with Linear Regression"""
    print("\n📊 Testing prediction with Linear Regression...")

    prediction_data = {
        'ticker': 'AAPL',
        'forecast_days': 30
    }

    cookies = {session_cookie.split('=')[0]: session_cookie.split('=')[1]}

    try:
        response = requests.post('http://localhost:5000/predict', json=prediction_data, cookies=cookies)
        print(f"Prediction response status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print("✅ Prediction successful")
            print(f"Current price: ${result.get('current_price', 'N/A')}")
            print(f"Algorithm: {result.get('algorithm', 'Unknown')}")
            predictions = result.get('predictions', [])
            if predictions:
                print(f"First prediction: ${predictions[0]:.2f}")
            return True
        else:
            print(f"❌ Prediction failed: {response.text}")
            return False

    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return False

def test_lstm_login():
    """Test login with LSTM selection"""
    print("\n🔐 Testing login with LSTM...")

    login_data = {
        'username': 'testuser',
        'password': 'testpass',
        'preferred_algorithm': 'LSTM',
        'submit': 'Sign In'
    }

    try:
        response = requests.post('http://localhost:5000/login', data=login_data, allow_redirects=False)
        print(f"Login response status: {response.status_code}")

        if 'Set-Cookie' in response.headers:
            session_cookie = response.headers['Set-Cookie'].split(';')[0]
            print("✅ Login successful, session cookie received")
            return session_cookie
        else:
            print("❌ Login failed - no session cookie")
            return None

    except Exception as e:
        print(f"❌ Login error: {e}")
        return None

def test_lstm_prediction(session_cookie):
    """Test prediction with LSTM"""
    print("\n📊 Testing prediction with LSTM...")

    prediction_data = {
        'ticker': 'AAPL',
        'forecast_days': 30
    }

    cookies = {session_cookie.split('=')[0]: session_cookie.split('=')[1]}

    try:
        response = requests.post('http://localhost:5000/predict', json=prediction_data, cookies=cookies)
        print(f"Prediction response status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print("✅ Prediction successful")
            print(f"Current price: ${result.get('current_price', 'N/A')}")
            print(f"Algorithm: {result.get('algorithm', 'Unknown')}")
            predictions = result.get('predictions', [])
            if predictions:
                print(f"First prediction: ${predictions[0]:.2f}")
            return True
        else:
            print(f"❌ Prediction failed: {response.text}")
            return False

    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting API tests...")
    print("=" * 50)

    # Test Linear Regression
    lr_session = test_login()
    if lr_session:
        test_prediction(lr_session)

    # Test LSTM
    lstm_session = test_lstm_login()
    if lstm_session:
        test_lstm_prediction(lstm_session)

    print("\n🎉 Testing completed!")
