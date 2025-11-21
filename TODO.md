# TODO: Update Prediction Routes to Support All Algorithms

## Tasks
- [ ] Update /predict route to handle all four algorithms (LSTM, Linear Regression, Random Forest, Prophet)
- [ ] Update /download route to handle all four algorithms
- [ ] Test the changes to ensure all algorithms work correctly

## Details
The login form allows users to select from four algorithms, but the prediction routes only support Linear Regression and LSTM. Need to add support for Random Forest and Prophet algorithms in both /predict and /download routes.

## Completed Tasks
- [x] Update requirements.txt to include prophet library
- [x] Update forms.py to add RandomForest and Prophet to algorithm choices
- [x] Add RandomForestRegressor and Prophet imports to app.py
- [x] Update gold and silver chart colors to use Bootstrap-inspired colors
- [x] Add company logos to stock cards in index.html
- [x] Add company logos to mutual fund cards in index.html
