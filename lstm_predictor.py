# lstm_predictor.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
import torch
from datetime import datetime, timedelta

class LSTMPredictor:
    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler
        self.sequence_length = 60

    def predict_future(self, historical_data, days=14):
        try:
            # Prepare the most recent data
            recent_data = historical_data['close'].values[-self.sequence_length:]
            scaled_data = self.scaler.transform(recent_data.reshape(-1, 1))
            
            # Initialize prediction sequence
            pred_sequence = scaled_data.reshape(1, self.sequence_length, 1)
            predictions = []
            
            # Generate predictions for next 14 days
            for _ in range(days):
                # Get prediction
                with torch.no_grad():
                    pred = self.model(torch.FloatTensor(pred_sequence))
                    pred_value = self.scaler.inverse_transform(pred.numpy())[0][0]
                    predictions.append(pred_value)
                
                # Update sequence for next prediction
                pred_sequence = np.roll(pred_sequence, -1)
                pred_sequence[0, -1, 0] = pred.numpy()[0][0]
            
            # Create future dates
            last_date = historical_data.index[-1]
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days)
            
            return pd.DataFrame({
                'timestamp': future_dates,
                'predicted_close': predictions
            })
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return None