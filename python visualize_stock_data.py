from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pymongo import MongoClient
import pandas as pd
from datetime import datetime, timedelta
import uvicorn
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import joblib

# Get the current directory
BASE_DIR = r"C:\Users\ousse\Desktop\data dec"

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# MongoDB connection (same as in main2.py)
client = MongoClient('localhost', 27017)
db = client['stock_data']
collection = db['stock_prices']

# Load LSTM model and scaler
try:
    model = load_model(os.path.join(BASE_DIR, 'stock-price-prediction-lstm.h5'))
    scaler = joblib.load(os.path.join(BASE_DIR, 'scaler.pkl'))
    print("LSTM model and scaler loaded successfully")
except Exception as e:
    print(f"Error loading LSTM model: {e}")
    model = None
    scaler = None

def prepare_data_for_prediction(data, sequence_length=60):
    try:
        # Prepare the data similar to your Kaggle notebook
        scaled_data = scaler.transform(data[['close']].values)
        X = []
        for i in range(len(scaled_data) - sequence_length):
            X.append(scaled_data[i:(i + sequence_length)])
        return np.array(X)
    except Exception as e:
        print(f"Error preparing data: {e}")
        return None

def create_candlestick_chart():
    try:
        # Get the last 30 days of data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        # Query MongoDB (using same collection as main2.py)
        cursor = collection.find({
            "timestamp": {
                "$gte": start_date.strftime("%Y-%m-%d"),
                "$lte": end_date.strftime("%Y-%m-%d")
            }
        }).sort("timestamp", 1)
        
        data = list(cursor)
        
        if not data:
            return "<p>No data available</p>"
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Get LSTM predictions if model is loaded
        predictions = None
        if model is not None and scaler is not None:
            X = prepare_data_for_prediction(df)
            if X is not None:
                predictions = model.predict(X)
                predictions = scaler.inverse_transform(predictions)
        
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add candlestick
        fig.add_trace(
            go.Candlestick(
                x=df['timestamp'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name="AAPL Stock"
            )
        )
        
        if predictions is not None:
            # Add predictions
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'][60:],
                    y=predictions.flatten(),
                    name="LSTM Predictions",
                    line=dict(color='yellow', width=2)
                )
            )
        
        fig.update_layout(
            title='AAPL Stock Price with LSTM Predictions',
            yaxis_title='Stock Price (USD)',
            xaxis_title='Date',
            template='plotly_dark',
            height=600
        )
        
        return fig.to_html(full_html=False)
    except Exception as e:
        print(f"Error in create_candlestick_chart: {e}")
        return f"<p>Error creating chart: {str(e)}</p>"

def create_volume_chart():
    try:
        # Get the last 30 days of data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        # Query MongoDB (using same collection as main2.py)
        cursor = collection.find({
            "timestamp": {
                "$gte": start_date.strftime("%Y-%m-%d"),
                "$lte": end_date.strftime("%Y-%m-%d")
            }
        }).sort("timestamp", 1)
        
        data = list(cursor)
        
        if not data:
            return "<p>No volume data available</p>"
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Create volume chart
        fig = go.Figure(data=[
            go.Bar(
                x=df['timestamp'],
                y=df['volume'],
                name="Volume",
                marker_color='rgba(0, 150, 255, 0.6)'
            )
        ])
        
        fig.update_layout(
            title='Trading Volume',
            yaxis_title='Volume',
            xaxis_title='Date',
            template='plotly_dark',
            height=400,
            showlegend=True
        )
        
        return fig.to_html(full_html=False)
    except Exception as e:
        print(f"Error in create_volume_chart: {e}")
        return f"<p>Error creating volume chart: {str(e)}</p>"

def get_latest_stats():
    try:
        latest = collection.find_one({}, sort=[("timestamp", -1)])
        if not latest:
            return None
        
        # Get previous record for price change calculation
        previous = collection.find_one(
            {"timestamp": {"$lt": latest["timestamp"]}},
            sort=[("timestamp", -1)]
        )
        
        daily_change = 0
        if previous:
            daily_change = ((latest["close"] - previous["close"]) / previous["close"]) * 100
        
        # Get next day prediction if model is available
        next_day_prediction = None
        if model is not None and scaler is not None:
            try:
                recent_data = pd.DataFrame(
                    list(collection.find().sort("timestamp", -1).limit(61))
                ).sort_values('timestamp')
                X = prepare_data_for_prediction(recent_data)
                if X is not None and len(X) > 0:
                    pred = model.predict(X[-1:])
                    next_day_prediction = scaler.inverse_transform(pred)[0][0]
            except Exception as e:
                print(f"Error making prediction: {e}")
        
        return {
            "current_price": latest["close"],
            "daily_change": daily_change,
            "daily_volume": latest["volume"],
            "daily_high": latest["high"],
            "daily_low": latest["low"],
            "last_updated": latest["timestamp"],
            "next_day_prediction": next_day_prediction
        }
    except Exception as e:
        print(f"Error in get_latest_stats: {e}")
        return None

@app.get("/")
async def home(request: Request):
    try:
        candlestick_chart = create_candlestick_chart()
        volume_chart = create_volume_chart()
        stats = get_latest_stats()
        
        if not stats:
            stats = {
                "current_price": 0,
                "daily_change": 0,
                "daily_volume": 0,
                "daily_high": 0,
                "daily_low": 0,
                "last_updated": "Error loading data",
                "next_day_prediction": None
            }
        
        return templates.TemplateResponse(
            "index.html", 
            {
                "request": request,
                "candlestick_chart": candlestick_chart,
                "volume_chart": volume_chart,
                "stats": stats
            }
        )
    except Exception as e:
        print(f"Error in home route: {e}")
        return templates.TemplateResponse(
            "index.html", 
            {
                "request": request,
                "error": str(e)
            }
        )

if __name__ == "__main__":
    print("Starting Stock Visualization Dashboard...")
    print(f"Base Directory: {BASE_DIR}")
    uvicorn.run(app, host="127.0.0.1", port=8000)

def save_model_files(model, scaler):
    # Save as H5 (requires h5py)
    import h5py
    with h5py.File('stock_model.h5', 'w') as f:
        for name, param in model.state_dict().items():
            f.create_dataset(name, data=param.cpu().numpy())
    print("Model saved as stock_model.h5")
    
    # Save scaler
    joblib.dump(scaler, 'scaler.pkl')
    print("Scaler saved as scaler.pkl")