from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pymongo import MongoClient
from datetime import datetime, timedelta
import os
import tensorflow as tf
import joblib
import numpy as np
import pandas as pd
import pytz
from alerts import AlertSystem
import traceback
import logging
from pydantic import BaseModel

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Get the current directory
BASE_DIR = r"C:\Users\ousse\Desktop\data dec"

# Create FastAPI app
app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# MongoDB connection
client = MongoClient('mongodb://localhost:27017/')
db = client['stock_data']
collection = db['stock_prices']

# Load model and scaler
try:
    model = tf.keras.models.load_model(os.path.join(BASE_DIR, 'model.keras'))
    scaler = joblib.load(os.path.join(BASE_DIR, 'scaler.pkl'))
    print("Model and scaler loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    scaler = None

# Add at the top with other initializations
alert_system = AlertSystem()
previous_price = None

def calculate_technical_indicators(data):
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    df.index = pd.to_datetime(df.index)
    
    # Handle zero values
    for col in ['close', 'high', 'low', 'open', 'volume']:
        df[col] = df[col].replace(0, np.nan)
        df[col] = df[col].ffill()
    
    # Calculate percentage changes
    price_columns = ['close', 'high', 'low', 'open']
    for col in price_columns:
        df[f'{col}_pct'] = df[col].pct_change()
        df[f'{col}_pct'] = df[f'{col}_pct'].replace([np.inf, -np.inf], np.nan)
        df[f'{col}_pct'] = df[f'{col}_pct'].ffill()
    
    # Log transform volume
    df['volume_log'] = np.log1p(df['volume'])
    
    # Moving averages
    df['SMA_5'] = df['close'].rolling(window=5, min_periods=1).mean()
    df['SMA_20'] = df['close'].rolling(window=20, min_periods=1).mean()
    df['EMA_5'] = df['close'].ewm(span=5, adjust=False, min_periods=1).mean()
    df['EMA_20'] = df['close'].ewm(span=20, adjust=False, min_periods=1).mean()
    
    # MACD
    exp1 = df['close'].ewm(span=12, adjust=False, min_periods=1).mean()
    exp2 = df['close'].ewm(span=26, adjust=False, min_periods=1).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False, min_periods=1).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['BB_middle'] = df['close'].rolling(window=20, min_periods=1).mean()
    bb_std = df['close'].rolling(window=20, min_periods=1).std()
    df['BB_upper'] = df['BB_middle'] + 2 * bb_std
    df['BB_lower'] = df['BB_middle'] - 2 * bb_std
    
    # Volume indicators
    df['Volume_MA'] = df['volume'].rolling(window=20, min_periods=1).mean()
    df['Volume_std'] = df['volume'].rolling(window=20, min_periods=1).std()
    
    # Handle any remaining NaN or infinite values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.ffill()
    df = df.bfill()
    
    return df

def prepare_prediction_data(df):
    # Select the same features used in training
    features = [
        'close_pct', 'volume_log', 'high_pct', 'low_pct', 'open_pct',
        'SMA_5', 'SMA_20', 'EMA_5', 'EMA_20', 
        'MACD', 'Signal_Line', 'RSI',
        'BB_middle', 'BB_upper', 'BB_lower',
        'Volume_MA', 'Volume_std'
    ]
    
    return df[features]

@app.get("/api/stock-data")
async def get_stock_data():
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)
        
        data = list(collection.find({
            "timestamp": {
                "$gte": start_date.strftime("%Y-%m-%d"),
                "$lte": end_date.strftime("%Y-%m-%d")
            }
        }).sort("timestamp", 1))
        
        if not data:
            return {"error": "No data available"}
        
        # Process data
        df = calculate_technical_indicators(data)
        feature_data = prepare_prediction_data(df)
        
        # Scale features
        scaled_data = scaler.transform(feature_data)
        
        timestamps = df.index.strftime("%Y-%m-%d %H:%M:%S").tolist()[-30:]
        prices = df['close'].values[-30:]
        volumes = df['volume'].values[-30:]
        
        # Get current price for predictions
        current_price = float(df['close'].iloc[-1])
        
        # Make predictions if model is available
        predictions = []
        prediction_dates = []
        
        if model is not None and scaler is not None and len(feature_data) >= 60:
            try:
                sequence_length = 60
                recent_data = scaled_data[-sequence_length:]
                X = recent_data.reshape(1, sequence_length, len(feature_data.columns))
                
                # Predict next 24 hours with dynamic features
                last_sequence = X.copy()
                last_price = current_price
                last_date = pd.to_datetime(timestamps[-1])
                
                for i in range(24):
                    # Get prediction (percentage change)
                    pred = model.predict(last_sequence, verbose=0)
                    predicted_change = float(pred[0, 0])
                    
                    # Add some randomness to make predictions more realistic
                    volatility = 0.001  # 0.1% volatility
                    noise = np.random.normal(0, volatility)
                    predicted_change = predicted_change + noise
                    
                    # Convert to actual price
                    pred_price = last_price * (1 + predicted_change)
                    predictions.append(pred_price)
                    
                    # Calculate next date
                    next_date = last_date + timedelta(hours=i+1)
                    prediction_dates.append(next_date.strftime("%Y-%m-%d %H:%M:%S"))
                    
                    # Update sequence with new features
                    new_row = np.zeros(len(feature_data.columns))
                    
                    # Calculate percentage change for the new prediction
                    price_change = (pred_price - last_price) / last_price
                    new_row[0] = price_change  # close_pct
                    
                    # Add some market dynamics
                    if price_change > 0:
                        # Upward trend might lead to higher volume
                        volume_change = np.random.uniform(0, 0.1)
                    else:
                        # Downward trend might lead to lower volume
                        volume_change = np.random.uniform(-0.1, 0)
                    
                    new_row[1] = volume_change  # volume_log
                    
                    # Update technical indicators in new_row
                    # This is a simplified version - you might want to add more sophisticated updates
                    new_row[4:] = last_sequence[0, -1, 4:]  # Keep other indicators
                    
                    # Update the sequence
                    last_sequence = np.roll(last_sequence, -1, axis=1)
                    last_sequence[0, -1] = new_row
                    
                    # Update last price for next iteration
                    last_price = pred_price
                
                print(f"Current price: ${current_price:.2f}")
                print("Predictions:", [f"${p:.2f}" for p in predictions[:5]], "...")
                
            except Exception as e:
                print(f"Prediction error: {e}")
                import traceback
                print(traceback.format_exc())
        
        return {
            "timestamps": timestamps,
            "prices": prices.tolist(),
            "volumes": volumes.tolist(),
            "predictions": predictions,
            "prediction_dates": prediction_dates
        }
    except Exception as e:
        print(f"Error: {e}")
        return {"error": str(e)}

@app.get("/api/stats")
async def get_stats():
    try:
        global previous_price
        latest = collection.find_one({}, sort=[("timestamp", -1)])
        if not latest:
            return {"error": "No data available"}
        
        current_price = float(latest["close"])
        
        # Check for price alerts with detailed logging
        if previous_price is not None:
            try:
                logger.info(f"Checking price movement - Current: ${current_price:.2f}, Previous: ${previous_price:.2f}")
                change_percent = ((current_price - previous_price) / previous_price) * 100
                logger.info(f"Price change calculated: {change_percent:.2f}%")
                
                if abs(change_percent) >= 1:
                    logger.info("Significant price change detected, attempting to send alert...")
                    alert_system.check_price_movement(current_price, previous_price)
                else:
                    logger.debug("Price change below threshold, no alert needed")
            except Exception as alert_error:
                logger.error("Error in alert system:")
                logger.error(traceback.format_exc())
                print(f"Alert system error: {str(alert_error)}")
        else:
            logger.info("No previous price available for comparison")
        
        previous_price = current_price
        
        prev_day = collection.find_one(
            {"timestamp": {"$lt": latest["timestamp"]}},
            sort=[("timestamp", -1)]
        )
        
        daily_change = 0
        if prev_day:
            daily_change = ((current_price - float(prev_day["close"])) / float(prev_day["close"])) * 100
        
        # Get next day prediction if model is available
        next_day_prediction = None
        if model is not None and scaler is not None:
            try:
                recent_docs = list(collection.find().sort("timestamp", -1).limit(90))
                recent_docs.reverse()
                
                df = calculate_technical_indicators(recent_docs)
                feature_data = prepare_prediction_data(df)
                
                # Scale features
                scaled_data = scaler.transform(feature_data)
                
                recent_data = scaled_data[-60:]
                X = recent_data.reshape(1, 60, len(feature_data.columns))
                
                # Get prediction (this is a percentage change)
                pred = model.predict(X, verbose=0)
                predicted_change = float(pred[0, 0])  # This is the percentage change
                
                # Convert percentage change to actual price
                next_day_prediction = current_price * (1 + predicted_change)
                logger.info(f"Current price: ${current_price:.2f}")
                logger.info(f"Predicted change: {predicted_change*100:.2f}%")
                logger.info(f"Predicted price: ${next_day_prediction:.2f}")
                
            except Exception as e:
                logger.error("Prediction error in stats:")
                logger.error(traceback.format_exc())
                next_day_prediction = None
        
        return {
            "current_price": current_price,
            "daily_change": daily_change,
            "daily_volume": int(latest["volume"]),
            "daily_high": float(latest["high"]),
            "daily_low": float(latest["low"]),
            "next_day_prediction": next_day_prediction
        }
    except Exception as e:
        logger.error("Error in get_stats endpoint:")
        logger.error(traceback.format_exc())
        return {"error": str(e)}

@app.get("/api/hourly-predictions")
async def get_hourly_predictions():
    try:
        print("Fetching hourly predictions...")
        recent_docs = list(collection.find().sort("timestamp", -1).limit(90))
        recent_docs.reverse()
        
        if not recent_docs:
            return {"error": "No data available"}
        
        df = calculate_technical_indicators(recent_docs)
        feature_data = prepare_prediction_data(df)
        
        hourly_predictions = []
        prediction_times = []
        hours_predicted = 0
        
        if model is not None and scaler is not None and len(feature_data) >= 60:
            try:
                sequence_length = 60
                recent_data = feature_data.iloc[-sequence_length:]
                scaled_data = pd.DataFrame(
                    scaler.transform(recent_data),
                    columns=feature_data.columns,
                    index=recent_data.index
                )
                X = scaled_data.values.reshape(1, sequence_length, len(feature_data.columns))
                last_sequence = X.copy()
                
                # Get last known price and time
                current_price = float(df['close'].iloc[-1])
                last_time = pd.to_datetime(df.index[-1])
                
                # Convert to Eastern Time
                et_tz = pytz.timezone('US/Eastern')
                last_time = last_time.tz_localize('UTC').tz_convert(et_tz)
                
                # Start predictions from next trading hour
                next_time = last_time
                last_price = current_price
                days_predicted = 0
                max_days = 5  # Predict up to 5 trading days
                
                while days_predicted < max_days:
                    # Skip to next day if after market hours
                    if next_time.hour >= 16:
                        next_time = (next_time + timedelta(days=1)).replace(hour=9, minute=30)
                        days_predicted += 1
                        continue
                    
                    # Skip weekends
                    if next_time.weekday() >= 5:
                        days_to_monday = (7 - next_time.weekday()) + 1
                        next_time = (next_time + timedelta(days=days_to_monday)).replace(hour=9, minute=30)
                        continue
                    
                    # Skip before market hours
                    if next_time.hour < 9 or (next_time.hour == 9 and next_time.minute < 30):
                        next_time = next_time.replace(hour=9, minute=30)
                        continue
                        
                    pred = model.predict(last_sequence, verbose=0)
                    predicted_change = float(pred[0, 0])
                    
                    # Add some randomness to make predictions more realistic
                    volatility = 0.001  # 0.1% volatility
                    noise = np.random.normal(0, volatility)
                    predicted_change = predicted_change + noise
                    
                    # Add more volatility based on time of day
                    if next_time.hour in [9, 15]:  # More volatile at market open/close
                        predicted_change *= np.random.uniform(0.8, 1.2)
                    
                    pred_price = last_price * (1 + predicted_change)
                    
                    hourly_predictions.append({
                        'time': next_time.strftime("%Y-%m-%d %H:%M:%S"),
                        'price': pred_price,
                        'change': predicted_change * 100
                    })
                    
                    prediction_times.append(next_time.strftime("%I:%M %p"))
                    
                    # Update sequence with new features
                    new_row = np.zeros(len(feature_data.columns))
                    price_change = (pred_price - last_price) / last_price
                    new_row[0] = price_change
                    
                    # Add market dynamics
                    if price_change > 0:
                        volume_change = np.random.uniform(0, 0.1)
                    else:
                        volume_change = np.random.uniform(-0.1, 0)
                    new_row[1] = volume_change
                    
                    # Update technical indicators
                    new_row[4:] = last_sequence[0, -1, 4:]
                    
                    # Update sequence
                    last_sequence = np.roll(last_sequence, -1, axis=1)
                    last_sequence[0, -1] = new_row
                    
                    last_price = pred_price
                    next_time = next_time + timedelta(hours=1)
                    hours_predicted += 1
                
                print(f"Current price: ${current_price:.2f}")
                print(f"Generated {len(hourly_predictions)} predictions over {max_days} trading days")
                print("Sample predictions:")
                for p in hourly_predictions[:5]:
                    print(f"Time: {p['time']}, Price: ${p['price']:.2f}, Change: {p['change']:.2f}%")
                
            except Exception as e:
                print(f"Prediction error: {e}")
                import traceback
                print(traceback.format_exc())
                
        return {
            "predictions": hourly_predictions,
            "times": prediction_times,
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
    except Exception as e:
        print(f"Error: {e}")
        return {"error": str(e)}

@app.get("/")
async def home(request: Request):
    try:
        latest = collection.find_one({}, sort=[("timestamp", -1)])
        
        # Get next day prediction
        next_day_prediction = None
        if model is not None and scaler is not None:
            try:
                recent_docs = list(collection.find().sort("timestamp", -1).limit(90))
                recent_docs.reverse()
                
                df = calculate_technical_indicators(recent_docs)
                feature_data = prepare_prediction_data(df)
                
                # Scale features
                scaled_data = scaler.transform(feature_data)
                
                recent_data = scaled_data[-60:]
                X = recent_data.reshape(1, 60, len(feature_data.columns))
                
                pred = model.predict(X, verbose=0)
                dummy_data = pd.DataFrame(
                    [[pred[0, 0]] + [0] * (len(feature_data.columns)-1)],
                    columns=feature_data.columns
                )
                next_day_prediction = float(scaler.inverse_transform(dummy_data)[0, 0])
            except Exception as e:
                print(f"Prediction error in home: {e}")
                next_day_prediction = None
        
        stats = {
            "current_price": f"{float(latest['close']):.2f}" if latest else "0.00",
            "daily_high": f"{float(latest['high']):.2f}" if latest else "0.00",
            "daily_low": f"{float(latest['low']):.2f}" if latest else "0.00",
            "last_updated": latest["timestamp"] if latest else "No data",
            "next_day_prediction": f"{next_day_prediction:.2f}" if next_day_prediction else None
        }
        return templates.TemplateResponse("index.html", {"request": request, "stats": stats})
    except Exception as e:
        print(f"Error: {e}")
        return templates.TemplateResponse("index.html", {"request": request, "error": str(e)})

@app.get("/api/test-alert")
async def test_alert():
    try:
        # Simulate a significant price change
        test_current_price = 254.59  # Current price from logs
        test_previous_price = 250.00  # Simulated previous price to force >1% change
        
        logger.info(f"Testing alert with simulated prices - Current: ${test_current_price:.2f}, Previous: ${test_previous_price:.2f}")
        
        # Calculate change
        change_percent = ((test_current_price - test_previous_price) / test_previous_price) * 100
        logger.info(f"Simulated price change: {change_percent:.2f}%")
        
        # Force alert
        alert_system.check_price_movement(test_current_price, test_previous_price)
        
        return {
            "message": "Alert test initiated",
            "current_price": test_current_price,
            "previous_price": test_previous_price,
            "change_percent": change_percent
        }
    except Exception as e:
        logger.error("Error in test alert:")
        logger.error(traceback.format_exc())
        return {"error": str(e)}

@app.get("/api/alert-receivers")
async def get_alert_receivers():
    return {"receivers": alert_system.receivers}

class EmailRequest(BaseModel):
    email: str

@app.post("/api/alert-receivers")
async def add_alert_receiver(email_request: EmailRequest):
    print(f"Received request to add email: {email_request.email}")
    success = alert_system.add_receiver(email_request.email)
    return {
        "success": success,
        "message": "Email added successfully" if success else "Email already exists or invalid",
        "current_receivers": alert_system.receivers
    }

@app.delete("/api/alert-receivers/{email}")
async def remove_alert_receiver(email: str):
    success = alert_system.remove_receiver(email)
    return {"success": success, "message": "Email removed" if success else "Email not found"}

@app.get("/api/debug/receivers")
async def debug_receivers():
    try:
        with open(alert_system.receivers_file, 'r') as f:
            content = f.read()
        return {
            "file_content": content,
            "current_receivers": alert_system.receivers,
            "file_exists": os.path.exists(alert_system.receivers_file),
            "file_path": os.path.abspath(alert_system.receivers_file)
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/test/add-email/{email}")
async def test_add_email(email: str):
    try:
        # Add email
        success = alert_system.add_receiver(email)
        
        # Get current list
        current_list = alert_system.receivers
        
        # Check file
        file_exists = os.path.exists(alert_system.receivers_file)
        file_path = os.path.abspath(alert_system.receivers_file)
        
        try:
            with open(alert_system.receivers_file, 'r') as f:
                file_content = f.read()
        except:
            file_content = "Could not read file"
            
        return {
            "success": success,
            "current_list": current_list,
            "file_exists": file_exists,
            "file_path": file_path,
            "file_content": file_content
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
