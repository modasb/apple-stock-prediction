from confluent_kafka import Producer
import json
import yfinance as yf
from pymongo import MongoClient
from datetime import datetime, timedelta
import pandas as pd
import time
import pytz

# Initialize Confluent Kafka Producer
conf = {'bootstrap.servers': 'localhost:9092'}
producer = Producer(**conf)

# Initialize MongoDB Client
client = MongoClient('mongodb://localhost:27017/')
db = client['stock_data']
collection = db['stock_prices']

def is_market_hours(timestamp):
    """Check if timestamp is during market hours (9:30 AM - 4:00 PM ET)"""
    et_tz = pytz.timezone('US/Eastern')
    ts_et = timestamp.astimezone(et_tz)
    
    # Check if it's a weekday
    if ts_et.weekday() >= 5:  # Saturday = 5, Sunday = 6
        return False
    
    # Convert to time for hour comparison
    market_open = ts_et.replace(hour=9, minute=30, second=0)
    market_close = ts_et.replace(hour=16, minute=0, second=0)
    
    return market_open <= ts_et <= market_close

def fetch_and_process_chunk(start_date, end_date):
    print(f"Fetching data from {start_date} to {end_date}")
    
    try:
        # Fetch hourly data
        data = yf.download('AAPL', 
                          start=start_date, 
                          end=end_date,
                          interval='1h',
                          prepost=False)  # Don't include pre/post market
        
        if data.empty:
            print("No data received for this period")
            return
        
        # Reset index to make Datetime a column
        data = data.reset_index()
        
        # Process each row
        for idx in range(len(data)):
            try:
                timestamp = data['Datetime'].iloc[idx]
                
                # Skip if not during market hours
                if not is_market_hours(timestamp):
                    continue
                
                # Format the record properly
                record = {
                    'symbol': 'AAPL',
                    'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                    'open': float(data['Open'].iloc[idx].iloc[0]),
                    'high': float(data['High'].iloc[idx].iloc[0]),
                    'low': float(data['Low'].iloc[idx].iloc[0]),
                    'close': float(data['Close'].iloc[idx].iloc[0]),
                    'volume': int(data['Volume'].iloc[idx].iloc[0]),
                    'is_live': False,
                    'interval': '1h'
                }

                # Check for duplicate
                existing = collection.find_one({
                    'symbol': record['symbol'], 
                    'timestamp': record['timestamp']
                })
                
                if not existing:
                    # Send to Kafka
                    producer.produce(
                        'stock_prices',
                        key=record['timestamp'].encode('utf-8'),
                        value=json.dumps(record).encode('utf-8')
                    )
                    producer.poll(0)
                    
                    # Store in MongoDB
                    collection.insert_one(record)
                    print(f"Processed: {record['timestamp']} | Price: {record['close']}")
                else:
                    print(f"Skipping duplicate: {record['timestamp']}")
                    
            except Exception as e:
                print(f"Error processing record at index {idx}: {e}")
                print(f"Data at index: {data.iloc[idx]}")
                
    except Exception as e:
        print(f"Error processing chunk: {e}")

try:
    # Process data in chunks to handle Yahoo Finance limitations
    start_date = datetime(2023, 1, 1)
    end_date = datetime.now()
    chunk_size = timedelta(days=7)  # Process 7 days at a time
    
    current_start = start_date
    while current_start < end_date:
        chunk_end = min(current_start + chunk_size, end_date)
        fetch_and_process_chunk(current_start, chunk_end)
        
        # Wait between chunks to avoid rate limiting
        producer.flush()
        print(f"Completed chunk from {current_start} to {chunk_end}")
        print("Waiting 2 seconds before next chunk...")
        time.sleep(2)
        
        current_start = chunk_end

except Exception as e:
    print(f"Error in main process: {e}")

finally:
    # Final flush of Kafka messages
    producer.flush()
    # Close MongoDB Connection
    client.close()
    print("Data processing completed.")
