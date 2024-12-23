import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import joblib
from pymongo import MongoClient
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Get the current directory
BASE_DIR = r"C:\Users\ousse\Desktop\data dec"

# MongoDB connection
client = MongoClient('mongodb://localhost:27017/')
db = client['stock_data']
collection = db['stock_prices']

def add_technical_indicators(df):
    # Moving averages
    df['SMA_5'] = df['close'].rolling(window=5).mean()
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['EMA_5'] = df['close'].ewm(span=5, adjust=False).mean()
    df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()
    
    # MACD
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['BB_middle'] = df['close'].rolling(window=20).mean()
    df['BB_upper'] = df['BB_middle'] + 2 * df['close'].rolling(window=20).std()
    df['BB_lower'] = df['BB_middle'] - 2 * df['close'].rolling(window=20).std()
    
    # Volume indicators
    df['Volume_MA'] = df['volume'].rolling(window=20).mean()
    df['Volume_std'] = df['volume'].rolling(window=20).std()
    
    return df

def prepare_data(sequence_length=60):
    print("Preparing data...")
    data = pd.DataFrame(list(collection.find(
        {},
        {'_id': 0, 'timestamp': 1, 'close': 1, 'volume': 1, 'high': 1, 'low': 1, 'open': 1}
    ).sort('timestamp', 1)))
    
    if data.empty:
        raise ValueError("No data found in MongoDB")
    
    print(f"Loaded {len(data)} records from MongoDB")
    
    # Convert timestamp to datetime and set as index
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data.set_index('timestamp', inplace=True)
    
    # Handle zero values before calculating percentage changes
    for col in ['close', 'high', 'low', 'open', 'volume']:
        data[col] = data[col].replace(0, np.nan)
        data[col] = data[col].fillna(method='ffill')
    
    # Calculate percentage changes and handle infinite values
    price_columns = ['close', 'high', 'low', 'open']
    for col in price_columns:
        data[f'{col}_pct'] = data[col].pct_change()
        # Replace infinite values with the maximum/minimum non-infinite values
        data[f'{col}_pct'] = data[f'{col}_pct'].replace([np.inf, -np.inf], np.nan)
        data[f'{col}_pct'] = data[f'{col}_pct'].fillna(method='ffill')
    
    # Normalize volume with log transformation instead of percentage change
    data['volume_log'] = np.log1p(data['volume'])
    
    # Add technical indicators
    data = add_technical_indicators(data)
    
    # Handle any remaining NaN or infinite values
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.fillna(method='ffill')
    data = data.fillna(method='bfill')  # For any remaining NaN at the beginning
    
    # Select features
    features = [
        'close_pct', 'volume_log', 'high_pct', 'low_pct', 'open_pct',
        'SMA_5', 'SMA_20', 'EMA_5', 'EMA_20', 
        'MACD', 'Signal_Line', 'RSI',
        'BB_middle', 'BB_upper', 'BB_lower',
        'Volume_MA', 'Volume_std'
    ]
    
    feature_data = data[features]
    
    # Verify no infinite values
    if np.any(np.isinf(feature_data.values)):
        raise ValueError("Infinite values found in feature data")
    
    # Store original close prices for later denormalization
    close_prices = data['close']
    
    # Scale the features
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(feature_data)
    scaled_df = pd.DataFrame(scaled_data, columns=features, index=feature_data.index)
    
    X, y = [], []
    for i in range(len(scaled_df) - sequence_length):
        X.append(scaled_df.iloc[i:(i + sequence_length)].values)
        # Target is the next day's return (with clipping to handle extreme values)
        current_price = close_prices.iloc[i + sequence_length - 1]
        next_price = close_prices.iloc[i + sequence_length]
        pct_change = (next_price - current_price) / current_price
        # Clip extreme values to ±20%
        pct_change = np.clip(pct_change, -0.2, 0.2)
        y.append(pct_change)
    
    X = np.array(X)
    y = np.array(y)
    
    # Print data statistics
    print("\nData Statistics:")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"X range: [{X.min():.2f}, {X.max():.2f}]")
    print(f"y range: [{y.min():.2f}, {y.max():.2f}]")
    
    # Split into train and test
    train_size = int(len(X) * 0.8)
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_test = X[train_size:]
    y_test = y[train_size:]
    
    print(f"\nTraining samples: {len(X_train)}, Test samples: {len(X_test)}")
    return X_train, y_train, X_test, y_test, scaler, features, close_prices

def create_model(input_shape):
    model = Sequential([
        # First LSTM layer
        Bidirectional(LSTM(64, return_sequences=True, 
                          kernel_regularizer=tf.keras.regularizers.l2(0.01)),
                     input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.2),
        
        # Second LSTM layer
        Bidirectional(LSTM(32, return_sequences=True,
                          kernel_regularizer=tf.keras.regularizers.l2(0.01))),
        BatchNormalization(),
        Dropout(0.2),
        
        # Third LSTM layer
        LSTM(16, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        BatchNormalization(),
        Dropout(0.2),
        
        # Dense layers
        Dense(16, activation='relu',
              kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        BatchNormalization(),
        Dropout(0.1),
        Dense(8, activation='relu',
              kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        BatchNormalization(),
        Dense(1, activation='tanh')  # Output is percentage change
    ])
    
    optimizer = Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss='huber',
        metrics=['mae', 'mse']
    )
    
    return model

def plot_training_history(history):
    plt.figure(figsize=(15, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot MAE
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'training_history.png'))
    plt.close()

def train_model(X_train, y_train, X_test, y_test, epochs=100):
    print("Creating model...")
    model = create_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        mode='min'
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6,
        mode='min'
    )
    
    model_checkpoint = ModelCheckpoint(
        os.path.join(BASE_DIR, 'best_model.keras'),
        monitor='val_loss',
        save_best_only=True,
        mode='min'
    )
    
    print("Training model...")
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping, reduce_lr, model_checkpoint],
        verbose=1
    )
    
    # Plot training history
    plot_training_history(history)
    
    return model, history

def evaluate_model(model, X_test, y_test, close_prices, scaler):
    # Make predictions (these are percentage changes)
    y_pred = model.predict(X_test)
    
    # Convert percentage changes back to actual prices
    test_start_idx = len(close_prices) - len(y_test)
    actual_prices = []
    predicted_prices = []
    last_price = close_prices.iloc[test_start_idx - 1]
    
    for i in range(len(y_test)):
        # Actual
        actual_change = y_test[i]
        actual_price = last_price * (1 + actual_change)
        actual_prices.append(actual_price)
        
        # Predicted
        pred_change = y_pred[i][0]
        pred_price = last_price * (1 + pred_change)
        predicted_prices.append(pred_price)
        
        last_price = actual_price
    
    # Calculate metrics
    mse = np.mean((np.array(actual_prices) - np.array(predicted_prices)) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(np.array(actual_prices) - np.array(predicted_prices)))
    
    print("\nModel Evaluation:")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    
    # Plot predictions vs actual
    plt.figure(figsize=(15, 6))
    plt.plot(actual_prices, label='Actual')
    plt.plot(predicted_prices, label='Predicted')
    plt.title('Actual vs Predicted Stock Prices')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig(os.path.join(BASE_DIR, 'predictions.png'))
    plt.close()

if __name__ == "__main__":
    try:
        # Prepare data
        X_train, y_train, X_test, y_test, scaler, features, close_prices = prepare_data()
        
        # Train model
        model, history = train_model(X_train, y_train, X_test, y_test)
        
        # Evaluate model
        evaluate_model(model, X_test, y_test, close_prices, scaler)
        
        # Save model and scaler
        model.save(os.path.join(BASE_DIR, 'model.keras'))
        joblib.dump(scaler, os.path.join(BASE_DIR, 'scaler.pkl'))
        
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        print(traceback.format_exc()) 