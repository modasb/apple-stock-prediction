import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import torch
from lstm_predictor import LSTMPredictor

def create_prediction_charts(collection, model, scaler, prepare_data_for_prediction):
    try:
        # Get historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        historical_data = pd.DataFrame(
            list(collection.find({
                "timestamp": {
                    "$gte": start_date.strftime("%Y-%m-%d"),
                    "$lte": end_date.strftime("%Y-%m-%d")
                }
            }).sort("timestamp", 1))
        )
        
        if historical_data.empty:
            return "<p>No data available</p>"
        
        historical_data['timestamp'] = pd.to_datetime(historical_data['timestamp'])
        historical_data.set_index('timestamp', inplace=True)
        
        # Initialize predictor and get predictions
        predictor = LSTMPredictor(model, scaler)
        future_data = predictor.predict_future(historical_data, days=14)
        
        if future_data is not None:
            # Create figure with subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Historical Price & 2-Week Prediction',
                    'Detailed Prediction View',
                    'Prediction Confidence',
                    'Daily Price Changes'
                ),
                specs=[
                    [{"secondary_y": True}, {}],
                    [{"type": "indicator"}, {}]
                ]
            )
            
            # 1. Main Chart with Historical + Predictions
            fig.add_trace(
                go.Candlestick(
                    x=historical_data.index,
                    open=historical_data['open'],
                    high=historical_data['high'],
                    low=historical_data['low'],
                    close=historical_data['close'],
                    name="Historical Data"
                ),
                row=1, col=1
            )
            
            # Add prediction line
            fig.add_trace(
                go.Scatter(
                    x=future_data['timestamp'],
                    y=future_data['predicted_close'],
                    name="2-Week Prediction",
                    line=dict(color='yellow', width=2, dash='dash')
                ),
                row=1, col=1
            )
            
            # 2. Detailed Prediction View
            fig.add_trace(
                go.Scatter(
                    x=future_data['timestamp'],
                    y=future_data['predicted_close'],
                    name="Detailed Prediction",
                    line=dict(color='cyan'),
                    mode='lines+markers'
                ),
                row=1, col=2
            )
            
            # Add prediction range
            fig.add_trace(
                go.Scatter(
                    x=future_data['timestamp'],
                    y=future_data['predicted_close'] * 1.05,
                    fill=None,
                    mode='lines',
                    line_color='rgba(0, 255, 255, 0.1)',
                    name='Upper Bound'
                ),
                row=1, col=2
            )
            
            fig.add_trace(
                go.Scatter(
                    x=future_data['timestamp'],
                    y=future_data['predicted_close'] * 0.95,
                    fill='tonexty',
                    mode='lines',
                    line_color='rgba(0, 255, 255, 0.1)',
                    name='Lower Bound'
                ),
                row=1, col=2
            )
            
            # 3. Prediction Confidence
            accuracy = 85  # Replace with actual model accuracy
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=accuracy,
                    title={'text': "Prediction Confidence"},
                    delta={'reference': 80},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "yellow"},
                        'steps': [
                            {'range': [0, 50], 'color': "red"},
                            {'range': [50, 75], 'color': "orange"},
                            {'range': [75, 100], 'color': "green"}
                        ]
                    }
                ),
                row=2, col=1
            )
            
            # 4. Daily Price Changes
            daily_changes = future_data['predicted_close'].diff()
            fig.add_trace(
                go.Bar(
                    x=future_data['timestamp'],
                    y=daily_changes,
                    name="Daily Changes",
                    marker_color=np.where(daily_changes >= 0, 'green', 'red')
                ),
                row=2, col=2
            )
            
            # Update layout
            fig.update_layout(
                title='AAPL Stock 2-Week Price Prediction',
                template='plotly_dark',
                height=800,
                showlegend=True,
                xaxis_rangeslider_visible=False
            )
            
            # Add hover tooltips
            fig.update_traces(
                hovertemplate="<br>".join([
                    "Date: %{x}",
                    "Price: $%{y:.2f}",
                    "<extra></extra>"
                ])
            )
            
            return fig.to_html(full_html=False, include_plotlyjs=True)
            
    except Exception as e:
        print(f"Error in create_prediction_charts: {e}")
        return f"<p>Error creating charts: {str(e)}</p>" 