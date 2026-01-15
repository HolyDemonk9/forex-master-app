import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
from textblob import TextBlob
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import warnings
from typing import Dict, Tuple, Optional
import time
import sys
warnings.filterwarnings('ignore')

# ✅ MUST be the FIRST Streamlit command
try:
    st.set_page_config(
        page_title="Pro Forex Trading Dashboard",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
except Exception as e:
    st.warning(f"Page config warning: {e}")

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
    }
    .signal-strong-buy {
        background: linear-gradient(135deg, #4CAF50, #2E7D32);
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        font-size: 1.3rem;
        border: 2px solid #2E7D32;
    }
    .signal-strong-sell {
        background: linear-gradient(135deg, #F44336, #C62828);
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        font-size: 1.3rem;
        border: 2px solid #C62828;
    }
    .signal-moderate-buy {
        background: linear-gradient(135deg, #81C784, #4CAF50);
        color: white;
        padding: 12px;
        border-radius: 8px;
        text-align: center;
        font-weight: bold;
        font-size: 1.1rem;
    }
    .signal-moderate-sell {
        background: linear-gradient(135deg, #E57373, #F44336);
        color: white;
        padding: 12px;
        border-radius: 8px;
        text-align: center;
        font-weight: bold;
        font-size: 1.1rem;
    }
    .signal-neutral {
        background: linear-gradient(135deg, #FFB74D, #FF9800);
        color: white;
        padding: 12px;
        border-radius: 8px;
        text-align: center;
        font-weight: bold;
        font-size: 1.1rem;
    }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #E0E0E0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 15px;
    }
    .risk-warning {
        background: linear-gradient(135deg, #FFF3CD, #FFEEBA);
        border: 1px solid #FFC107;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">📊 Algorithmic Forex Trading Dashboard</h1>', unsafe_allow_html=True)

# Initialize session state
if 'last_success_fetch' not in st.session_state:
    st.session_state.last_success_fetch = None

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Trading Configuration")
    
    # Asset selection
    assets = {
        "EUR/USD": "EURUSD=X",
        "GBP/USD": "GBPUSD=X",
        "USD/JPY": "JPY=X",
        "AUD/USD": "AUDUSD=X",
        "USD/CAD": "CAD=X",
        "BTC/USD": "BTC-USD",
        "ETH/USD": "ETH-USD",
        "Gold": "GC=F",
        "Silver": "SI=F",
        "S&P 500": "^GSPC"
    }
    
    selected_asset = st.selectbox(
        "Select Asset:",
        list(assets.keys()),
        index=0
    )
    ticker = assets[selected_asset]
    
    # Trading mode selection
    trading_modes = {
        "Swing Trading": {
            "period": "1y",
            "interval": "1d",
            "higher_timeframe": "1wk",
            "description": "Daily data with weekly trend confirmation"
        },
        "Day Trading": {
            "period": "5d",
            "interval": "15m",
            "higher_timeframe": "1h",
            "description": "15-min data with hourly trend confirmation"
        },
        "Scalping": {
            "period": "1d",
            "interval": "5m",
            "higher_timeframe": "15m",
            "description": "5-min data with 15-min trend confirmation"
        },
        "Sniper Mode": {
            "period": "1d",
            "interval": "1m",
            "higher_timeframe": "5m",
            "description": "1-min data with 5-min trend confirmation"
        }
    }
    
    selected_mode = st.selectbox(
        "Select Trading Mode:",
        list(trading_modes.keys()),
        index=0
    )
    
    mode_config = trading_modes[selected_mode]
    st.markdown(f"**Strategy:** {mode_config['description']}")
    
    # Risk Management
    st.markdown("---")
    st.subheader("📊 Risk Management")
    
    account_balance = st.number_input(
        "Account Balance ($)",
        min_value=100.0,
        max_value=1000000.0,
        value=10000.0,
        step=1000.0
    )
    
    risk_per_trade = st.select_slider(
        "Risk per Trade (%)",
        options=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        value=1.0
    )
    
    atr_multiplier_sl = st.slider("Stop Loss (ATR)", 1.0, 3.0, 1.5, 0.1)
    atr_multiplier_tp = st.slider("Take Profit (ATR)", 1.0, 5.0, 3.0, 0.1)
    
    st.markdown("---")
    st.caption("⚠️ Trading involves risk. Past performance ≠ future results")

# Helper function to ensure 1D data for ta library
def ensure_1d(data):
    """Ensure data is 1-dimensional for ta library"""
    if isinstance(data, pd.DataFrame):
        if data.shape[1] == 1:
            return data.iloc[:, 0]
        else:
            raise ValueError("DataFrame must have exactly one column")
    elif isinstance(data, pd.Series):
        return data
    elif isinstance(data, np.ndarray):
        if data.ndim > 1:
            return data.squeeze()
        return data
    else:
        return data

@st.cache_data(ttl=60)
def fetch_data(ticker: str, period: str, interval: str, max_retries: int = 2) -> Optional[pd.DataFrame]:
    """Fetch data with retry logic"""
    for attempt in range(max_retries):
        try:
            # Download data
            data = yf.download(
                tickers=ticker,
                period=period,
                interval=interval,
                progress=False,
                timeout=10
            )
            
            if data.empty:
                continue
            
            # Ensure we have proper column names
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            # Rename columns to standard names
            column_mapping = {
                'Adj Close': 'Close',
                'adjclose': 'Close',
                'Adj Close': 'Close'
            }
            
            for old_col, new_col in column_mapping.items():
                if old_col in data.columns:
                    data[new_col] = data[old_col]
            
            # Ensure required columns exist
            required_cols = ['Open', 'High', 'Low', 'Close']
            for col in required_cols:
                if col not in data.columns:
                    # Try to find similar columns
                    for existing_col in data.columns:
                        if col.lower() in existing_col.lower():
                            data[col] = data[existing_col]
                            break
                    else:
                        # If still not found, use the last column
                        data[col] = data.iloc[:, -1]
            
            # Ensure Volume column exists
            if 'Volume' not in data.columns:
                data['Volume'] = 0
            
            # Convert index to datetime if needed
            if not isinstance(data.index, pd.DatetimeIndex):
                data.index = pd.to_datetime(data.index)
            
            st.session_state.last_success_fetch = datetime.now()
            return data
            
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
    
    st.error(f"Failed to fetch data for {ticker}")
    return None

def calculate_indicators(data: pd.DataFrame, mode: str) -> Tuple[pd.DataFrame, str, str, dict]:
    """Calculate technical indicators with proper 1D data handling"""
    df = data.copy()
    
    # Ensure we have enough data
    if len(df) < 20:
        return df, "NEUTRAL", "Insufficient data", {'signal_strength': 0, 'volume_strong': False}
    
    # Calculate volume metrics
    if 'Volume' in df.columns and len(df) > 20:
        try:
            # Ensure volume is 1D
            volume_1d = ensure_1d(df['Volume'])
            df['Volume_SMA_20'] = ta.trend.sma_indicator(volume_1d, window=20)
            df['Volume_Ratio'] = volume_1d / df['Volume_SMA_20'].replace(0, 1)
        except:
            df['Volume_Ratio'] = 1.0
    else:
        df['Volume_Ratio'] = 1.0
    
    # Calculate ATR with proper 1D data
    try:
        if len(df) > 14:
            # Ensure data is 1D
            high_1d = ensure_1d(df['High'])
            low_1d = ensure_1d(df['Low'])
            close_1d = ensure_1d(df['Close'])
            
            atr_indicator = ta.volatility.AverageTrueRange(
                high=high_1d,
                low=low_1d,
                close=close_1d,
                window=14
            )
            df['ATR'] = atr_indicator.average_true_range()
        else:
            df['ATR'] = np.nan
    except Exception as e:
        df['ATR'] = df['Close'].rolling(window=14).std().fillna(df['Close'].std())
    
    # Initialize signal metadata
    signal_metadata = {
        'signal_strength': 0,
        'volume_strong': False,
        'indicators': []
    }
    
    signal = "NEUTRAL"
    signal_reason = "No clear signal"
    
    try:
        # Ensure close price is 1D
        close_1d = ensure_1d(df['Close'])
        
        if mode == "Swing Trading":
            if len(df) >= 200:
                # Calculate SMAs
                df['SMA_50'] = ta.trend.sma_indicator(close_1d, window=50)
                df['SMA_200'] = ta.trend.sma_indicator(close_1d, window=200)
                signal_metadata['indicators'] = ['SMA_50', 'SMA_200']
                
                # Check volume
                volume_strong = df['Volume_Ratio'].iloc[-1] > 1.2
                signal_metadata['volume_strong'] = volume_strong
                
                # Generate signal
                if not pd.isna(df['SMA_50'].iloc[-1]) and not pd.isna(df['SMA_200'].iloc[-1]):
                    if df['SMA_50'].iloc[-1] > df['SMA_200'].iloc[-1]:
                        signal = "BUY"
                        signal_reason = "Golden Cross (SMA50 > SMA200)"
                        signal_metadata['signal_strength'] = 2 if volume_strong else 1
                    else:
                        signal = "SELL"
                        signal_reason = "Death Cross (SMA50 < SMA200)"
                        signal_metadata['signal_strength'] = 2 if volume_strong else 1
        
        elif mode == "Day Trading":
            if len(df) >= 21:
                # Calculate EMAs
                df['EMA_9'] = ta.trend.ema_indicator(close_1d, window=9)
                df['EMA_21'] = ta.trend.ema_indicator(close_1d, window=21)
                signal_metadata['indicators'] = ['EMA_9', 'EMA_21']
                
                # Check volume
                volume_strong = df['Volume_Ratio'].iloc[-1] > 1.3
                signal_metadata['volume_strong'] = volume_strong
                
                # Check for cross
                if len(df) > 1:
                    ema9_current = df['EMA_9'].iloc[-1]
                    ema21_current = df['EMA_21'].iloc[-1]
                    ema9_prev = df['EMA_9'].iloc[-2]
                    ema21_prev = df['EMA_21'].iloc[-2]
                    
                    if not (pd.isna(ema9_current) or pd.isna(ema21_current) or 
                           pd.isna(ema9_prev) or pd.isna(ema21_prev)):
                        
                        if ema9_prev <= ema21_prev and ema9_current > ema21_current:
                            signal = "BUY"
                            signal_reason = "EMA9 crossed above EMA21"
                            signal_metadata['signal_strength'] = 2 if volume_strong else 1
                        elif ema9_prev >= ema21_prev and ema9_current < ema21_current:
                            signal = "SELL"
                            signal_reason = "EMA9 crossed below EMA21"
                            signal_metadata['signal_strength'] = 2 if volume_strong else 1
        
        elif mode == "Scalping":
            if len(df) >= 14:
                # Calculate RSI
                df['RSI'] = ta.momentum.RSIIndicator(close_1d, window=14).rsi()
                signal_metadata['indicators'] = ['RSI']
                
                # Check volume
                volume_strong = df['Volume_Ratio'].iloc[-1] > 1.5
                signal_metadata['volume_strong'] = volume_strong
                
                # Generate signal
                current_rsi = df['RSI'].iloc[-1]
                if not pd.isna(current_rsi):
                    if current_rsi < 30:
                        signal = "BUY"
                        signal_reason = f"RSI oversold ({current_rsi:.1f})"
                        signal_metadata['signal_strength'] = 2 if volume_strong else 1
                    elif current_rsi > 70:
                        signal = "SELL"
                        signal_reason = f"RSI overbought ({current_rsi:.1f})"
                        signal_metadata['signal_strength'] = 2 if volume_strong else 1
        
        else:  # Sniper Mode
            if len(df) >= 20:
                # Calculate Bollinger Bands
                bb = ta.volatility.BollingerBands(close=close_1d, window=20, window_dev=2)
                df['BB_upper'] = bb.bollinger_hband()
                df['BB_middle'] = bb.bollinger_mavg()
                df['BB_lower'] = bb.bollinger_lband()
                
                # Calculate RSI
                df['RSI'] = ta.momentum.RSIIndicator(close_1d, window=14).rsi()
                signal_metadata['indicators'] = ['BB_upper', 'BB_middle', 'BB_lower', 'RSI']
                
                # Check volume
                volume_strong = df['Volume_Ratio'].iloc[-1] > 1.8
                signal_metadata['volume_strong'] = volume_strong
                
                # Generate signal
                current_price = df['Close'].iloc[-1]
                current_bb_lower = df['BB_lower'].iloc[-1]
                current_rsi = df['RSI'].iloc[-1]
                
                if (not pd.isna(current_bb_lower) and not pd.isna(current_rsi) and
                    current_price < current_bb_lower and current_rsi < 25):
                    signal = "BUY"
                    signal_reason = f"Price < Lower BB & RSI={current_rsi:.1f}"
                    signal_metadata['signal_strength'] = 3 if volume_strong else 2
                elif (not pd.isna(df['BB_upper'].iloc[-1]) and not pd.isna(current_rsi) and
                      current_price > df['BB_upper'].iloc[-1] and current_rsi > 75):
                    signal = "SELL"
                    signal_reason = f"Price > Upper BB & RSI={current_rsi:.1f}"
                    signal_metadata['signal_strength'] = 3 if volume_strong else 2
    
    except Exception as e:
        st.error(f"Indicator calculation error: {str(e)[:100]}")
        signal = "NEUTRAL"
        signal_reason = f"Calculation error: {str(e)[:50]}"
    
    return df, signal, signal_reason, signal_metadata

@st.cache_data(ttl=300)
def get_news_sentiment(ticker_symbol: str) -> Tuple[float, list, str]:
    """Get simplified news sentiment"""
    try:
        # Extract asset name
        asset_name = ticker_symbol.replace('=X', '').replace('-USD', '')
        
        # Try to fetch news
        url = f"https://news.google.com/rss/search?q={asset_name}+forex&hl=en-US&gl=US&ceid=US:en"
        
        response = requests.get(url, timeout=5)
        soup = BeautifulSoup(response.content, 'xml')
        
        headlines = []
        polarities = []
        
        for item in soup.find_all('item')[:5]:
            title = item.find('title').text
            if ' - ' in title:
                title = title.split(' - ')[0]
            
            headlines.append(title)
            
            # Analyze sentiment
            blob = TextBlob(title)
            polarities.append(blob.sentiment.polarity)
        
        if polarities:
            avg_polarity = np.mean(polarities)
            
            if avg_polarity > 0.1:
                sentiment = "POSITIVE"
            elif avg_polarity < -0.1:
                sentiment = "NEGATIVE"
            else:
                sentiment = "NEUTRAL"
            
            return avg_polarity, headlines, sentiment
        else:
            return 0.0, ["No news available"], "NEUTRAL"
            
    except Exception as e:
        return 0.0, [f"News error: {str(e)[:30]}"], "NEUTRAL"

def calculate_position_size(account_balance: float, risk_percent: float, 
                          entry_price: float, stop_loss: float) -> Dict:
    """Calculate position size"""
    risk_amount = account_balance * (risk_percent / 100)
    
    # Calculate risk per unit
    if entry_price > 0 and abs(entry_price - stop_loss) > 0:
        risk_per_unit = abs(entry_price - stop_loss)
        position_units = risk_amount / risk_per_unit
        position_value = position_units * entry_price
        position_percent = (position_value / account_balance) * 100
    else:
        position_units = 0
        position_value = 0
        position_percent = 0
        risk_per_unit = 0
    
    return {
        'risk_amount': risk_amount,
        'position_units': position_units,
        'position_value': position_value,
        'position_percent': min(position_percent, 10.0),  # Cap at 10%
        'risk_per_unit': risk_per_unit
    }

def calculate_risk_metrics(current_price: float, atr_value: float, 
                          atr_multiplier_sl: float, atr_multiplier_tp: float,
                          signal: str) -> Dict:
    """Calculate risk metrics"""
    if pd.isna(atr_value) or atr_value <= 0:
        atr_value = current_price * 0.01  # Default 1%
    
    # Calculate SL and TP
    if signal == "BUY":
        stop_loss = current_price - (atr_value * atr_multiplier_sl)
        take_profit = current_price + (atr_value * atr_multiplier_tp)
    elif signal == "SELL":
        stop_loss = current_price + (atr_value * atr_multiplier_sl)
        take_profit = current_price - (atr_value * atr_multiplier_tp)
    else:
        stop_loss = current_price - (atr_value * atr_multiplier_sl)
        take_profit = current_price + (atr_value * atr_multiplier_tp)
    
    # Calculate distances
    sl_distance = abs(current_price - stop_loss)
    tp_distance = abs(current_price - take_profit)
    rr_ratio = tp_distance / sl_distance if sl_distance > 0 else 0
    
    return {
        'stop_loss': stop_loss,
        'take_profit': take_profit,
        'sl_distance': sl_distance,
        'tp_distance': tp_distance,
        'risk_reward_ratio': rr_ratio,
        'atr_value': atr_value
    }

def generate_decision(technical_signal: str, signal_strength: int, 
                     news_sentiment: str, news_polarity: float,
                     volume_strong: bool) -> Tuple[str, str, str, float]:
    """Generate trading decision"""
    # Calculate technical score (70%)
    tech_score = 0
    
    if technical_signal == "BUY":
        tech_score = 70
        tech_score += signal_strength * 10
    elif technical_signal == "SELL":
        tech_score = 30
        tech_score -= signal_strength * 10
    
    # Volume bonus
    if volume_strong:
        tech_score += 10
    
    # Sentiment score (30%)
    sentiment_map = {
        "POSITIVE": 25,
        "NEUTRAL": 15,
        "NEGATIVE": 10
    }
    
    sent_score = sentiment_map.get(news_sentiment, 15)
    
    # Adjust by polarity
    sent_score += news_polarity * 10
    
    # Combined score
    combined_score = (tech_score * 0.7) + (sent_score * 0.3)
    
    # Generate decision
    if technical_signal == "BUY":
        if combined_score >= 75:
            decision = "STRONG BUY SIGNAL"
            decision_class = "signal-strong-buy"
            confidence = "HIGH"
        elif combined_score >= 60:
            decision = "MODERATE BUY SIGNAL"
            decision_class = "signal-moderate-buy"
            confidence = "MEDIUM"
        else:
            decision = "WEAK BUY / HOLD"
            decision_class = "signal-neutral"
            confidence = "LOW"
    
    elif technical_signal == "SELL":
        if combined_score <= 25:
            decision = "STRONG SELL SIGNAL"
            decision_class = "signal-strong-sell"
            confidence = "HIGH"
        elif combined_score <= 40:
            decision = "MODERATE SELL SIGNAL"
            decision_class = "signal-moderate-sell"
            confidence = "MEDIUM"
        else:
            decision = "WEAK SELL / HOLD"
            decision_class = "signal-neutral"
            confidence = "LOW"
    
    else:
        decision = "NO CLEAR SIGNAL - HOLD"
        decision_class = "signal-neutral"
        confidence = "WAIT"
    
    return decision, decision_class, confidence, combined_score

def create_chart(data: pd.DataFrame, indicators: list, mode: str):
    """Create price chart with indicators"""
    try:
        # Get last 100 periods for cleaner chart
        chart_data = data.tail(100).copy()
        
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.7, 0.3],
            subplot_titles=(f"{selected_asset} Price", "Volume/RSI")
        )
        
        # Candlestick
        fig.add_trace(
            go.Candlestick(
                x=chart_data.index,
                open=chart_data['Open'],
                high=chart_data['High'],
                low=chart_data['Low'],
                close=chart_data['Close'],
                name="Price",
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Add indicators based on mode
        if mode == "Swing Trading" and 'SMA_50' in chart_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=chart_data.index,
                    y=chart_data['SMA_50'],
                    mode='lines',
                    name='SMA 50',
                    line=dict(color='orange', width=2)
                ),
                row=1, col=1
            )
            
            if 'SMA_200' in chart_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=chart_data.index,
                        y=chart_data['SMA_200'],
                        mode='lines',
                        name='SMA 200',
                        line=dict(color='blue', width=2)
                    ),
                    row=1, col=1
                )
        
        elif mode == "Day Trading" and 'EMA_9' in chart_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=chart_data.index,
                    y=chart_data['EMA_9'],
                    mode='lines',
                    name='EMA 9',
                    line=dict(color='red', width=1.5)
                ),
                row=1, col=1
            )
            
            if 'EMA_21' in chart_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=chart_data.index,
                        y=chart_data['EMA_21'],
                        mode='lines',
                        name='EMA 21',
                        line=dict(color='green', width=1.5)
                    ),
                    row=1, col=1
                )
        
        elif mode == "Sniper Mode" and 'BB_upper' in chart_data.columns:
            # Bollinger Bands
            fig.add_trace(
                go.Scatter(
                    x=chart_data.index,
                    y=chart_data['BB_upper'],
                    mode='lines',
                    name='BB Upper',
                    line=dict(color='gray', width=1, dash='dash'),
                    opacity=0.7
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=chart_data.index,
                    y=chart_data['BB_middle'],
                    mode='lines',
                    name='BB Middle',
                    line=dict(color='gray', width=1),
                    opacity=0.7
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=chart_data.index,
                    y=chart_data['BB_lower'],
                    mode='lines',
                    name='BB Lower',
                    line=dict(color='gray', width=1, dash='dash'),
                    opacity=0.7,
                    fill='tonexty',
                    fillcolor='rgba(128, 128, 128, 0.1)'
                ),
                row=1, col=1
            )
        
        # Bottom subplot
        if 'RSI' in chart_data.columns:
            # RSI plot
            fig.add_trace(
                go.Scatter(
                    x=chart_data.index,
                    y=chart_data['RSI'],
                    mode='lines',
                    name='RSI',
                    line=dict(color='purple', width=2)
                ),
                row=2, col=1
            )
            
            # Add RSI reference lines
            fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)
            fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.3, row=2, col=1)
            
            fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
        else:
            # Volume plot
            colors = ['red' if chart_data['Close'].iloc[i] < chart_data['Open'].iloc[i] else 'green' 
                     for i in range(len(chart_data))]
            
            fig.add_trace(
                go.Bar(
                    x=chart_data.index,
                    y=chart_data['Volume'] if 'Volume' in chart_data.columns else [0] * len(chart_data),
                    name='Volume',
                    marker_color=colors,
                    opacity=0.5
                ),
                row=2, col=1
            )
            
            fig.update_yaxes(title_text="Volume", row=2, col=1)
        
        fig.update_layout(
            title=f"{selected_mode} - {selected_asset}",
            template="plotly_white",
            height=600,
            showlegend=True,
            xaxis_rangeslider_visible=False
        )
        
        return fig
        
    except Exception as e:
        # Simple fallback chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data.index[-50:],
            y=data['Close'].iloc[-50:],
            mode='lines',
            name='Price'
        ))
        fig.update_layout(title=f"{selected_asset} Price", height=400)
        return fig

# Main app logic
def main():
    try:
        # Fetch data
        with st.spinner(f"📊 Loading {selected_asset} data..."):
            data = fetch_data(ticker, mode_config['period'], mode_config['interval'])
            
            if data is None or data.empty:
                st.error("❌ Failed to load market data. Please try again.")
                if st.session_state.last_success_fetch:
                    st.info(f"Last successful fetch: {st.session_state.last_success_fetch}")
                return
        
        # Calculate indicators
        data_with_indicators, technical_signal, technical_reason, signal_metadata = calculate_indicators(
            data, selected_mode
        )
        
        # Get current metrics
        current_price = data_with_indicators['Close'].iloc[-1]
        previous_price = data_with_indicators['Close'].iloc[-2] if len(data_with_indicators) > 1 else current_price
        price_change_pct = ((current_price - previous_price) / previous_price) * 100 if previous_price != 0 else 0
        
        current_atr = data_with_indicators['ATR'].iloc[-1] if 'ATR' in data_with_indicators.columns else current_price * 0.01
        
        # Get news sentiment
        with st.spinner("📰 Analyzing market sentiment..."):
            news_polarity, news_headlines, news_sentiment = get_news_sentiment(ticker)
        
        # Generate decision
        decision, decision_class, confidence, score = generate_decision(
            technical_signal,
            signal_metadata['signal_strength'],
            news_sentiment,
            news_polarity,
            signal_metadata['volume_strong']
        )
        
        # Calculate risk metrics
        risk_metrics = calculate_risk_metrics(
            current_price, current_atr,
            atr_multiplier_sl, atr_multiplier_tp,
            technical_signal
        )
        
        # Calculate position size
        position_metrics = calculate_position_size(
            account_balance, risk_per_trade,
            current_price, risk_metrics['stop_loss']
        )
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(
                label="💰 Current Price",
                value=f"${current_price:.4f}" if current_price < 100 else f"${current_price:.2f}",
                delta=f"{price_change_pct:+.2f}%"
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(
                label="🎯 Technical Signal",
                value=technical_signal,
                delta=technical_reason[:30] + "..." if len(technical_reason) > 30 else technical_reason
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(
                label="📰 News Sentiment",
                value=news_sentiment,
                delta=f"Score: {news_polarity:.3f}"
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric(
                    label="🛑 Stop Loss",
                    value=f"${risk_metrics['stop_loss']:.4f}",
                    delta=f"{(risk_metrics['stop_loss'] - current_price):+.4f}"
                )
            with col_b:
                st.metric(
                    label="🎯 Take Profit",
                    value=f"${risk_metrics['take_profit']:.4f}",
                    delta=f"{(risk_metrics['take_profit'] - current_price):+.4f}"
                )
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(
                label="📊 Position Size",
                value=f"${position_metrics['position_value']:.0f}",
                delta=f"{position_metrics['position_percent']:.1f}%"
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(
                label="⚡ Signal Score",
                value=f"{score:.0f}/100",
                delta=confidence
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Display decision
        st.markdown(f'<div class="{decision_class}">', unsafe_allow_html=True)
        st.markdown(f"## {decision}")
        st.markdown(f"**Score:** {score:.1f}/100 | **Confidence:** {confidence}")
        st.markdown(f"**Technical:** {technical_signal} | **Sentiment:** {news_sentiment}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Chart
        st.markdown("---")
        st.subheader("📈 Price Analysis")
        
        fig = create_chart(data_with_indicators, signal_metadata.get('indicators', []), selected_mode)
        st.plotly_chart(fig, use_container_width=True)
        
        # Tabs for additional information
        tab1, tab2, tab3 = st.tabs(["📊 Market Data", "📰 News", "⚙️ Trade Setup"])
        
        with tab1:
            # Show recent data
            display_cols = ['Open', 'High', 'Low', 'Close']
            if 'Volume' in data_with_indicators.columns:
                display_cols.append('Volume')
            if 'Volume_Ratio' in data_with_indicators.columns:
                display_cols.append('Volume_Ratio')
            
            display_data = data_with_indicators[display_cols].tail(10).copy()
            st.dataframe(display_data.style.format("{:.4f}"))
            
            # Volume analysis
            if 'Volume_Ratio' in data_with_indicators.columns:
                current_volume_ratio = data_with_indicators['Volume_Ratio'].iloc[-1]
                st.progress(
                    min(current_volume_ratio / 2.0, 1.0),
                    text=f"Volume Ratio: {current_volume_ratio:.2f}x average"
                )
        
        with tab2:
            if news_headlines:
                st.subheader("Recent Market News")
                for i, headline in enumerate(news_headlines[:5], 1):
                    blob = TextBlob(headline)
                    polarity = blob.sentiment.polarity
                    
                    # Determine sentiment color
                    if polarity > 0.1:
                        icon = "🟢"
                    elif polarity < -0.1:
                        icon = "🔴"
                    else:
                        icon = "⚪"
                    
                    st.write(f"{icon} **{headline}**")
                    st.caption(f"Sentiment: {polarity:.3f}")
                    st.divider()
            else:
                st.info("No news headlines available.")
        
        with tab3:
            col_t1, col_t2 = st.columns(2)
            
            with col_t1:
                st.markdown("#### 📊 Technical Setup")
                st.write(f"**Signal:** {technical_signal}")
                st.write(f"**Strength:** {signal_metadata['signal_strength']}/3")
                st.write(f"**Volume Confirmation:** {'✅ Strong' if signal_metadata['volume_strong'] else '⚠️ Weak'}")
                st.write(f"**Price Change:** {price_change_pct:+.2f}%")
                
                # ATR info
                if 'ATR' in data_with_indicators.columns:
                    atr_percent = (risk_metrics['atr_value'] / current_price) * 100
                    st.write(f"**ATR Volatility:** {atr_percent:.2f}%")
            
            with col_t2:
                st.markdown("#### 💼 Risk Management")
                st.write(f"**Account Balance:** ${account_balance:,.2f}")
                st.write(f"**Risk Amount:** ${position_metrics['risk_amount']:.2f} ({risk_per_trade}%)")
                st.write(f"**Stop Loss:** ${risk_metrics['stop_loss']:.4f}")
                st.write(f"**Take Profit:** ${risk_metrics['take_profit']:.4f}")
                st.write(f"**Risk/Reward:** {risk_metrics['risk_reward_ratio']:.2f}:1")
                st.write(f"**Position Size:** ${position_metrics['position_value']:.2f}")
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div class="risk-warning">
        ⚠️ **DISCLAIMER:** This dashboard is for educational purposes only. 
        Forex trading involves substantial risk of loss. Past performance does not guarantee future results.
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
        st.info("Please try refreshing the page or selecting different parameters.")

# Run the app
if __name__ == "__main__":
    main()
