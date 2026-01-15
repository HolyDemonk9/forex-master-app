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

# ✅ MUST be the FIRST Streamlit command, before any other st. commands
try:
    st.set_page_config(
        page_title="Pro Forex Trading Dashboard",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
except Exception as e:
    # If set_page_config fails, we'll handle it gracefully
    print(f"Page config warning: {e}")

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
    .stButton>button {
        background: linear-gradient(135deg, #1E88E5, #1565C0);
        color: white;
        font-weight: bold;
        border: none;
        padding: 10px 24px;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">📊 Algorithmic Forex Trading Dashboard</h1>', unsafe_allow_html=True)

# Initialize session state for caching
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
    
    # Risk Management Section
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
    
    atr_multiplier_sl = st.slider(
        "Stop Loss (ATR Multiplier)",
        1.0, 3.0, 1.5, 0.1
    )
    atr_multiplier_tp = st.slider(
        "Take Profit (ATR Multiplier)",
        1.0, 5.0, 3.0, 0.1
    )
    
    enable_trailing = st.checkbox("Enable Trailing Stop", value=True)
    trailing_atr_multiplier = st.slider(
        "Trailing Stop Distance (ATR)",
        0.5, 2.0, 1.0, 0.1,
        disabled=not enable_trailing
    )
    
    st.markdown("---")
    st.caption("⚠️ Trading involves risk. Past performance ≠ future results")
    st.caption("📊 Data: Yahoo Finance | 🎯 Signals: Algorithmic Analysis")

# Improved fetch_data function with better error handling
@st.cache_data(ttl=60)
def fetch_data(ticker: str, period: str, interval: str, max_retries: int = 2) -> Optional[pd.DataFrame]:
    """Fetch data with retry logic and error handling"""
    for attempt in range(max_retries):
        try:
            data = yf.download(
                tickers=ticker,
                period=period,
                interval=interval,
                progress=False,
                timeout=10
            )
            
            if data.empty:
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                return None
            
            # Ensure required columns
            required = ['Open', 'High', 'Low', 'Close']
            for col in required:
                if col not in data.columns:
                    data[col] = data['Close'] if 'Close' in data.columns else data.iloc[:, -1]
            
            if 'Volume' not in data.columns:
                data['Volume'] = 0
            
            st.session_state.last_success_fetch = datetime.now()
            return data
            
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            st.error(f"Failed to fetch data after {max_retries} attempts")
            return None
    
    return None

def fetch_higher_timeframe_data(ticker: str, base_interval: str, higher_interval: str) -> Optional[pd.DataFrame]:
    """Fetch higher timeframe data for trend confirmation"""
    try:
        period_map = {
            '1m': '1d', '5m': '1d', '15m': '5d',
            '1h': '1mo', '1d': '6mo', '1wk': '2y'
        }
        
        period = period_map.get(higher_interval, '1mo')
        
        data = yf.download(
            tickers=ticker,
            period=period,
            interval=higher_interval,
            progress=False,
            timeout=10
        )
        
        if not data.empty and 'Close' in data.columns:
            return data
    except:
        pass
    
    return None

def calculate_higher_timeframe_trend(data: pd.DataFrame) -> Tuple[str, str]:
    """Determine trend direction on higher timeframe"""
    if data is None or len(data) < 20:
        return "NEUTRAL", "Insufficient data"
    
    try:
        df = data.copy()
        
        # Calculate simple moving averages
        if len(df) >= 50:
            df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
            df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
            
            current_price = df['Close'].iloc[-1]
            sma20 = df['SMA_20'].iloc[-1]
            sma50 = df['SMA_50'].iloc[-1]
            
            if current_price > sma20 > sma50:
                return "BULLISH", "Price > SMA20 > SMA50"
            elif current_price < sma20 < sma50:
                return "BEARISH", "Price < SMA20 < SMA50"
        
        return "NEUTRAL", "No clear trend"
    except:
        return "NEUTRAL", "Calculation error"

def calculate_indicators(data: pd.DataFrame, mode: str, higher_tf_data: pd.DataFrame = None) -> Tuple[pd.DataFrame, str, str, list, dict]:
    """Calculate technical indicators with multi-timeframe confluence"""
    df = data.copy()
    
    # Get higher timeframe trend
    higher_tf_trend, higher_tf_reason = calculate_higher_timeframe_trend(higher_tf_data)
    
    # Calculate volume metrics
    if 'Volume' in df.columns and len(df) > 20:
        df['Volume_SMA_20'] = ta.trend.sma_indicator(df['Volume'], window=20)
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_20'].replace(0, 1)
    else:
        df['Volume_Ratio'] = 1.0
    
    # Calculate ATR
    if len(df) > 14:
        try:
            df['ATR'] = ta.volatility.AverageTrueRange(
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                window=14
            ).average_true_range()
        except:
            df['ATR'] = df['Close'].rolling(14).std()
    else:
        df['ATR'] = np.nan
    
    # Initialize signal metadata
    signal_metadata = {
        'volume_strong': False,
        'higher_tf_aligned': False,
        'signal_strength': 0
    }
    
    signal = "NEUTRAL"
    signal_reason = ""
    indicators = []
    
    try:
        if mode == "Swing Trading":
            df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
            df['SMA_200'] = ta.trend.sma_indicator(df['Close'], window=200)
            indicators = ['SMA_50', 'SMA_200']
            
            if len(df) >= 200:
                volume_strong = df['Volume_Ratio'].iloc[-1] > 1.2
                signal_metadata['volume_strong'] = volume_strong
                
                if df['SMA_50'].iloc[-1] > df['SMA_200'].iloc[-1]:
                    if higher_tf_trend == "BULLISH":
                        signal = "BUY"
                        signal_reason = "Golden Cross + Bullish HTF"
                        signal_metadata['higher_tf_aligned'] = True
                        signal_metadata['signal_strength'] = 2 if volume_strong else 1
                    else:
                        signal = "BUY"
                        signal_reason = "Golden Cross (HTF neutral)"
                        signal_metadata['signal_strength'] = 1
                else:
                    if higher_tf_trend == "BEARISH":
                        signal = "SELL"
                        signal_reason = "Death Cross + Bearish HTF"
                        signal_metadata['higher_tf_aligned'] = True
                        signal_metadata['signal_strength'] = 2 if volume_strong else 1
                    else:
                        signal = "NEUTRAL"
                        signal_reason = "No HTF alignment"
        
        elif mode == "Day Trading":
            df['EMA_9'] = ta.trend.ema_indicator(df['Close'], window=9)
            df['EMA_21'] = ta.trend.ema_indicator(df['Close'], window=21)
            indicators = ['EMA_9', 'EMA_21']
            
            if len(df) >= 21:
                volume_strong = df['Volume_Ratio'].iloc[-1] > 1.3
                signal_metadata['volume_strong'] = volume_strong
                
                ema9_current = df['EMA_9'].iloc[-1]
                ema21_current = df['EMA_21'].iloc[-1]
                
                if len(df) > 1:
                    ema9_prev = df['EMA_9'].iloc[-2]
                    ema21_prev = df['EMA_21'].iloc[-2]
                    
                    # Check for bullish cross
                    if ema9_prev <= ema21_prev and ema9_current > ema21_current:
                        if higher_tf_trend in ["BULLISH", "NEUTRAL"]:
                            signal = "BUY"
                            signal_reason = "EMA9 crossed above EMA21"
                            signal_metadata['higher_tf_aligned'] = higher_tf_trend == "BULLISH"
                            signal_metadata['signal_strength'] = 2 if volume_strong and higher_tf_trend == "BULLISH" else 1
                    
                    # Check for bearish cross
                    elif ema9_prev >= ema21_prev and ema9_current < ema21_current:
                        if higher_tf_trend in ["BEARISH", "NEUTRAL"]:
                            signal = "SELL"
                            signal_reason = "EMA9 crossed below EMA21"
                            signal_metadata['higher_tf_aligned'] = higher_tf_trend == "BEARISH"
                            signal_metadata['signal_strength'] = 2 if volume_strong and higher_tf_trend == "BEARISH" else 1
        
        elif mode == "Scalping":
            df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
            indicators = ['RSI']
            
            if len(df) >= 14:
                volume_strong = df['Volume_Ratio'].iloc[-1] > 1.5
                signal_metadata['volume_strong'] = volume_strong
                current_rsi = df['RSI'].iloc[-1]
                
                if current_rsi < 30 and volume_strong:
                    signal = "BUY"
                    signal_reason = f"RSI oversold ({current_rsi:.1f})"
                    signal_metadata['signal_strength'] = 2 if higher_tf_trend != "BEARISH" else 1
                elif current_rsi > 70 and volume_strong:
                    signal = "SELL"
                    signal_reason = f"RSI overbought ({current_rsi:.1f})"
                    signal_metadata['signal_strength'] = 2 if higher_tf_trend != "BULLISH" else 1
        
        else:  # Sniper Mode
            bb = ta.volatility.BollingerBands(close=df['Close'], window=20, window_dev=2)
            df['BB_upper'] = bb.bollinger_hband()
            df['BB_middle'] = bb.bollinger_mavg()
            df['BB_lower'] = bb.bollinger_lband()
            df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
            indicators = ['BB_upper', 'BB_middle', 'BB_lower', 'RSI']
            
            if len(df) >= 20:
                volume_strong = df['Volume_Ratio'].iloc[-1] > 1.8
                signal_metadata['volume_strong'] = volume_strong
                
                price = df['Close'].iloc[-1]
                bb_lower = df['BB_lower'].iloc[-1]
                bb_upper = df['BB_upper'].iloc[-1]
                rsi = df['RSI'].iloc[-1]
                
                if price < bb_lower and rsi < 25 and volume_strong:
                    signal = "BUY"
                    signal_reason = f"Price < Lower BB, RSI={rsi:.1f}"
                    signal_metadata['signal_strength'] = 3
                elif price > bb_upper and rsi > 75 and volume_strong:
                    signal = "SELL"
                    signal_reason = f"Price > Upper BB, RSI={rsi:.1f}"
                    signal_metadata['signal_strength'] = 3
    except Exception as e:
        signal = "NEUTRAL"
        signal_reason = f"Indicator error: {str(e)[:50]}"
    
    return df, signal, signal_reason, indicators, signal_metadata

# Enhanced sentiment analysis
MARKET_KEYWORDS = {
    'bullish': 1.5, 'bearish': -1.5,
    'hawkish': -1.2, 'dovish': 1.2,
    'inflation': -1.0, 'deflation': 1.0,
    'rate hike': -1.3, 'rate cut': 1.3,
    'strong': 1.2, 'weak': -1.2,
    'growth': 1.0, 'recession': -1.5
}

@st.cache_data(ttl=300)
def get_news_sentiment(ticker_symbol: str) -> Tuple[float, list, str, float]:
    """Get news sentiment with market keyword weighting"""
    try:
        asset_name = ticker_symbol.split('=')[0] if '=' in ticker_symbol else ticker_symbol.split('-')[0]
        
        url = f"https://news.google.com/rss/search?q={asset_name}+forex&hl=en-US&gl=US&ceid=US:en"
        
        response = requests.get(url, timeout=5)
        soup = BeautifulSoup(response.content, 'xml')
        
        headlines = []
        weighted_polarities = []
        
        for item in soup.find_all('item')[:10]:
            title = item.find('title').text
            if ' - ' in title:
                title = title.split(' - ')[0]
            
            # Base polarity
            blob = TextBlob(title)
            base_polarity = blob.sentiment.polarity
            
            # Apply keyword weighting
            weighted_polarity = base_polarity
            title_lower = title.lower()
            
            for keyword, weight in MARKET_KEYWORDS.items():
                if keyword in title_lower:
                    weighted_polarity *= weight
                    break
            
            headlines.append(title)
            weighted_polarities.append(weighted_polarity)
        
        if not weighted_polarities:
            return 0.0, ["No news available"], "NEUTRAL", 0.0
        
        avg_weighted = np.mean(weighted_polarities)
        avg_base = np.mean([TextBlob(h).sentiment.polarity for h in headlines[:5]]) if headlines else 0
        
        # Determine sentiment
        if avg_weighted > 0.15:
            sentiment = "STRONGLY POSITIVE"
        elif avg_weighted > 0.05:
            sentiment = "POSITIVE"
        elif avg_weighted < -0.15:
            sentiment = "STRONGLY NEGATIVE"
        elif avg_weighted < -0.05:
            sentiment = "NEGATIVE"
        else:
            sentiment = "NEUTRAL"
        
        return avg_weighted, headlines[:5], sentiment, avg_base
        
    except Exception as e:
        return 0.0, [f"News error: {str(e)[:30]}"], "NEUTRAL", 0.0

def calculate_position_size(account_balance: float, risk_percent: float, 
                          entry_price: float, stop_loss: float) -> Dict:
    """Calculate professional position sizing"""
    risk_amount = account_balance * (risk_percent / 100)
    
    # Calculate risk per unit (simplified for demo)
    risk_per_unit = abs(entry_price - stop_loss)
    
    if risk_per_unit > 0:
        position_units = risk_amount / risk_per_unit
        position_value = position_units * entry_price
        position_percent = (position_value / account_balance) * 100
    else:
        position_units = 0
        position_value = 0
        position_percent = 0
    
    return {
        'risk_amount': risk_amount,
        'position_units': position_units,
        'position_value': position_value,
        'position_percent': min(position_percent, 10.0),  # Cap at 10%
        'risk_per_unit': risk_per_unit
    }

def calculate_risk_metrics(current_price: float, atr_value: float, 
                          atr_multiplier_sl: float, atr_multiplier_tp: float,
                          signal: str, enable_trailing: bool = False,
                          trailing_multiplier: float = 1.0) -> Dict:
    """Advanced risk management with trailing stops and breakeven"""
    if pd.isna(atr_value) or atr_value <= 0:
        atr_value = current_price * 0.01  # Default 1%
    
    # Calculate SL and TP
    if signal == "BUY":
        stop_loss = current_price - (atr_value * atr_multiplier_sl)
        take_profit = current_price + (atr_value * atr_multiplier_tp)
        breakeven = current_price + (atr_value * atr_multiplier_sl)
        trailing_stop = current_price - (atr_value * trailing_multiplier) if enable_trailing else None
    elif signal == "SELL":
        stop_loss = current_price + (atr_value * atr_multiplier_sl)
        take_profit = current_price - (atr_value * atr_multiplier_tp)
        breakeven = current_price - (atr_value * atr_multiplier_sl)
        trailing_stop = current_price + (atr_value * trailing_multiplier) if enable_trailing else None
    else:
        stop_loss = current_price - (atr_value * atr_multiplier_sl)
        take_profit = current_price + (atr_value * atr_multiplier_tp)
        breakeven = current_price
        trailing_stop = None
    
    # Calculate distances
    sl_distance = abs(current_price - stop_loss)
    tp_distance = abs(current_price - take_profit)
    rr_ratio = tp_distance / sl_distance if sl_distance > 0 else 0
    
    return {
        'stop_loss': stop_loss,
        'take_profit': take_profit,
        'breakeven_price': breakeven,
        'trailing_stop': trailing_stop,
        'sl_distance': sl_distance,
        'tp_distance': tp_distance,
        'risk_reward_ratio': rr_ratio
    }

def generate_decision(technical_signal: str, signal_strength: int, 
                     news_sentiment: str, news_polarity: float,
                     volume_strong: bool, higher_tf_aligned: bool) -> Tuple[str, str, str, float]:
    """Advanced decision engine with weighted scoring"""
    # Technical score (70%)
    tech_score = 0
    
    if technical_signal == "BUY":
        tech_score = 70
        if signal_strength == 2:
            tech_score += 10
        elif signal_strength == 3:
            tech_score += 20
    elif technical_signal == "SELL":
        tech_score = 30
        if signal_strength == 2:
            tech_score -= 10
        elif signal_strength == 3:
            tech_score -= 20
    
    # Volume bonus
    if volume_strong:
        tech_score += 10
    
    # Higher TF alignment bonus
    if higher_tf_aligned:
        tech_score += 15
    
    # Sentiment score (30%)
    sentiment_map = {
        "STRONGLY POSITIVE": 30,
        "POSITIVE": 25,
        "NEUTRAL": 15,
        "NEGATIVE": 10,
        "STRONGLY NEGATIVE": 5
    }
    
    sent_score = sentiment_map.get(news_sentiment, 15)
    
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

def create_simple_chart(data: pd.DataFrame, mode: str):
    """Create a simple chart for visualization"""
    fig = go.Figure()
    
    # Add candlestick
    if len(data) > 0:
        fig.add_trace(go.Candlestick(
            x=data.index[-50:],  # Last 50 periods
            open=data['Open'].iloc[-50:],
            high=data['High'].iloc[-50:],
            low=data['Low'].iloc[-50:],
            close=data['Close'].iloc[-50:],
            name="Price"
        ))
    
    # Add indicators based on mode
    if mode == "Swing Trading" and 'SMA_50' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index[-50:],
            y=data['SMA_50'].iloc[-50:],
            name="SMA 50",
            line=dict(color='orange', width=2)
        ))
    
    fig.update_layout(
        title=f"{selected_asset} - {selected_mode}",
        template="plotly_white",
        height=500
    )
    
    return fig

# Main app logic
def main():
    try:
        # Show loading spinner
        with st.spinner(f"📊 Loading {selected_asset} data for {selected_mode}..."):
            # Fetch primary data
            data = fetch_data(ticker, mode_config['period'], mode_config['interval'])
            
            if data is None or data.empty:
                st.error("❌ Failed to load market data. Please try again or select a different asset.")
                
                # Show last successful fetch time if available
                if st.session_state.last_success_fetch:
                    st.info(f"Last successful data fetch: {st.session_state.last_success_fetch}")
                return
            
            # Fetch higher timeframe data
            higher_tf_data = fetch_higher_timeframe_data(
                ticker, mode_config['interval'], mode_config['higher_timeframe']
            )
        
        # Calculate indicators
        data_with_indicators, technical_signal, technical_reason, indicators, signal_metadata = calculate_indicators(
            data, selected_mode, higher_tf_data
        )
        
        # Get current metrics
        current_price = data_with_indicators['Close'].iloc[-1]
        previous_price = data_with_indicators['Close'].iloc[-2] if len(data_with_indicators) > 1 else current_price
        price_change = ((current_price - previous_price) / previous_price) * 100
        
        current_atr = data_with_indicators['ATR'].iloc[-1] if 'ATR' in data_with_indicators.columns else current_price * 0.01
        
        # Get news sentiment
        with st.spinner("📰 Analyzing market sentiment..."):
            news_polarity, news_headlines, news_sentiment, base_polarity = get_news_sentiment(ticker)
        
        # Generate decision
        decision, decision_class, confidence, score = generate_decision(
            technical_signal,
            signal_metadata['signal_strength'],
            news_sentiment,
            news_polarity,
            signal_metadata['volume_strong'],
            signal_metadata['higher_tf_aligned']
        )
        
        # Calculate risk metrics
        risk_metrics = calculate_risk_metrics(
            current_price, current_atr,
            atr_multiplier_sl, atr_multiplier_tp,
            technical_signal, enable_trailing, trailing_atr_multiplier
        )
        
        # Calculate position size
        position_metrics = calculate_position_size(
            account_balance, risk_per_trade,
            current_price, risk_metrics['stop_loss']
        )
        
        # Display metrics in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(
                label="💰 Current Price",
                value=f"${current_price:.4f}" if current_price < 100 else f"${current_price:.2f}",
                delta=f"{price_change:+.2f}%"
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
                delta=f"{position_metrics['position_percent']:.1f}% of account"
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
        
        # Chart tab
        st.markdown("---")
        st.subheader("📈 Price Chart")
        
        fig = create_simple_chart(data_with_indicators, selected_mode)
        st.plotly_chart(fig, use_container_width=True)
        
        # Data and Analysis tabs
        tab1, tab2, tab3 = st.tabs(["📊 Market Data", "📰 News Analysis", "⚙️ Trade Setup"])
        
        with tab1:
            # Show recent data
            display_data = data_with_indicators[['Open', 'High', 'Low', 'Close', 'Volume']].tail(10).copy()
            
            # Add indicators if available
            for indicator in indicators:
                if indicator in data_with_indicators.columns:
                    display_data[indicator] = data_with_indicators[indicator].tail(10)
            
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
                st.info("No news headlines available for this asset.")
        
        with tab3:
            col_t1, col_t2 = st.columns(2)
            
            with col_t1:
                st.markdown("#### 📊 Technical Setup")
                st.write(f"**Signal:** {technical_signal}")
                st.write(f"**Strength:** {signal_metadata['signal_strength']}/3")
                st.write(f"**Volume Confirmation:** {'✅ Strong' if signal_metadata['volume_strong'] else '⚠️ Weak'}")
                st.write(f"**Higher TF Alignment:** {'✅ Yes' if signal_metadata['higher_tf_aligned'] else '⚠️ No'}")
                
                # ATR info
                if current_atr > 0:
                    atr_percent = (current_atr / current_price) * 100
                    st.write(f"**ATR Volatility:** {atr_percent:.2f}%")
            
            with col_t2:
                st.markdown("#### 💼 Risk Management")
                st.write(f"**Account Balance:** ${account_balance:,.2f}")
                st.write(f"**Risk Amount:** ${position_metrics['risk_amount']:.2f} ({risk_per_trade}%)")
                st.write(f"**Stop Loss:** ${risk_metrics['stop_loss']:.4f}")
                st.write(f"**Take Profit:** ${risk_metrics['take_profit']:.4f}")
                
                if risk_metrics['trailing_stop']:
                    st.write(f"**Trailing Stop:** ${risk_metrics['trailing_stop']:.4f}")
                
                st.write(f"**Risk/Reward:** {risk_metrics['risk_reward_ratio']:.2f}:1")
            
            # Trading checklist
            st.markdown("---")
            st.subheader("✅ Trading Checklist")
            
            check1, check2, check3 = st.columns(3)
            
            with check1:
                st.checkbox("Signal strength ≥ 2", value=signal_metadata['signal_strength'] >= 2)
                st.checkbox("Volume confirmation", value=signal_metadata['volume_strong'])
            
            with check2:
                st.checkbox("R/R ratio > 1.5", value=risk_metrics['risk_reward_ratio'] > 1.5)
                st.checkbox("Position < 10%", value=position_metrics['position_percent'] < 10)
            
            with check3:
                st.checkbox("HTF alignment", value=signal_metadata['higher_tf_aligned'])
                st.checkbox("Sentiment agrees", value=not (
                    (technical_signal == "BUY" and news_sentiment in ["NEGATIVE", "STRONGLY NEGATIVE"]) or
                    (technical_signal == "SELL" and news_sentiment in ["POSITIVE", "STRONGLY POSITIVE"])
                ))
        
        # Footer disclaimer
        st.markdown("---")
        st.markdown("""
        <div class="risk-warning">
        ⚠️ **RISK DISCLAIMER:** This dashboard is for educational purposes only. 
        Forex trading involves substantial risk of loss. Past performance does not guarantee future results. 
        Always trade with capital you can afford to lose and consider consulting a financial advisor.
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
        st.info("Please try refreshing the page or selecting different parameters.")

# Run the app
if __name__ == "__main__":
    main()
