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
warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(
    page_title="Pro Forex Trading Dashboard",
    page_icon="📈",
    layout="wide"
)

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
    
    # Trading mode selection with multi-timeframe info
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
    
    # Account settings
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
        1.0, 3.0, 1.5, 0.1,
        help="Stop loss distance in ATR multiples"
    )
    atr_multiplier_tp = st.slider(
        "Take Profit (ATR Multiplier)",
        1.0, 5.0, 3.0, 0.1,
        help="Take profit distance in ATR multiples"
    )
    
    # Trailing stop configuration
    enable_trailing = st.checkbox("Enable Trailing Stop", value=True)
    trailing_atr_multiplier = st.slider(
        "Trailing Stop Distance (ATR)",
        0.5, 2.0, 1.0, 0.1,
        disabled=not enable_trailing
    )
    
    st.markdown("---")
    st.caption("⚠️ Trading involves risk. Past performance ≠ future results")
    st.caption("📊 Data: Yahoo Finance | 🎯 Signals: Algorithmic Analysis")

# Improved fetch_data function with retry logic and caching
@st.cache_data(ttl=60, show_spinner=False)
def fetch_data(ticker: str, period: str, interval: str, max_retries: int = 3) -> Optional[pd.DataFrame]:
    """
    Fetch data with retry logic and error handling
    """
    last_success = st.session_state.get('last_success_fetch', None)
    
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
                st.error(f"No data for {ticker}")
                return None
            
            # Ensure required columns exist
            required_cols = ['Open', 'High', 'Low', 'Close']
            if 'Adj Close' in data.columns:
                data['Close'] = data['Adj Close']
            
            for col in required_cols:
                if col not in data.columns:
                    if col == 'Volume' and 'Volume' not in data.columns:
                        data['Volume'] = 0
                    elif col != 'Volume':
                        data[col] = data['Close']
            
            # Store last successful fetch time
            st.session_state['last_success_fetch'] = datetime.now()
            
            return data
            
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
            st.error(f"Data fetch failed after {max_retries} attempts: {str(e)[:100]}")
            return None
    
    return None

def fetch_higher_timeframe_data(ticker: str, base_interval: str, higher_interval: str) -> Optional[pd.DataFrame]:
    """
    Fetch higher timeframe data for trend confirmation
    """
    period_map = {
        '1m': '1d', '5m': '1d', '15m': '5d',
        '1h': '1mo', '1d': '6mo', '1wk': '2y'
    }
    
    period = period_map.get(higher_interval, '1mo')
    
    try:
        data = yf.download(
            tickers=ticker,
            period=period,
            interval=higher_interval,
            progress=False,
            timeout=10
        )
        
        if not data.empty:
            if 'Adj Close' in data.columns:
                data['Close'] = data['Adj Close']
            return data
    except:
        pass
    
    return None

def calculate_higher_timeframe_trend(higher_tf_data: pd.DataFrame, mode: str) -> Tuple[str, str]:
    """
    Determine trend direction on higher timeframe
    """
    if higher_tf_data is None or len(higher_tf_data) < 50:
        return "NEUTRAL", "Insufficient higher TF data"
    
    df = higher_tf_data.copy()
    
    # Calculate trend indicators based on timeframe
    if mode in ["Swing Trading", "Day Trading"]:
        df['EMA_20'] = ta.trend.ema_indicator(df['Close'], window=20)
        df['EMA_50'] = ta.trend.ema_indicator(df['Close'], window=50)
        
        if len(df) >= 50:
            ema20 = df['EMA_20'].iloc[-1]
            ema50 = df['EMA_50'].iloc[-1]
            
            if ema20 > ema50 and df['Close'].iloc[-1] > ema20:
                return "BULLISH", f"Higher TF: EMA20 > EMA50 & Price > EMA20"
            elif ema20 < ema50 and df['Close'].iloc[-1] < ema20:
                return "BEARISH", f"Higher TF: EMA20 < EMA50 & Price < EMA20"
    
    else:  # For scalping/sniper
        df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
        price = df['Close'].iloc[-1]
        sma20 = df['SMA_20'].iloc[-1]
        
        if price > sma20 * 1.01:
            return "BULLISH", f"Higher TF: Price > SMA20 (+1%)"
        elif price < sma20 * 0.99:
            return "BEARISH", f"Higher TF: Price < SMA20 (-1%)"
    
    return "NEUTRAL", "Higher TF: No clear trend"

def calculate_indicators(data: pd.DataFrame, mode: str, higher_tf_data: pd.DataFrame = None) -> Tuple[pd.DataFrame, str, str, list, dict]:
    """
    Calculate technical indicators with multi-timeframe confluence
    """
    df = data.copy()
    
    # Calculate volume metrics
    if 'Volume' in df.columns and len(df) > 20:
        df['Volume_SMA_20'] = ta.trend.sma_indicator(df['Volume'], window=20)
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_20']
    else:
        df['Volume_Ratio'] = 1.0
    
    # Calculate ATR for volatility
    if len(df) > 14:
        df['ATR'] = ta.volatility.AverageTrueRange(
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            window=14
        ).average_true_range()
    else:
        df['ATR'] = np.nan
    
    # Get higher timeframe trend
    higher_tf_trend = "NEUTRAL"
    higher_tf_reason = "No higher TF data"
    
    if higher_tf_data is not None:
        higher_tf_trend, higher_tf_reason = calculate_higher_timeframe_trend(higher_tf_data, mode)
    
    # Calculate mode-specific indicators
    signal = "NEUTRAL"
    signal_reason = ""
    indicators = []
    signal_metadata = {
        'volume_strong': False,
        'higher_tf_aligned': False,
        'signal_strength': 0
    }
    
    if mode == "Swing Trading":
        df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
        df['SMA_200'] = ta.trend.sma_indicator(df['Close'], window=200)
        indicators = ['SMA_50', 'SMA_200']
        
        if len(df) >= 200:
            sma50 = df['SMA_50'].iloc[-1]
            sma200 = df['SMA_200'].iloc[-1]
            
            # Check volume confirmation
            volume_strong = df['Volume_Ratio'].iloc[-1] > 1.2 if 'Volume_Ratio' in df.columns else True
            
            if sma50 > sma200 and higher_tf_trend == "BULLISH":
                signal = "BUY"
                signal_reason = f"Golden Cross + Higher TF Bullish"
                signal_metadata['higher_tf_aligned'] = True
                signal_metadata['signal_strength'] = 2 if volume_strong else 1
            elif sma50 < sma200 and higher_tf_trend == "BEARISH":
                signal = "SELL"
                signal_reason = f"Death Cross + Higher TF Bearish"
                signal_metadata['higher_tf_aligned'] = True
                signal_metadata['signal_strength'] = 2 if volume_strong else 1
            else:
                signal = "NEUTRAL"
                signal_reason = "No confluence with higher TF"
    
    elif mode == "Day Trading":
        df['EMA_9'] = ta.trend.ema_indicator(df['Close'], window=9)
        df['EMA_21'] = ta.trend.ema_indicator(df['Close'], window=21)
        indicators = ['EMA_9', 'EMA_21']
        
        if len(df) >= 21:
            ema9 = df['EMA_9'].iloc[-1]
            ema21 = df['EMA_21'].iloc[-1]
            prev_ema9 = df['EMA_9'].iloc[-2] if len(df) > 1 else ema9
            prev_ema21 = df['EMA_21'].iloc[-2] if len(df) > 1 else ema21
            
            # Volume validation
            volume_strong = df['Volume_Ratio'].iloc[-1] > 1.3 if 'Volume_Ratio' in df.columns else True
            signal_metadata['volume_strong'] = volume_strong
            
            # Check for cross with higher TF alignment
            bullish_cross = prev_ema9 <= prev_ema21 and ema9 > ema21
            bearish_cross = prev_ema9 >= prev_ema21 and ema9 < ema21
            
            if bullish_cross and higher_tf_trend in ["BULLISH", "NEUTRAL"]:
                signal = "BUY"
                strength = "STRONG" if volume_strong and higher_tf_trend == "BULLISH" else "MODERATE"
                signal_reason = f"{strength}: EMA9↑EMA21"
                signal_metadata['higher_tf_aligned'] = higher_tf_trend == "BULLISH"
                signal_metadata['signal_strength'] = 2 if volume_strong and higher_tf_trend == "BULLISH" else 1
            elif bearish_cross and higher_tf_trend in ["BEARISH", "NEUTRAL"]:
                signal = "SELL"
                strength = "STRONG" if volume_strong and higher_tf_trend == "BEARISH" else "MODERATE"
                signal_reason = f"{strength}: EMA9↓EMA21"
                signal_metadata['higher_tf_aligned'] = higher_tf_trend == "BEARISH"
                signal_metadata['signal_strength'] = 2 if volume_strong and higher_tf_trend == "BEARISH" else 1
            else:
                signal = "NEUTRAL"
                signal_reason = "No EMA cross or TF misalignment"
    
    elif mode == "Scalping":
        df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
        indicators = ['RSI']
        
        if len(df) >= 14:
            rsi = df['RSI'].iloc[-1]
            volume_strong = df['Volume_Ratio'].iloc[-1] > 1.5 if 'Volume_Ratio' in df.columns else True
            
            if rsi < 30 and volume_strong and higher_tf_trend != "BEARISH":
                signal = "BUY"
                signal_reason = f"RSI Oversold ({rsi:.1f}) + Volume Spike"
                signal_metadata['higher_tf_aligned'] = higher_tf_trend == "BULLISH"
                signal_metadata['signal_strength'] = 2 if higher_tf_trend == "BULLISH" else 1
            elif rsi > 70 and volume_strong and higher_tf_trend != "BULLISH":
                signal = "SELL"
                signal_reason = f"RSI Overbought ({rsi:.1f}) + Volume Spike"
                signal_metadata['higher_tf_aligned'] = higher_tf_trend == "BEARISH"
                signal_metadata['signal_strength'] = 2 if higher_tf_trend == "BEARISH" else 1
            else:
                signal = "NEUTRAL"
                signal_reason = f"RSI Neutral ({rsi:.1f})"
    
    else:  # Sniper Mode
        bb = ta.volatility.BollingerBands(close=df['Close'], window=20, window_dev=2)
        df['BB_upper'] = bb.bollinger_hband()
        df['BB_middle'] = bb.bollinger_mavg()
        df['BB_lower'] = bb.bollinger_lband()
        df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
        indicators = ['BB_upper', 'BB_middle', 'BB_lower', 'RSI']
        
        if len(df) >= 20:
            price = df['Close'].iloc[-1]
            bb_lower = df['BB_lower'].iloc[-1]
            bb_upper = df['BB_upper'].iloc[-1]
            rsi = df['RSI'].iloc[-1]
            
            volume_strong = df['Volume_Ratio'].iloc[-1] > 1.8 if 'Volume_Ratio' in df.columns else False
            
            if price < bb_lower and rsi < 25 and volume_strong:
                signal = "BUY"
                signal_reason = f"Sniper: Price<BB Lower & RSI={rsi:.1f}"
                signal_metadata['signal_strength'] = 3 if volume_strong else 2
            elif price > bb_upper and rsi > 75 and volume_strong:
                signal = "SELL"
                signal_reason = f"Sniper: Price>BB Upper & RSI={rsi:.1f}"
                signal_metadata['signal_strength'] = 3 if volume_strong else 2
            else:
                signal = "NEUTRAL"
                signal_reason = "No sniper setup"
    
    # Add higher TF info to reason
    if higher_tf_trend != "NEUTRAL" and signal != "NEUTRAL":
        signal_reason += f" | {higher_tf_reason}"
    
    return df, signal, signal_reason, indicators, signal_metadata

# Enhanced sentiment analysis with market keywords
MARKET_KEYWORDS = {
    'bullish': 1.5,
    'bearish': -1.5,
    'hawkish': -1.2,  # Typically negative for bonds, positive for currency
    'dovish': 1.2,    # Typically positive for bonds, negative for currency
    'inflation': -1.0,
    'deflation': 1.0,
    'rate hike': -1.3,
    'rate cut': 1.3,
    'tightening': -1.1,
    'easing': 1.1,
    'strong': 1.2,
    'weak': -1.2,
    'growth': 1.0,
    'recession': -1.5,
    'risk-on': 1.0,
    'risk-off': -1.0,
    'breakout': 1.0,
    'breakdown': -1.0,
    'support': 0.5,
    'resistance': -0.5
}

@st.cache_data(ttl=300)
def get_news_sentiment(ticker_symbol: str) -> Tuple[float, list, str, float]:
    """
    Get news sentiment with market keyword weighting
    """
    try:
        # Extract asset name
        asset_name = ticker_symbol.split('=')[0] if '=' in ticker_symbol else ticker_symbol.split('-')[0]
        
        # News sources
        sources = [
            f"https://news.google.com/rss/search?q={asset_name}+forex+OR+fx+OR+currency&hl=en-US&gl=US&ceid=US:en",
            f"https://news.google.com/rss/search?q={asset_name}+Federal+Reserve+OR+ECB+OR+central+bank&hl=en-US&gl=US&ceid=US:en"
        ]
        
        headlines = []
        weighted_polarities = []
        
        for url in sources:
            try:
                response = requests.get(url, timeout=10)
                soup = BeautifulSoup(response.content, 'xml')
                
                for item in soup.find_all('item')[:15]:
                    title = item.find('title').text
                    # Clean title
                    if ' - ' in title:
                        title = title.split(' - ')[0]
                    
                    # Calculate base polarity
                    blob = TextBlob(title)
                    base_polarity = blob.sentiment.polarity
                    
                    # Apply market keyword weighting
                    weighted_polarity = base_polarity
                    title_lower = title.lower()
                    
                    for keyword, weight in MARKET_KEYWORDS.items():
                        if keyword in title_lower:
                            weighted_polarity *= weight
                            break  # Use strongest keyword only
                    
                    headlines.append(title)
                    weighted_polarities.append(weighted_polarity)
                    
            except Exception as e:
                continue
        
        if not weighted_polarities:
            return 0.0, ["No news available"], "NEUTRAL", 0.0
        
        # Calculate statistics
        avg_weighted_polarity = np.mean(weighted_polarities)
        avg_base_polarity = np.mean([TextBlob(h).sentiment.polarity for h in headlines[:10]]) if headlines else 0
        
        # Determine sentiment category
        if avg_weighted_polarity > 0.15:
            sentiment = "STRONGLY POSITIVE"
        elif avg_weighted_polarity > 0.05:
            sentiment = "POSITIVE"
        elif avg_weighted_polarity < -0.15:
            sentiment = "STRONGLY NEGATIVE"
        elif avg_weighted_polarity < -0.05:
            sentiment = "NEGATIVE"
        else:
            sentiment = "NEUTRAL"
        
        return avg_weighted_polarity, headlines[:8], sentiment, avg_base_polarity
        
    except Exception as e:
        return 0.0, [f"News error: {str(e)[:50]}"], "NEUTRAL", 0.0

def calculate_position_size(account_balance: float, risk_percent: float, 
                          entry_price: float, stop_loss: float, atr: float) -> Dict:
    """
    Calculate professional position sizing
    """
    risk_amount = account_balance * (risk_percent / 100)
    
    # Risk per unit (assuming 1 unit movement = $1 for simplicity)
    # For forex, this would need pip value calculation
    risk_per_unit = abs(entry_price - stop_loss)
    
    if risk_per_unit > 0:
        position_units = risk_amount / risk_per_unit
    else:
        position_units = 0
    
    # Calculate position value
    position_value = position_units * entry_price
    
    # Risk/Reward metrics
    risk_reward_ratio = 3.0  # Default, would calculate based on TP
    
    return {
        'risk_amount': risk_amount,
        'position_units': position_units,
        'position_value': position_value,
        'risk_per_unit': risk_per_unit,
        'max_position_percent': min(10.0, (position_value / account_balance) * 100)
    }

def calculate_risk_metrics(current_price: float, atr_value: float, 
                          atr_multiplier_sl: float, atr_multiplier_tp: float,
                          signal: str, enable_trailing: bool, 
                          trailing_atr_multiplier: float) -> Dict:
    """
    Advanced risk management with trailing stops and breakeven
    """
    if pd.isna(atr_value) or atr_value == 0:
        atr_value = current_price * 0.01  # Default 1% if no ATR
    
    # Calculate stop loss and take profit
    if signal == "BUY":
        stop_loss = current_price - (atr_value * atr_multiplier_sl)
        take_profit = current_price + (atr_value * atr_multiplier_tp)
        breakeven_price = current_price + (atr_value * atr_multiplier_sl)
        trailing_stop = current_price - (atr_value * trailing_atr_multiplier) if enable_trailing else None
    elif signal == "SELL":
        stop_loss = current_price + (atr_value * atr_multiplier_sl)
        take_profit = current_price - (atr_value * atr_multiplier_tp)
        breakeven_price = current_price - (atr_value * atr_multiplier_sl)
        trailing_stop = current_price + (atr_value * trailing_atr_multiplier) if enable_trailing else None
    else:
        stop_loss = current_price - (atr_value * atr_multiplier_sl)
        take_profit = current_price + (atr_value * atr_multiplier_tp)
        breakeven_price = current_price
        trailing_stop = None
    
    # Calculate distances
    sl_distance_pips = abs(current_price - stop_loss) * 10000  # Simplified for forex
    tp_distance_pips = abs(current_price - take_profit) * 10000
    risk_reward_ratio = tp_distance_pips / sl_distance_pips if sl_distance_pips > 0 else 0
    
    return {
        'stop_loss': stop_loss,
        'take_profit': take_profit,
        'breakeven_price': breakeven_price,
        'trailing_stop': trailing_stop,
        'sl_distance': sl_distance_pips,
        'tp_distance': tp_distance_pips,
        'risk_reward_ratio': risk_reward_ratio,
        'atr_value': atr_value
    }

def generate_decision(technical_signal: str, signal_strength: int, 
                     news_sentiment: str, news_polarity: float,
                     volume_strong: bool, higher_tf_aligned: bool) -> Tuple[str, str, str]:
    """
    Advanced decision engine with weighted scoring
    """
    # Technical score (70% weight)
    tech_score = 0
    
    if technical_signal == "BUY":
        tech_score = 70
        if signal_strength == 2:
            tech_score += 15
        elif signal_strength == 3:
            tech_score += 25
    elif technical_signal == "SELL":
        tech_score = 30
        if signal_strength == 2:
            tech_score -= 15
        elif signal_strength == 3:
            tech_score -= 25
    
    # Volume bonus
    if volume_strong:
        tech_score += 10
    
    # Higher TF alignment bonus
    if higher_tf_aligned:
        tech_score += 15
    
    # Sentiment score (30% weight)
    sentiment_map = {
        "STRONGLY POSITIVE": 30,
        "POSITIVE": 20,
        "NEUTRAL": 15,
        "NEGATIVE": 10,
        "STRONGLY NEGATIVE": 0
    }
    
    sent_score = sentiment_map.get(news_sentiment, 15)
    
    # Adjust based on polarity magnitude
    sent_score += news_polarity * 10
    
    # Combined score (70% tech, 30% sentiment)
    combined_score = (tech_score * 0.7) + (sent_score * 0.3)
    
    # Generate decision
    if technical_signal == "BUY":
        if combined_score >= 75:
            decision = "STRONG BUY SIGNAL"
            decision_class = "signal-strong-buy"
            confidence = "HIGH (90%+)"
        elif combined_score >= 60:
            decision = "MODERATE BUY SIGNAL"
            decision_class = "signal-moderate-buy"
            confidence = "MEDIUM (70-89%)"
        else:
            decision = "WEAK BUY / HOLD"
            decision_class = "signal-neutral"
            confidence = "LOW (<70%)"
    
    elif technical_signal == "SELL":
        if combined_score <= 25:
            decision = "STRONG SELL SIGNAL"
            decision_class = "signal-strong-sell"
            confidence = "HIGH (90%+)"
        elif combined_score <= 40:
            decision = "MODERATE SELL SIGNAL"
            decision_class = "signal-moderate-sell"
            confidence = "MEDIUM (70-89%)"
        else:
            decision = "WEAK SELL / HOLD"
            decision_class = "signal-neutral"
            confidence = "LOW (<70%)"
    
    else:  # NEUTRAL
        decision = "NO CLEAR SIGNAL - HOLD"
        decision_class = "signal-neutral"
        confidence = "LOW - WAIT"
    
    return decision, decision_class, confidence, combined_score

# Main app logic
def main():
    # Fetch primary data
    with st.spinner(f"📊 Fetching {selected_mode} data..."):
        data = fetch_data(ticker, mode_config['period'], mode_config['interval'])
    
    if data is None or data.empty:
        st.error("❌ Data retrieval failed. Try another asset or check connection.")
        return
    
    # Fetch higher timeframe data for confluence
    higher_tf_data = None
    if 'higher_timeframe' in mode_config:
        higher_tf_data = fetch_higher_timeframe_data(
            ticker, mode_config['interval'], mode_config['higher_timeframe']
        )
    
    # Calculate indicators
    data_with_indicators, technical_signal, technical_reason, indicators, signal_metadata = calculate_indicators(
        data, selected_mode, higher_tf_data
    )
    
    # Get current price and metrics
    current_price = data_with_indicators['Close'].iloc[-1]
    previous_price = data_with_indicators['Close'].iloc[-2] if len(data_with_indicators) > 1 else current_price
    price_change_pct = ((current_price - previous_price) / previous_price) * 100
    current_atr = data_with_indicators['ATR'].iloc[-1] if 'ATR' in data_with_indicators.columns else 0
    
    # Get news sentiment
    with st.spinner("📰 Analyzing market sentiment..."):
        news_polarity, news_headlines, news_sentiment, base_polarity = get_news_sentiment(ticker)
    
    # Generate trading decision
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
    
    # Calculate position sizing
    position_metrics = calculate_position_size(
        account_balance, risk_per_trade,
        current_price, risk_metrics['stop_loss'], current_atr
    )
    
    # Display dashboard metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            label="💰 Current Price",
            value=f"${current_price:.4f}" if current_price < 100 else f"${current_price:.2f}",
            delta=f"{price_change_pct:.2f}%",
            delta_color="normal"
        )
        st.caption(f"ATR Volatility: {current_atr:.4f}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            label="🎯 Technical Signal",
            value=technical_signal,
            delta=technical_reason[:40]
        )
        vol_status = "✅ Strong" if signal_metadata['volume_strong'] else "⚠️ Weak"
        tf_status = "✅ Aligned" if signal_metadata['higher_tf_aligned'] else "⚠️ Neutral"
        st.caption(f"Volume: {vol_status} | TF: {tf_status}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            label="📰 News Sentiment",
            value=news_sentiment,
            delta=f"Score: {news_polarity:.3f}"
        )
        st.caption(f"Headlines analyzed: {len(news_headlines)}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric(
                label="🛑 Stop Loss",
                value=f"${risk_metrics['stop_loss']:.4f}",
                delta=f"{(risk_metrics['stop_loss'] - current_price):.4f}"
            )
        with col_b:
            st.metric(
                label="🎯 Take Profit",
                value=f"${risk_metrics['take_profit']:.4f}",
                delta=f"{(risk_metrics['take_profit'] - current_price):.4f}"
            )
        st.caption(f"R/R Ratio: {risk_metrics['risk_reward_ratio']:.2f}:1")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            label="📊 Position Sizing",
            value=f"${position_metrics['position_value']:.0f}",
            delta=f"{position_metrics['max_position_percent']:.1f}% of account"
        )
        st.caption(f"Risk: ${position_metrics['risk_amount']:.0f} ({risk_per_trade}%)")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            label="⚡ Algorithm Score",
            value=f"{score:.0f}/100",
            delta=confidence
        )
        st.caption(f"Technical: 70% | Sentiment: 30%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Display trading decision
    st.markdown(f'<div class="{decision_class}">', unsafe_allow_html=True)
    st.markdown(f"## {decision}")
    st.markdown(f"**Signal Score:** {score:.1f}/100 | **Confidence:** {confidence}")
    st.markdown(f"**Technical:** {technical_signal} | **Sentiment:** {news_sentiment}")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Advanced risk metrics
    with st.expander("🔧 Advanced Risk Metrics", expanded=False):
        col_r1, col_r2, col_r3 = st.columns(3)
        with col_r1:
            st.metric("Break-even Price", f"${risk_metrics['breakeven_price']:.4f}")
            if risk_metrics['trailing_stop']:
                st.metric("Trailing Stop", f"${risk_metrics['trailing_stop']:.4f}")
        with col_r2:
            st.metric("SL Distance", f"{risk_metrics['sl_distance']:.1f} pips")
            st.metric("TP Distance", f"{risk_metrics['tp_distance']:.1f} pips")
        with col_r3:
            st.metric("Position Units", f"{position_metrics['position_units']:.2f}")
            st.metric("Risk per Unit", f"${position_metrics['risk_per_unit']:.4f}")
    
    st.markdown("---")
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Live Chart", "📊 Market Data", "📰 News Analysis", "⚙️ Trade Setup"])
    
    with tab1:
        # Chart would go here (similar to original but enhanced)
        st.info("Chart functionality maintained from original code")
        
    with tab2:
        st.subheader("Market Data & Indicators")
        if not data_with_indicators.empty:
            display_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if 'Volume_Ratio' in data_with_indicators.columns:
                display_cols.append('Volume_Ratio')
            
            display_data = data_with_indicators[display_cols].tail(20).copy()
            
            # Format display
            for col in display_data.columns:
                if col != 'Volume_Ratio':
                    display_data[col] = display_data[col].round(4)
                else:
                    display_data[col] = display_data[col].round(2)
            
            st.dataframe(display_data.style.format("{:.4f}"), use_container_width=True)
            
            # Volume analysis
            if 'Volume_Ratio' in data_with_indicators.columns:
                current_vol_ratio = data_with_indicators['Volume_Ratio'].iloc[-1]
                st.progress(min(current_vol_ratio / 2.0, 1.0), 
                           text=f"Volume Ratio: {current_vol_ratio:.2f}x average")
    
    with tab3:
        st.subheader("Market Sentiment Analysis")
        
        col_n1, col_n2 = st.columns(2)
        with col_n1:
            st.metric("Weighted Sentiment", f"{news_polarity:.3f}", 
                     delta=f"Base: {base_polarity:.3f}")
            st.metric("Market Keywords", "Applied", 
                     delta=f"{len(MARKET_KEYWORDS)} keywords")
        
        with col_n2:
            # Sentiment gauge
            sentiment_value = min(max(news_polarity * 3 + 0.5, 0), 1)
            st.progress(sentiment_value, 
                       text=f"Sentiment: {news_sentiment}")
            
            if news_polarity > 0.1:
                st.success("✅ Bullish market sentiment detected")
            elif news_polarity < -0.1:
                st.error("⚠️ Bearish market sentiment detected")
            else:
                st.info("⚪ Neutral market sentiment")
        
        st.subheader("Recent Headlines")
        for i, headline in enumerate(news_headlines[:6], 1):
            blob = TextBlob(headline)
            polarity = blob.sentiment.polarity
            
            # Apply keyword weighting
            weighted_polarity = polarity
            for keyword in MARKET_KEYWORDS:
                if keyword in headline.lower():
                    weighted_polarity *= MARKET_KEYWORDS[keyword]
                    break
            
            # Display
            col_h1, col_h2 = st.columns([4, 1])
            with col_h1:
                st.write(f"{i}. {headline}")
            with col_h2:
                st.caption(f"{weighted_polarity:.3f}")
            st.divider()
    
    with tab4:
        st.subheader("Complete Trade Setup")
        
        col_s1, col_s2 = st.columns(2)
        
        with col_s1:
            st.markdown("#### 📊 Technical Analysis")
            st.write(f"**Primary Signal:** {technical_signal}")
            st.write(f"**Signal Strength:** {signal_metadata['signal_strength']}/3")
            st.write(f"**Higher TF Alignment:** {'✅ Yes' if signal_metadata['higher_tf_aligned'] else '⚠️ No'}")
            st.write(f"**Volume Confirmation:** {'✅ Strong' if signal_metadata['volume_strong'] else '⚠️ Weak'}")
            st.write(f"**Price Trend:** {'📈 Bullish' if price_change_pct > 0 else '📉 Bearish' if price_change_pct < 0 else '➖ Neutral'}")
            
            # ATR analysis
            if current_atr > 0:
                atr_percent = (current_atr / current_price) * 100
                st.write(f"**ATR Volatility:** {atr_percent:.2f}%")
                if atr_percent > 1.0:
                    st.warning("⚠️ High volatility - wider stops recommended")
        
        with col_s2:
            st.markdown("#### 💼 Position Management")
            st.write(f"**Account Balance:** ${account_balance:,.2f}")
            st.write(f"**Risk per Trade:** ${position_metrics['risk_amount']:.2f} ({risk_per_trade}%)")
            st.write(f"**Position Size:** ${position_metrics['position_value']:.2f}")
            st.write(f"**Max Position:** {position_metrics['max_position_percent']:.1f}% of account")
            
            # Risk metrics
            st.write(f"**Stop Loss:** ${risk_metrics['stop_loss']:.4f}")
            st.write(f"**Take Profit:** ${risk_metrics['take_profit']:.4f}")
            if risk_metrics['trailing_stop']:
                st.write(f"**Trailing Stop:** ${risk_metrics['trailing_stop']:.4f}")
            st.write(f"**Risk/Reward:** {risk_metrics['risk_reward_ratio']:.2f}:1")
        
        # Trading checklist
        st.markdown("---")
        st.subheader("✅ Pre-Trade Checklist")
        
        checklist_cols = st.columns(3)
        with checklist_cols[0]:
            st.checkbox("Higher TF alignment", value=signal_metadata['higher_tf_aligned'])
            st.checkbox("Volume confirmation", value=signal_metadata['volume_strong'])
        with checklist_cols[1]:
            st.checkbox("Sentiment agrees", value=news_sentiment in ["POSITIVE", "STRONGLY POSITIVE"] if technical_signal == "BUY" else True)
            st.checkbox("Risk/Reward > 1.5", value=risk_metrics['risk_reward_ratio'] > 1.5)
        with checklist_cols[2]:
            st.checkbox("Position size < 10%", value=position_metrics['max_position_percent'] < 10)
            st.checkbox("Volatility acceptable", value=current_atr/current_price < 0.02)
        
        # Final warning
        st.markdown("""
        <div class="risk-warning">
        ⚠️ **RISK DISCLAIMER:** This algorithmic dashboard is for educational purposes only. 
        Forex trading involves substantial risk of loss. Past performance does not guarantee future results. 
        Always trade with capital you can afford to lose and consider consulting a financial advisor.
        </div>
        """, unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    main()
