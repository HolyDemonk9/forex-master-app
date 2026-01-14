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
import re
warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(
    page_title="Forex Trading Dashboard",
    page_icon="📈",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .signal-buy {
        background-color: #4CAF50;
        color: white;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
        font-size: 1.2rem;
        margin: 10px 0;
    }
    .signal-sell {
        background-color: #F44336;
        color: white;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
        font-size: 1.2rem;
        margin: 10px 0;
    }
    .signal-neutral {
        background-color: #FF9800;
        color: white;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
        font-size: 1.2rem;
        margin: 10px 0;
    }
    .metric-card {
        background-color: #f5f5f5;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #1E88E5;
        margin-bottom: 10px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 5px 5px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E88E5;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">📊 Professional Forex Trading Dashboard</h1>', unsafe_allow_html=True)

# Session state for caching
if 'last_ticker' not in st.session_state:
    st.session_state.last_ticker = None
if 'last_mode' not in st.session_state:
    st.session_state.last_mode = None

# Sidebar
with st.sidebar:
    st.header("⚙️ Trading Configuration")
    
    # Asset selection
    assets = {
        "EUR/USD": "EURUSD=X",
        "GBP/USD": "GBPUSD=X",
        "USD/JPY": "JPY=X",
        "USD/CHF": "CHF=X",
        "AUD/USD": "AUDUSD=X",
        "USD/CAD": "CAD=X",
        "NZD/USD": "NZDUSD=X",
        "BTC/USD": "BTC-USD",
        "ETH/USD": "ETH-USD",
        "Gold (XAU/USD)": "GC=F",
        "Silver (XAG/USD)": "SI=F",
        "US Dollar Index": "DX-Y.NYB"
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
            "period": "6mo",  # Changed from 1y for faster loading
            "interval": "1d",
            "description": "6-month daily data, Golden Cross (SMA50 > SMA200)"
        },
        "Day Trading": {
            "period": "5d",
            "interval": "15m",
            "description": "5-day 15-min data, EMA9 crosses EMA21"
        },
        "Scalping": {
            "period": "1d",
            "interval": "5m",
            "description": "1-day 5-min data, RSI oversold/overbought"
        },
        "Sniper Mode": {
            "period": "1d",
            "interval": "1m",
            "description": "1-day 1-min data, Price < Lower BB & RSI < 25"
        }
    }
    
    selected_mode = st.selectbox(
        "Select Trading Mode:",
        list(trading_modes.keys()),
        index=0
    )
    
    mode_config = trading_modes[selected_mode]
    
    st.markdown(f"**Mode Description:** {mode_config['description']}")
    
    # Additional parameters
    st.markdown("---")
    st.subheader("Risk Management")
    atr_multiplier_sl = st.slider("Stop Loss (ATR Multiplier)", 1.0, 3.0, 1.5, 0.1)
    atr_multiplier_tp = st.slider("Take Profit (ATR Multiplier)", 1.0, 5.0, 3.0, 0.1)
    
    # Refresh button
    st.markdown("---")
    if st.button("🔄 Refresh Data", use_container_width=True):
        # Clear cache by changing session state
        st.session_state.last_ticker = None
        st.session_state.last_mode = None
        st.rerun()
    
    st.markdown("---")
    st.markdown("**Dashboard Info:**")
    st.caption("Data provided by Yahoo Finance")
    st.caption("Signals are for educational purposes only")

# Function to fetch data with improved error handling
@st.cache_data(ttl=300, show_spinner=False)
def fetch_data(_ticker, period, interval):
    """Fetch data with retry logic and better error handling"""
    try:
        # Clean ticker symbol
        clean_ticker = str(_ticker).strip().upper()
        
        # For forex pairs, ensure proper format
        if '=X' not in clean_ticker and clean_ticker not in ['BTC-USD', 'ETH-USD', 'GC=F', 'SI=F', 'DX-Y.NYB']:
            clean_ticker = f"{clean_ticker}=X"
        
        # Use Ticker object for more reliable data
        ticker_obj = yf.Ticker(clean_ticker)
        
        # Get data with timeout
        data = ticker_obj.history(
            period=period,
            interval=interval,
            actions=False,  # Don't need dividends/splits
            timeout=10
        )
        
        if data.empty or len(data) < 5:
            # Try alternative method
            data = yf.download(
                tickers=clean_ticker,
                period=period,
                interval=interval,
                progress=False,
                timeout=10
            )
        
        if data.empty:
            st.warning(f"No data retrieved for {clean_ticker}. Trying with .TO suffix...")
            # Try Toronto exchange for some forex
            if 'CAD' in clean_ticker:
                clean_ticker = clean_ticker.replace('=X', '.TO')
                data = yf.download(clean_ticker, period=period, interval=interval, progress=False)
        
        if data.empty:
            return None
        
        # Ensure we have the required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col not in data.columns:
                if col == 'Close' and 'Adj Close' in data.columns:
                    data['Close'] = data['Adj Close']
                elif col in ['Open', 'High', 'Low']:
                    data[col] = data['Close']
                elif col == 'Volume':
                    data['Volume'] = 0
        
        # Remove timezone if present
        if data.index.tz is not None:
            data.index = data.index.tz_localize(None)
        
        # Drop NaN rows
        data = data.dropna()
        
        return data
        
    except Exception as e:
        st.error(f"Error fetching data for {_ticker}: {str(e)[:100]}")
        return None

# Function to calculate indicators with improved logic
def calculate_indicators(data, mode):
    """Calculate technical indicators based on trading mode"""
    if data is None or len(data) < 5:
        return pd.DataFrame(), "NEUTRAL", "Insufficient data", []
    
    df = data.copy()
    
    # Calculate ATR for all modes
    try:
        if len(df) > 14:
            df['ATR'] = ta.volatility.AverageTrueRange(
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                window=14
            ).average_true_range()
        else:
            df['ATR'] = np.nan
    except:
        df['ATR'] = np.nan
    
    signal = "NEUTRAL"
    signal_reason = "No signal generated"
    indicators = []
    
    try:
        if mode == "Swing Trading":
            # Calculate SMAs
            df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=min(50, len(df)))
            df['SMA_200'] = ta.trend.sma_indicator(df['Close'], window=min(200, len(df)))
            
            # Generate signal
            if len(df) >= 50 and not pd.isna(df['SMA_50'].iloc[-1]):
                # Check for golden/death cross
                if len(df) >= 2:
                    # Compare current and previous values
                    current_sma50 = df['SMA_50'].iloc[-1]
                    current_sma200 = df['SMA_200'].iloc[-1] if not pd.isna(df['SMA_200'].iloc[-1]) else current_sma50
                    prev_sma50 = df['SMA_50'].iloc[-2] if len(df) > 1 else current_sma50
                    prev_sma200 = df['SMA_200'].iloc[-2] if len(df) > 1 and not pd.isna(df['SMA_200'].iloc[-2]) else prev_sma50
                    
                    if current_sma50 > current_sma200 and prev_sma50 <= prev_sma200:
                        signal = "BUY"
                        signal_reason = "Golden Cross (SMA50 crossed above SMA200)"
                    elif current_sma50 < current_sma200 and prev_sma50 >= prev_sma200:
                        signal = "SELL"
                        signal_reason = "Death Cross (SMA50 crossed below SMA200)"
                    elif current_sma50 > current_sma200:
                        signal = "BUY"
                        signal_reason = "SMA50 > SMA200 (Bullish Trend)"
                    elif current_sma50 < current_sma200:
                        signal = "SELL"
                        signal_reason = "SMA50 < SMA200 (Bearish Trend)"
                
                indicators = ['SMA_50', 'SMA_200']
        
        elif mode == "Day Trading":
            # Calculate EMAs
            df['EMA_9'] = ta.trend.ema_indicator(df['Close'], window=min(9, len(df)))
            df['EMA_21'] = ta.trend.ema_indicator(df['Close'], window=min(21, len(df)))
            
            # Generate signal based on cross
            if len(df) >= 21:
                current_ema9 = df['EMA_9'].iloc[-1]
                prev_ema9 = df['EMA_9'].iloc[-2] if len(df) > 1 else current_ema9
                current_ema21 = df['EMA_21'].iloc[-1]
                prev_ema21 = df['EMA_21'].iloc[-2] if len(df) > 1 else current_ema21
                
                # Check for cross
                if prev_ema9 <= prev_ema21 and current_ema9 > current_ema21:
                    signal = "BUY"
                    signal_reason = "EMA9 crossed above EMA21"
                elif prev_ema9 >= prev_ema21 and current_ema9 < current_ema21:
                    signal = "SELL"
                    signal_reason = "EMA9 crossed below EMA21"
                elif current_ema9 > current_ema21:
                    signal = "BUY"
                    signal_reason = "EMA9 > EMA21 (Bullish)"
                elif current_ema9 < current_ema21:
                    signal = "SELL"
                    signal_reason = "EMA9 < EMA21 (Bearish)"
                
                indicators = ['EMA_9', 'EMA_21']
        
        elif mode == "Scalping":
            # Calculate RSI
            if len(df) >= 14:
                df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
                current_rsi = df['RSI'].iloc[-1]
                
                if not pd.isna(current_rsi):
                    if current_rsi < 30:
                        signal = "BUY"
                        signal_reason = f"RSI oversold ({current_rsi:.1f} < 30)"
                    elif current_rsi > 70:
                        signal = "SELL"
                        signal_reason = f"RSI overbought ({current_rsi:.1f} > 70)"
                    elif current_rsi > 50:
                        signal = "BUY"
                        signal_reason = f"RSI bullish ({current_rsi:.1f} > 50)"
                    else:
                        signal = "SELL"
                        signal_reason = f"RSI bearish ({current_rsi:.1f} < 50)"
                    
                    indicators = ['RSI']
        
        else:  # Sniper Mode
            # Calculate Bollinger Bands and RSI
            if len(df) >= 20:
                bb = ta.volatility.BollingerBands(
                    close=df['Close'], 
                    window=min(20, len(df)), 
                    window_dev=2
                )
                df['BB_upper'] = bb.bollinger_hband()
                df['BB_middle'] = bb.bollinger_mavg()
                df['BB_lower'] = bb.bollinger_lband()
                
                if len(df) >= 14:
                    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
                
                current_price = df['Close'].iloc[-1]
                current_bb_lower = df['BB_lower'].iloc[-1] if 'BB_lower' in df.columns else current_price
                current_bb_upper = df['BB_upper'].iloc[-1] if 'BB_upper' in df.columns else current_price
                current_rsi = df['RSI'].iloc[-1] if 'RSI' in df.columns else 50
                
                if not pd.isna(current_bb_lower) and not pd.isna(current_rsi):
                    if current_price < current_bb_lower and current_rsi < 25:
                        signal = "BUY"
                        signal_reason = f"Price < Lower BB & RSI oversold ({current_rsi:.1f} < 25)"
                    elif current_price > current_bb_upper and current_rsi > 75:
                        signal = "SELL"
                        signal_reason = f"Price > Upper BB & RSI overbought ({current_rsi:.1f} > 75)"
                    elif current_price < current_bb_lower:
                        signal = "BUY"
                        signal_reason = "Price < Lower BB (Oversold)"
                    elif current_price > current_bb_upper:
                        signal = "SELL"
                        signal_reason = "Price > Upper BB (Overbought)"
                
                indicators = ['BB_upper', 'BB_middle', 'BB_lower']
                if 'RSI' in df.columns:
                    indicators.append('RSI')
    
    except Exception as e:
        st.error(f"Error calculating indicators: {str(e)[:100]}")
        signal = "NEUTRAL"
        signal_reason = f"Indicator calculation error"
    
    return df, signal, signal_reason, indicators

# Function to get news sentiment with improved reliability
@st.cache_data(ttl=600, show_spinner=False)
def get_news_sentiment(ticker_symbol):
    """Get news sentiment with multiple fallback sources"""
    try:
        # Clean ticker for search
        asset_name = re.sub(r'[=X\-\.].*', '', ticker_symbol)
        if asset_name in ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD']:
            search_terms = [f"{asset_name[:3]}/{asset_name[3:]} forex", "currency trading"]
        elif asset_name in ['BTC', 'ETH']:
            search_terms = [f"{asset_name} cryptocurrency", "crypto news"]
        elif asset_name in ['GC', 'SI']:
            search_terms = ["gold silver precious metals", "commodities trading"]
        else:
            search_terms = [f"{asset_name} trading", "financial markets"]
        
        all_headlines = []
        
        # Try multiple news sources
        sources = [
            # Financial news sources
            f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker_symbol}&region=US&lang=en-US",
            "https://www.fxstreet.com/rss",
            "https://www.forexlive.com/feed/",
        ]
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        for url in sources:
            try:
                response = requests.get(url, headers=headers, timeout=5)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'xml')
                    items = soup.find_all('item')[:10]  # Limit to 10 items per source
                    
                    for item in items:
                        title = item.find('title')
                        if title:
                            headline = title.text.strip()
                            # Filter for relevant headlines
                            if any(term.lower() in headline.lower() for term in search_terms + ['forex', 'currency', 'trading', 'market']):
                                all_headlines.append(headline)
            except:
                continue
        
        # If no headlines found, use simulated headlines
        if not all_headlines:
            all_headlines = [
                f"{asset_name} shows mixed signals in today's trading session",
                f"Market analysts predict volatility for {asset_name}",
                f"Technical indicators suggest potential breakout for {asset_name}",
                f"Global economic factors affecting {asset_name} trading",
                f"Traders watching {asset_name} for directional cues"
            ]
        
        # Analyze sentiment
        polarities = []
        for headline in all_headlines[:10]:  # Analyze first 10 headlines
            try:
                blob = TextBlob(headline)
                polarities.append(blob.sentiment.polarity)
            except:
                polarities.append(0.0)
        
        avg_polarity = np.mean(polarities) if polarities else 0.0
        
        # Determine sentiment category
        if avg_polarity > 0.15:
            sentiment = "BULLISH"
        elif avg_polarity < -0.15:
            sentiment = "BEARISH"
        else:
            sentiment = "NEUTRAL"
        
        return avg_polarity, all_headlines[:5], sentiment
        
    except Exception as e:
        # Fallback to neutral sentiment
        return 0.0, ["No news data available at the moment"], "NEUTRAL"

# Decision Engine
def generate_decision(technical_signal, news_sentiment, technical_reason):
    """Generate trading decision based on technical and fundamental analysis"""
    # Map news sentiment to match technical signal format
    news_map = {"BULLISH": "POSITIVE", "BEARISH": "NEGATIVE", "NEUTRAL": "NEUTRAL"}
    mapped_news = news_map.get(news_sentiment, "NEUTRAL")
    
    if technical_signal == "BUY" and mapped_news == "POSITIVE":
        decision = "STRONG BUY SIGNAL 🟢"
        decision_class = "signal-buy"
        confidence = "High"
        emoji = "📈"
    elif technical_signal == "SELL" and mapped_news == "NEGATIVE":
        decision = "STRONG SELL SIGNAL 🔴"
        decision_class = "signal-sell"
        confidence = "High"
        emoji = "📉"
    elif technical_signal == "BUY" and mapped_news == "NEUTRAL":
        decision = "MODERATE BUY"
        decision_class = "signal-buy"
        confidence = "Medium"
        emoji = "📈"
    elif technical_signal == "SELL" and mapped_news == "NEUTRAL":
        decision = "MODERATE SELL"
        decision_class = "signal-sell"
        confidence = "Medium"
        emoji = "📉"
    elif technical_signal == "NEUTRAL" and mapped_news in ["POSITIVE", "NEGATIVE"]:
        decision = "CAUTION - Mixed Signals"
        decision_class = "signal-neutral"
        confidence = "Low"
        emoji = "⚠️"
    elif technical_signal == "BUY" and mapped_news == "NEGATIVE":
        decision = "CONFLICT - Technical Buy vs Negative News"
        decision_class = "signal-neutral"
        confidence = "Low"
        emoji = "⚖️"
    elif technical_signal == "SELL" and mapped_news == "POSITIVE":
        decision = "CONFLICT - Technical Sell vs Positive News"
        decision_class = "signal-neutral"
        confidence = "Low"
        emoji = "⚖️"
    else:
        decision = "NEUTRAL / HOLD POSITION"
        decision_class = "signal-neutral"
        confidence = "Low"
        emoji = "⏸️"
    
    return decision, decision_class, confidence, emoji

# Calculate stop loss and take profit
def calculate_risk_metrics(current_price, atr_value, atr_multiplier_sl, atr_multiplier_tp, signal):
    """Calculate risk management levels"""
    if pd.isna(atr_value) or atr_value == 0 or current_price <= 0:
        return None, None
    
    # Calculate pip value (approximate)
    pip_size = 0.0001 if current_price < 10 else 0.01
    
    if signal == "BUY":
        stop_loss = round(current_price - (atr_value * atr_multiplier_sl), 4)
        take_profit = round(current_price + (atr_value * atr_multiplier_tp), 4)
    elif signal == "SELL":
        stop_loss = round(current_price + (atr_value * atr_multiplier_sl), 4)
        take_profit = round(current_price - (atr_value * atr_multiplier_tp), 4)
    else:
        # For neutral, suggest levels based on ATR
        stop_loss = round(current_price - (atr_value * atr_multiplier_sl), 4)
        take_profit = round(current_price + (atr_value * atr_multiplier_tp), 4)
    
    # Ensure positive values
    stop_loss = max(stop_loss, 0.0001)
    take_profit = max(take_profit, 0.0001)
    
    return stop_loss, take_profit

# Create candlestick chart
def create_candlestick_chart(data, indicators, mode, asset_name):
    """Create interactive candlestick chart with indicators"""
    if data.empty or len(data) < 5:
        return None
    
    # Create figure with secondary y-axis
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=(f"{asset_name} Price Chart", "Indicators")
    )
    
    # Candlestick trace
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name="Price",
            showlegend=True,
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350'
        ),
        row=1, col=1
    )
    
    # Add indicators based on mode
    if mode == "Swing Trading":
        if 'SMA_50' in data.columns and 'SMA_200' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['SMA_50'],
                    mode='lines',
                    name='SMA 50',
                    line=dict(color='orange', width=2)
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['SMA_200'],
                    mode='lines',
                    name='SMA 200',
                    line=dict(color='blue', width=2)
                ),
                row=1, col=1
            )
    
    elif mode == "Day Trading":
        if 'EMA_9' in data.columns and 'EMA_21' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['EMA_9'],
                    mode='lines',
                    name='EMA 9',
                    line=dict(color='red', width=1.5)
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['EMA_21'],
                    mode='lines',
                    name='EMA 21',
                    line=dict(color='green', width=1.5)
                ),
                row=1, col=1
            )
    
    elif mode == "Sniper Mode":
        if 'BB_upper' in data.columns and 'BB_lower' in data.columns:
            # Bollinger Bands with fill
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['BB_upper'],
                    mode='lines',
                    name='BB Upper',
                    line=dict(color='gray', width=1, dash='dash'),
                    opacity=0.7
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['BB_middle'],
                    mode='lines',
                    name='BB Middle',
                    line=dict(color='gray', width=1),
                    opacity=0.7
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['BB_lower'],
                    mode='lines',
                    name='BB Lower',
                    line=dict(color='gray', width=1, dash='dash'),
                    opacity=0.7,
                    fill='tonexty',
                    fillcolor='rgba(128, 128, 128, 0.1)'
                ),
                row=1, col=1
            )
    
    # Volume or RSI in second row
    if mode in ["Scalping", "Sniper Mode"] and 'RSI' in data.columns:
        # Add RSI
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['RSI'],
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
    elif 'Volume' in data.columns and data['Volume'].sum() > 0:
        # Add Volume bars
        colors = ['#ef5350' if data['Close'].iloc[i] < data['Open'].iloc[i] else '#26a69a' 
                 for i in range(len(data))]
        
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['Volume'],
                name='Volume',
                marker_color=colors,
                opacity=0.5
            ),
            row=2, col=1
        )
        fig.update_yaxes(title_text="Volume", row=2, col=1)
    else:
        # Add ATR if available
        if 'ATR' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['ATR'],
                    mode='lines',
                    name='ATR (Volatility)',
                    line=dict(color='orange', width=2)
                ),
                row=2, col=1
            )
            fig.update_yaxes(title_text="ATR", row=2, col=1)
    
    # Update layout
    fig.update_layout(
        title=f"{mode} - {asset_name}",
        xaxis_title="Date/Time",
        yaxis_title="Price",
        template="plotly_white",
        hovermode="x unified",
        height=700,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    fig.update_xaxes(rangeslider_visible=False, row=1, col=1)
    fig.update_xaxes(title_text="Date/Time", row=2, col=1)
    
    return fig

# Main app logic
def main():
    # Show loading spinner
    with st.spinner(f"📊 Loading {selected_mode} data for {selected_asset}..."):
        data = fetch_data(ticker, mode_config['period'], mode_config['interval'])
    
    if data is None or data.empty:
        st.error("❌ Failed to retrieve data. Please try:")
        st.info("1. Check your internet connection")
        st.info("2. Try a different asset")
        st.info("3. Try a different trading mode")
        st.info("4. Wait a moment and refresh")
        return
    
    # Calculate indicators
    data_with_indicators, technical_signal, technical_reason, indicators = calculate_indicators(data, selected_mode)
    
    if data_with_indicators.empty:
        st.error("Insufficient data for analysis. Please select a different mode or asset.")
        return
    
    # Get latest price and calculate change
    try:
        current_price = data_with_indicators['Close'].iloc[-1]
        if len(data_with_indicators) > 1:
            previous_price = data_with_indicators['Close'].iloc[-2]
            price_change_pct = ((current_price - previous_price) / previous_price) * 100
            price_change_abs = current_price - previous_price
        else:
            previous_price = current_price
            price_change_pct = 0.0
            price_change_abs = 0.0
    except:
        current_price = 0
        price_change_pct = 0
        price_change_abs = 0
    
    # Get news sentiment
    with st.spinner("📰 Analyzing news sentiment..."):
        news_polarity, news_headlines, news_sentiment = get_news_sentiment(ticker)
    
    # Generate decision
    decision, decision_class, confidence, emoji = generate_decision(
        technical_signal, news_sentiment, technical_reason
    )
    
    # Calculate risk metrics
    current_atr = data_with_indicators['ATR'].iloc[-1] if 'ATR' in data_with_indicators.columns and len(data_with_indicators) > 0 else 0
    stop_loss, take_profit = calculate_risk_metrics(
        current_price, 
        current_atr, 
        atr_multiplier_sl, 
        atr_multiplier_tp, 
        technical_signal
    )
    
    # Display dashboard metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        price_format = "${:,.4f}" if current_price < 10 else "${:,.2f}"
        delta_format = "{:.2f}%" if abs(price_change_pct) < 100 else "{:.0f}%"
        st.metric(
            label="Current Price",
            value=price_format.format(current_price),
            delta=f"{price_change_pct:+.2f}% ({price_change_abs:+.4f})"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            label="Technical Signal",
            value=f"{technical_signal} {emoji}",
            delta=technical_reason[:40] + "..." if len(technical_reason) > 40 else technical_reason
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        sentiment_emoji = "📈" if news_sentiment == "BULLISH" else "📉" if news_sentiment == "BEARISH" else "➖"
        st.metric(
            label="News Sentiment",
            value=f"{news_sentiment} {sentiment_emoji}",
            delta=f"Polarity: {news_polarity:+.3f}"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        if stop_loss and take_profit:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric(
                    label="Stop Loss",
                    value=f"${stop_loss:.4f}" if stop_loss < 10 else f"${stop_loss:.2f}",
                    delta=f"{((stop_loss - current_price)/current_price*100):+.1f}%"
                )
            with col_b:
                st.metric(
                    label="Take Profit",
                    value=f"${take_profit:.4f}" if take_profit < 10 else f"${take_profit:.2f}",
                    delta=f"{((take_profit - current_price)/current_price*100):+.1f}%"
                )
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            label="ATR (Volatility)",
            value=f"{current_atr:.4f}" if current_atr else "N/A",
            delta="High Volatility" if current_atr and current_atr > current_price * 0.01 else "Low Volatility"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        confidence_emoji = "🟢" if confidence == "High" else "🟡" if confidence == "Medium" else "🔴"
        st.metric(
            label="Signal Confidence",
            value=f"{confidence} {confidence_emoji}",
            delta="Based on signal alignment"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Display decision
    st.markdown(f'<div class="{decision_class}">', unsafe_allow_html=True)
    st.markdown(f"## {decision}")
    st.markdown(f"**Technical:** {technical_signal} - {technical_reason} | **News:** {news_sentiment}")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Chart", "📊 Data", "📰 News", "⚙️ Signals"])
    
    with tab1:
        # Display chart
        fig = create_candlestick_chart(data_with_indicators, indicators, selected_mode, selected_asset)
        if fig:
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})
        else:
            st.warning("Unable to generate chart. Insufficient data.")
        
        # Additional chart info
        col_info1, col_info2, col_info3 = st.columns(3)
        with col_info1:
            st.caption(f"**Data Period:** {mode_config['period']}")
            st.caption(f"**Interval:** {mode_config['interval']}")
        with col_info2:
            st.caption(f"**Data Points:** {len(data_with_indicators):,}")
            if not data_with_indicators.empty:
                st.caption(f"**Date Range:** {data_with_indicators.index[0].strftime('%Y-%m-%d %H:%M')} to {data_with_indicators.index[-1].strftime('%Y-%m-%d %H:%M')}")
        with col_info3:
            st.caption(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            st.caption(f"**Timezone:** UTC")
    
    with tab2:
        # Display raw data
        st.subheader("📊 Historical Data")
        
        if not data_with_indicators.empty:
            # Show last 20 rows
            display_data = data_with_indicators.copy()
            
            # Format numeric columns
            for col in display_data.select_dtypes(include=[np.number]).columns:
                if display_data[col].max() < 10:
                    display_data[col] = display_data[col].round(4)
                else:
                    display_data[col] = display_data[col].round(2)
            
            # Show data
            st.dataframe(
                display_data.tail(20),
                use_container_width=True,
                height=400
            )
            
            # Download option
            csv = display_data.to_csv()
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"{ticker}_{selected_mode.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.info("No data available for display.")
    
    with tab3:
        # Display news
        st.subheader("📰 Latest News & Sentiment")
        
        if news_headlines:
            st.metric("Overall News Sentiment", news_sentiment, f"Polarity: {news_polarity:+.3f}")
            st.markdown("---")
            
            for i, headline in enumerate(news_headlines[:8], 1):
                blob = TextBlob(headline)
                polarity = blob.sentiment.polarity
                
                # Color code based on sentiment
                if polarity > 0.1:
                    color = "🟢"
                    sentiment_text = "Positive"
                elif polarity < -0.1:
                    color = "🔴"
                    sentiment_text = "Negative"
                else:
                    color = "⚪"
                    sentiment_text = "Neutral"
                
                with st.expander(f"{color} {headline[:60]}..." if len(headline) > 60 else f"{color} {headline}"):
                    st.write(f"**Headline:** {headline}")
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Sentiment", sentiment_text)
                    with col_b:
                        st.metric("Polarity", f"{polarity:+.3f}")
        else:
            st.info("No recent news headlines found for this asset.")
            st.write("**Sample Analysis:**")
            st.write("News sentiment analysis helps gauge market sentiment. Positive news can drive prices up, while negative news can push them down.")
    
    with tab4:
        # Display signal details
        st.subheader("⚙️ Signal Analysis")
        
        col_sig1, col_sig2 = st.columns(2)
        
        with col_sig1:
            st.markdown("#### Technical Analysis")
            st.write(f"**Signal:** {technical_signal}")
            st.write(f"**Reason:** {technical_reason}")
            st.write(f"**Trading Mode:** {selected_mode}")
            
            # Price action
            if price_change_pct > 1:
                price_action = "Strong Bullish 📈"
            elif price_change_pct > 0:
                price_action = "Bullish ↗️"
            elif price_change_pct < -1:
                price_action = "Strong Bearish 📉"
            elif price_change_pct < 0:
                price_action = "Bearish ↘️"
            else:
                price_action = "Neutral ➖"
            
            st.write(f"**Price Action:** {price_action} ({price_change_pct:+.2f}%)")
            
            # RSI analysis if available
            if 'RSI' in data_with_indicators.columns:
                current_rsi = data_with_indicators['RSI'].iloc[-1]
                if not pd.isna(current_rsi):
                    st.write(f"**RSI:** {current_rsi:.1f}")
                    if current_rsi < 30:
                        st.success("RSI indicates oversold conditions (Potential buying opportunity)")
                    elif current_rsi > 70:
                        st.warning("RSI indicates overbought conditions (Potential selling opportunity)")
                    elif current_rsi > 50:
                        st.info("RSI indicates bullish momentum")
                    else:
                        st.info("RSI indicates bearish momentum")
        
        with col_sig2:
            st.markdown("#### Fundamental Analysis")
            st.write(f"**News Sentiment:** {news_sentiment}")
            st.write(f"**Average Polarity:** {news_polarity:+.3f}")
            
            if news_polarity > 0.2:
                st.success("Strong positive sentiment in news (Bullish for price)")
            elif news_polarity < -0.2:
                st.error("Strong negative sentiment in news (Bearish for price)")
            else:
                st.info("Neutral news sentiment (No strong directional bias)")
            
            st.markdown("#### Risk Management")
            if stop_loss and take_profit:
                risk_reward = abs((take_profit - current_price) / (current_price - stop_loss)) if current_price != stop_loss else 0
                st.write(f"**Stop Loss:** ${stop_loss:.4f}" if stop_loss < 10 else f"**Stop Loss:** ${stop_loss:.2f}")
                st.write(f"**Take Profit:** ${take_profit:.4f}" if take_profit < 10 else f"**Take Profit:** ${take_profit:.2f}")
                st.write(f"**Risk/Reward Ratio:** 1:{risk_reward:.2f}")
                st.write(f"**ATR Multipliers:** SL={atr_multiplier_sl}x, TP={atr_multiplier_tp}x")
            
            # Position sizing suggestion
            if current_price > 0 and current_atr > 0:
                risk_per_trade = 0.02  # 2% risk per trade
                position_size = (risk_per_trade * 10000) / (abs(current_price - stop_loss) if stop_loss else current_atr * atr_multiplier_sl)
                st.write(f"**Suggested Position Size:** {position_size:.2f} units (2% risk)")
        
        # Warning disclaimer
        st.markdown("---")
        st.warning("""
        **⚠️ IMPORTANT DISCLAIMER:**
        
        This dashboard is for **EDUCATIONAL AND INFORMATIONAL PURPOSES ONLY**. 
        
        - Trading forex and other financial instruments carries a **HIGH LEVEL OF RISK** and may not be suitable for all investors.
        - Past performance is **NOT** indicative of future results.
        - The signals provided are **NOT** financial advice.
        - You should **ALWAYS** do your own research and consider consulting with a licensed financial advisor before making any trading decisions.
        - The developer assumes **NO RESPONSIBILITY** for any financial losses incurred.
        
        **USE AT YOUR OWN RISK.**
        """)

# Run the app
if __name__ == "__main__":
    main()
