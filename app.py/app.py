

Skip to content
Using Gmail with screen readers
Enable desktop notifications for Gmail.
   OK  No thanks
Your new, simpler toolbar
Now it’s easier to focus on common tasks. Find other actions under More (⋮)
2 of 2,113
deepseek_python_20260114_bbce98.py
Inbox

Arthur Delight <delightarthur365@gmail.com>
Attachments
11:55 AM (21 minutes ago)
to me

 One attachment
  •  Scanned by Gmail
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
    }
    .signal-sell {
        background-color: #F44336;
        color: white;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .signal-neutral {
        background-color: #FF9800;
        color: white;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .metric-card {
        background-color: #f5f5f5;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #1E88E5;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">📊 Professional Forex Trading Dashboard</h1>', unsafe_allow_html=True)

# Sidebar
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
            "description": "1-year daily data, Golden Cross (SMA50 > SMA200)"
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
    
    st.markdown("---")
    st.markdown("**Dashboard Info:**")
    st.caption("Data provided by Yahoo Finance")
    st.caption("Signals are for educational purposes only")

# Function to fetch data
@st.cache_data(ttl=60)
def fetch_data(ticker, period, interval):
    try:
        data = yf.download(
            tickers=ticker,
            period=period,
            interval=interval,
            progress=False
        )
        if data.empty:
            st.error(f"No data retrieved for {ticker}. Check the ticker symbol.")
            return None
        
        # Ensure we have the required columns
        if 'Close' not in data.columns:
            if 'Adj Close' in data.columns:
                data = data.rename(columns={'Adj Close': 'Close'})
            else:
                data['Close'] = data.iloc[:, -1]
        
        # Ensure we have Open, High, Low columns
        for col in ['Open', 'High', 'Low']:
            if col not in data.columns:
                data[col] = data['Close']
        
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# Function to calculate indicators based on trading mode
def calculate_indicators(data, mode):
    df = data.copy()
    
    # Calculate ATR for all modes
    if len(df) > 14:
        df['ATR'] = ta.volatility.AverageTrueRange(
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            window=14
        ).average_true_range()
    else:
        df['ATR'] = np.nan
    
    if mode == "Swing Trading":
        # Calculate SMAs
        df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
        df['SMA_200'] = ta.trend.sma_indicator(df['Close'], window=200)
        
        # Generate signal
        if len(df) >= 200 and not pd.isna(df['SMA_50'].iloc[-1]) and not pd.isna(df['SMA_200'].iloc[-1]):
            if df['SMA_50'].iloc[-1] > df['SMA_200'].iloc[-1]:
                signal = "BUY"
                signal_reason = "Golden Cross (SMA50 > SMA200)"
            else:
                signal = "SELL"
                signal_reason = "Death Cross (SMA50 < SMA200)"
        else:
            signal = "NEUTRAL"
            signal_reason = "Insufficient data for SMA calculation"
            
        indicators = ['SMA_50', 'SMA_200']
        
    elif mode == "Day Trading":
        # Calculate EMAs
        df['EMA_9'] = ta.trend.ema_indicator(df['Close'], window=9)
        df['EMA_21'] = ta.trend.ema_indicator(df['Close'], window=21)
        
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
            else:
                signal = "NEUTRAL"
                signal_reason = "No EMA cross detected"
        else:
            signal = "NEUTRAL"
            signal_reason = "Insufficient data for EMA calculation"
            
        indicators = ['EMA_9', 'EMA_21']
        
    elif mode == "Scalping":
        # Calculate RSI
        df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
        
        # Generate signal
        if len(df) >= 14:
            current_rsi = df['RSI'].iloc[-1]
            if current_rsi < 30:
                signal = "BUY"
                signal_reason = f"RSI oversold ({current_rsi:.2f} < 30)"
            elif current_rsi > 70:
                signal = "SELL"
                signal_reason = f"RSI overbought ({current_rsi:.2f} > 70)"
            else:
                signal = "NEUTRAL"
                signal_reason = f"RSI neutral ({current_rsi:.2f})"
        else:
            signal = "NEUTRAL"
            signal_reason = "Insufficient data for RSI calculation"
            
        indicators = ['RSI']
        
    else:  # Sniper Mode
        # Calculate Bollinger Bands and RSI
        bb = ta.volatility.BollingerBands(close=df['Close'], window=20, window_dev=2)
        df['BB_upper'] = bb.bollinger_hband()
        df['BB_middle'] = bb.bollinger_mavg()
        df['BB_lower'] = bb.bollinger_lband()
        df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
        
        # Generate signal
        if len(df) >= 20:
            current_price = df['Close'].iloc[-1]
            current_bb_lower = df['BB_lower'].iloc[-1]
            current_rsi = df['RSI'].iloc[-1]
            
            if current_price < current_bb_lower and current_rsi < 25:
                signal = "BUY"
                signal_reason = f"Price < Lower BB & RSI oversold ({current_rsi:.2f} < 25)"
            elif current_price > df['BB_upper'].iloc[-1] and current_rsi > 75:
                signal = "SELL"
                signal_reason = f"Price > Upper BB & RSI overbought ({current_rsi:.2f} > 75)"
            else:
                signal = "NEUTRAL"
                signal_reason = "No sniper signal detected"
        else:
            signal = "NEUTRAL"
            signal_reason = "Insufficient data for calculation"
            
        indicators = ['BB_upper', 'BB_middle', 'BB_lower', 'RSI']
    
    return df, signal, signal_reason, indicators

# Function to get news sentiment
@st.cache_data(ttl=300)
def get_news_sentiment(ticker_symbol):
    try:
        # Extract asset name for news search
        asset_name = ticker_symbol.replace('=X', '').replace('-USD', '')
        
        # Try multiple news sources
        sources = [
            f"https://news.google.com/rss/search?q={asset_name}+forex+OR+currency&hl=en-US&gl=US&ceid=US:en",
            f"https://news.google.com/rss/search?q={asset_name}+trading&hl=en-US&gl=US&ceid=US:en"
        ]
        
        all_headlines = []
        
        for url in sources:
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'xml')
                items = soup.find_all('item')
                
                for item in items[:10]:  # Limit to 10 per source
                    title = item.find('title').text
                    # Remove source from title (e.g., "Title - Source")
                    if ' - ' in title:
                        title = title.split(' - ')[0]
                    all_headlines.append(title)
                
            except Exception as e:
                continue
        
        if not all_headlines:
            return 0.0, ["No news headlines found"], "No news data available"
        
        # Analyze sentiment
        polarities = []
        for headline in all_headlines[:15]:  # Analyze first 15 headlines
            blob = TextBlob(headline)
            polarities.append(blob.sentiment.polarity)
        
        avg_polarity = np.mean(polarities) if polarities else 0.0
        
        # Determine sentiment category
        if avg_polarity > 0.1:
            sentiment = "POSITIVE"
        elif avg_polarity < -0.1:
            sentiment = "NEGATIVE"
        else:
            sentiment = "NEUTRAL"
        
        return avg_polarity, all_headlines[:5], sentiment
        
    except Exception as e:
        return 0.0, [f"Error fetching news: {str(e)[:50]}..."], "NEUTRAL"

# Decision Engine
def generate_decision(technical_signal, news_sentiment, technical_reason):
    if technical_signal == "BUY" and news_sentiment == "POSITIVE":
        decision = "STRONG BUY (ENTER NOW)"
        decision_class = "signal-buy"
        confidence = "High"
    elif technical_signal == "SELL" and news_sentiment == "NEGATIVE":
        decision = "STRONG SELL (EXIT NOW)"
        decision_class = "signal-sell"
        confidence = "High"
    elif technical_signal == "BUY" and news_sentiment == "NEUTRAL":
        decision = "MODERATE BUY"
        decision_class = "signal-buy"
        confidence = "Medium"
    elif technical_signal == "SELL" and news_sentiment == "NEUTRAL":
        decision = "MODERATE SELL"
        decision_class = "signal-sell"
        confidence = "Medium"
    elif technical_signal == "NEUTRAL" and (news_sentiment in ["POSITIVE", "NEGATIVE"]):
        decision = "CAUTION - Mixed Signals"
        decision_class = "signal-neutral"
        confidence = "Low"
    elif technical_signal == "BUY" and news_sentiment == "NEGATIVE":
        decision = "CONFLICT - Technical Buy vs Negative News"
        decision_class = "signal-neutral"
        confidence = "Low"
    elif technical_signal == "SELL" and news_sentiment == "POSITIVE":
        decision = "CONFLICT - Technical Sell vs Positive News"
        decision_class = "signal-neutral"
        confidence = "Low"
    else:
        decision = "NEUTRAL / HOLD"
        decision_class = "signal-neutral"
        confidence = "Low"
    
    return decision, decision_class, confidence

# Calculate stop loss and take profit
def calculate_risk_metrics(current_price, atr_value, atr_multiplier_sl, atr_multiplier_tp, signal):
    if pd.isna(atr_value) or atr_value == 0:
        return None, None
    
    if signal == "BUY":
        stop_loss = current_price - (atr_value * atr_multiplier_sl)
        take_profit = current_price + (atr_value * atr_multiplier_tp)
    elif signal == "SELL":
        stop_loss = current_price + (atr_value * atr_multiplier_sl)
        take_profit = current_price - (atr_value * atr_multiplier_tp)
    else:
        stop_loss = current_price - (atr_value * atr_multiplier_sl)
        take_profit = current_price + (atr_value * atr_multiplier_tp)
    
    return stop_loss, take_profit

# Create candlestick chart
def create_candlestick_chart(data, indicators, mode):
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=(f"{selected_asset} Price Chart", "Volume")
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
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Add indicators based on mode
    if mode == "Swing Trading" and 'SMA_50' in data.columns:
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
    
    elif mode == "Day Trading" and 'EMA_9' in data.columns:
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
    
    elif mode == "Sniper Mode" and 'BB_upper' in data.columns:
        # Bollinger Bands
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
    
    # Volume trace
    if 'Volume' in data.columns:
        colors = ['red' if data['Close'].iloc[i] < data['Open'].iloc[i] else 'green' 
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
    
    # Add RSI subplot if in Scalping or Sniper mode
    if (mode == "Scalping" or mode == "Sniper Mode") and 'RSI' in data.columns:
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
    else:
        fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    # Update layout
    fig.update_layout(
        title=f"{selected_mode} - {selected_asset}",
        xaxis_title="Date/Time",
        yaxis_title="Price",
        template="plotly_dark",
        hovermode="x unified",
        height=700,
        showlegend=True
    )
    
    fig.update_xaxes(rangeslider_visible=False)
    
    return fig

# Main app logic
def main():
    # Fetch data
    with st.spinner(f"Fetching {selected_mode} data for {selected_asset}..."):
        data = fetch_data(ticker, mode_config['period'], mode_config['interval'])
    
    if data is None or data.empty:
        st.error("Failed to retrieve data. Please try another asset or mode.")
        return
    
    # Calculate indicators
    data_with_indicators, technical_signal, technical_reason, indicators = calculate_indicators(data, selected_mode)
    
    # Get latest price
    current_price = data_with_indicators['Close'].iloc[-1]
    previous_price = data_with_indicators['Close'].iloc[-2] if len(data_with_indicators) > 1 else current_price
    price_change = ((current_price - previous_price) / previous_price) * 100
    
    # Get news sentiment
    with st.spinner("Analyzing news sentiment..."):
        news_polarity, news_headlines, news_sentiment = get_news_sentiment(ticker)
    
    # Generate decision
    decision, decision_class, confidence = generate_decision(technical_signal, news_sentiment, technical_reason)
    
    # Calculate risk metrics
    current_atr = data_with_indicators['ATR'].iloc[-1] if 'ATR' in data_with_indicators.columns and len(data_with_indicators) > 0 else 0
    stop_loss, take_profit = calculate_risk_metrics(
        current_price, 
        current_atr, 
        atr_multiplier_sl, 
        atr_multiplier_tp, 
        technical_signal
    )
    
    # Display dashboard
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            label="Current Price",
            value=f"${current_price:.4f}" if current_price < 1000 else f"${current_price:.2f}",
            delta=f"{price_change:.2f}%"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            label="Technical Signal",
            value=technical_signal,
            delta=technical_reason[:30] + "..." if len(technical_reason) > 30 else technical_reason
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            label="News Sentiment",
            value=news_sentiment,
            delta=f"Polarity: {news_polarity:.3f}"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        if stop_loss and take_profit:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric(
                    label="Stop Loss",
                    value=f"${stop_loss:.4f}" if stop_loss < 1000 else f"${stop_loss:.2f}",
                    delta=f"{(stop_loss - current_price):.4f}"
                )
            with col_b:
                st.metric(
                    label="Take Profit",
                    value=f"${take_profit:.4f}" if take_profit < 1000 else f"${take_profit:.2f}",
                    delta=f"{(take_profit - current_price):.4f}"
                )
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            label="ATR (Volatility)",
            value=f"{current_atr:.4f}" if current_atr else "N/A",
            delta="Higher = More Volatile" if current_atr and current_atr > 0.01 else "Lower = Less Volatile"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            label="Confidence",
            value=confidence,
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
        fig = create_candlestick_chart(data_with_indicators, indicators, selected_mode)
        st.plotly_chart(fig, use_container_width=True)
        
        # Additional chart info
        col_info1, col_info2, col_info3 = st.columns(3)
        with col_info1:
            st.caption(f"Data Period: {mode_config['period']}")
            st.caption(f"Interval: {mode_config['interval']}")
        with col_info2:
            st.caption(f"Data Points: {len(data_with_indicators)}")
            st.caption(f"Date Range: {data_with_indicators.index[0]} to {data_with_indicators.index[-1]}")
        with col_info3:
            st.caption(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    with tab2:
        # Display raw data
        st.subheader("Historical Data")
        
        # Show last 20 rows
        display_data = data_with_indicators.copy()
        
        # Format numeric columns
        for col in display_data.select_dtypes(include=[np.number]).columns:
            if display_data[col].max() < 1000:
                display_data[col] = display_data[col].round(4)
            else:
                display_data[col] = display_data[col].round(2)
        
        st.dataframe(
            display_data.tail(20),
            use_container_width=True
        )
        
        # Download option
        csv = display_data.to_csv()
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"{ticker}_{selected_mode.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )
    
    with tab3:
        # Display news
        st.subheader("Latest News Headlines")
        
        if news_headlines:
            for i, headline in enumerate(news_headlines[:10], 1):
                blob = TextBlob(headline)
                polarity = blob.sentiment.polarity
                
                # Color code based on sentiment
                if polarity > 0.1:
                    color = "🟢"
                elif polarity < -0.1:
                    color = "🔴"
                else:
                    color = "⚪"
                
                st.write(f"{color} **{headline}**")
                st.caption(f"Sentiment Polarity: {polarity:.3f}")
                st.divider()
        else:
            st.info("No recent news headlines found for this asset.")
    
    with tab4:
        # Display signal details
        st.subheader("Signal Analysis")
        
        col_sig1, col_sig2 = st.columns(2)
        
        with col_sig1:
            st.markdown("#### Technical Analysis")
            st.write(f"**Signal:** {technical_signal}")
            st.write(f"**Reason:** {technical_reason}")
            st.write(f"**Mode:** {selected_mode}")
            st.write(f"**Price Action:** {'Bullish' if price_change > 0 else 'Bearish' if price_change < 0 else 'Neutral'}")
            
            if 'RSI' in data_with_indicators.columns:
                current_rsi = data_with_indicators['RSI'].iloc[-1]
                st.write(f"**RSI:** {current_rsi:.2f}")
                if current_rsi < 30:
                    st.info("RSI indicates oversold conditions")
                elif current_rsi > 70:
                    st.warning("RSI indicates overbought conditions")
        
        with col_sig2:
            st.markdown("#### Fundamental Analysis")
            st.write(f"**News Sentiment:** {news_sentiment}")
            st.write(f"**Average Polarity:** {news_polarity:.3f}")
            
            if news_polarity > 0.2:
                st.success("Strong positive sentiment in news")
            elif news_polarity < -0.2:
                st.error("Strong negative sentiment in news")
            else:
                st.info("Neutral news sentiment")
            
            st.markdown("#### Risk Management")
            if stop_loss and take_profit:
                st.write(f"**Stop Loss:** ${stop_loss:.4f}")
                st.write(f"**Take Profit:** ${take_profit:.4f}")
                st.write(f"**Risk/Reward Ratio:** 1:{atr_multiplier_tp/atr_multiplier_sl:.2f}")
            
            st.write(f"**ATR Multipliers:** SL={atr_multiplier_sl}x, TP={atr_multiplier_tp}x")
        
        # Warning disclaimer
        st.markdown("---")
        st.warning("""
        **Disclaimer:** This dashboard is for educational and informational purposes only. 
        Trading forex and other financial instruments carries a high level of risk and may not be suitable for all investors. 
        Past performance is not indicative of future results. Always do your own research and consider consulting with a licensed financial advisor.
        """)

# Run the app
if __name__ == "__main__":
    main()
deepseek_python_20260114_bbce98.py
Displaying deepseek_python_20260114_bbce98.py.
