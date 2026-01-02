import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# SWITCH TO CENTERED LAYOUT
st.set_page_config(page_title="Forex Master Pro", layout="centered")

st.title("📊 Forex Signal Master Pro")
st.info("👇 Select assets below:")

# CONTROLS
selected_pairs = st.multiselect(
    "Choose Assets:", 
    ["EURUSD=X", "GBPUSD=X", "JPY=X", "BTC-USD", "GC=F", "ETH-USD"],
    default=["EURUSD=X", "GBPUSD=X"]
)

# ATR CALCULATION
def calculate_atr(data, period=14):
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    return true_range.rolling(window=period).mean()

if len(selected_pairs) > 0:
    st.write(f"Analyzing {len(selected_pairs)} assets...")
    
    for symbol in selected_pairs:
        try:
            data = yf.download(symbol, period="1y", interval="1d", progress=False)
            
            if len(data) > 0:
                # Calculations
                data['SMA_50'] = data['Close'].rolling(window=50).mean()
                data['SMA_200'] = data['Close'].rolling(window=200).mean()
                data['ATR'] = calculate_atr(data)
                
                # Get Values
                try:
                    price = float(data['Close'].iloc[-1])
                    sma50 = float(data['SMA_50'].iloc[-1])
                    sma200 = float(data['SMA_200'].iloc[-1])
                    atr = float(data['ATR'].iloc[-1])
                except:
                    price = data['Close'].iloc[-1].item()
                    sma50 = data['SMA_50'].iloc[-1].item()
                    sma200 = data['SMA_200'].iloc[-1].item()
                    atr = data['ATR'].iloc[-1].item()

                # Logic
                if sma50 > sma200:
                    signal = "BUY"
                    stop_loss = price - (atr * 1.5)
                    take_profit = price + (atr * 3.0)
                    color = "success"
                else:
                    signal = "SELL"
                    stop_loss = price + (atr * 1.5)
                    take_profit = price - (atr * 3.0)
                    color = "error"

                # --- DISPLAY ---
                with st.container():
                    st.markdown("---")
                    st.subheader(f"{symbol} : {signal}")
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Current Price", f"{price:.4f}")
                    c2.metric("🎯 Take Profit", f"{take_profit:.4f}")
                    c3.metric("🛑 Stop Loss", f"{stop_loss:.4f}")
                    
                    if signal == "BUY":
                        st.success(f"**SIGNAL: BUY** (Golden Cross Detected)")
                    else:
                        st.error(f"**SIGNAL: SELL** (Death Cross Detected)")

                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(data['Close'], label='Price', color='gray', alpha=0.5)
                    ax.plot(data['SMA_50'], label='SMA 50', color='blue')
                    ax.plot(data['SMA_200'], label='SMA 200', color='red')
                    ax.axhline(y=stop_loss, color='red', linestyle='--', label='Stop Loss')
                    ax.axhline(y=take_profit, color='green', linestyle='--', label='Take Profit')
                    ax.legend()
                    st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Error: {e}")