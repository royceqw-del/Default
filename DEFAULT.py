"""
Mobile-Optimized Default Risk Dashboard
- Live monitoring of SPY, TBT, VIX, GLD, BTC
- Toggleable backtesting
- Real-time aggressive daily playbook
- Fully single-column layout for phones
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
from math import log, sqrt, exp, erf

# ----------------- Black-Scholes helpers -----------------
def norm_cdf(x):
    return 0.5 * (1 + erf(x / sqrt(2)))

def bs_price(S, K, T, r, sigma, option_type='put'):
    if T <= 0 or sigma <= 0:
        return max(K - S, 0) if option_type=='put' else max(S - K, 0)
    d1 = (log(S/K) + (r+0.5*sigma**2)*T)/(sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)
    if option_type=='call':
        return S*norm_cdf(d1) - K*exp(-r*T)*norm_cdf(d2)
    else:
        return K*exp(-r*T)*norm_cdf(-d2) - S*norm_cdf(-d1)

# ----------------- Sidebar Controls -----------------
st.set_page_config(page_title="Mobile Default Risk Dashboard", layout="wide")
st.sidebar.title("Controls")

# Live monitoring thresholds
ten_y_thresh = st.sidebar.number_input("10Y yield trigger (%)", value=5.0, step=0.1, format="%.2f")
vix_panic = st.sidebar.number_input("VIX panic trigger", value=28.0, step=1.0)
gold_breakout = st.sidebar.number_input("Gold breakout (GLD $)", value=195.0, step=1.0)

# Backtest settings
st.sidebar.markdown("---")
st.sidebar.title("Backtest Settings")
enable_bt = st.sidebar.checkbox("Enable Backtesting", value=True)
bt_start = st.sidebar.date_input("Backtest start date", value=dt.date(2018,1,1))
bt_end = st.sidebar.date_input("Backtest end date (inclusive)", value=dt.date.today())
hold_days = st.sidebar.number_input("Hold days per trade", min_value=1, max_value=180, value=10)
leverage = st.sidebar.slider("Aggressive leverage multiplier", 1, 10, 5)
capital = st.sidebar.number_input("Starting capital ($)", min_value=1000, value=100000, step=1000)
assets = st.sidebar.multiselect("Assets to backtest", options=["SPY","TBT","VXX","GLD"], default=["TBT","VXX","SPY"])
include_options = st.sidebar.checkbox("Include simplified options strategies", value=True)
opt_strategy = st.sidebar.selectbox("Options play", ["Buy deep OTM puts","Buy ATM puts","VIX call proxy"])
enable_playbook = st.sidebar.checkbox("Generate Daily Playbook", value=True)

# ----------------- Live Monitoring -----------------
TRACK_TICKERS = {"10Y_YIELD_PROXY":"^TNX","VIX":"^VIX","GLD":"GLD","BTC":"BTC-USD","SPY":"SPY","TBT":"TBT","VXX":"VXX"}

@st.cache_data(ttl=30)
def fetch_current_price(ticker):
    try:
        t = yf.Ticker(ticker)
        df = t.history(period='1d', interval='1m')
        return float(df['Close'].iloc[-1])
    except:
        return None

st.header("Live Market Snapshot (Mobile)")

snap = {}
for name, tk in TRACK_TICKERS.items():
    price = fetch_current_price(tk)
    if price is None:
        st.write(f"**{name}** — data unavailable")
        snap[name] = None
    else:
        st.metric(label=name, value=round(price,4))
        snap[name] = price

# Signals
st.subheader("Signals")
signals = []
tnx = snap.get("10Y_YIELD_PROXY")
if tnx is not None:
    ten_y = tnx/10.0 if tnx>50 else tnx
    signals.append(f"🚨 10Y yield {ten_y:.2f}% >= {ten_y_thresh} → Scale TBT/TMV" if ten_y>=ten_y_thresh else f"10Y yield {ten_y:.2f}% — below trigger")
vixp = snap.get("VIX")
if vixp is not None:
    signals.append(f"🚨 VIX {vixp:.2f} >= {vix_panic} → Buy UVXY calls" if vixp>=vix_panic else f"VIX {vixp:.2f} — below panic threshold")
gld = snap.get("GLD")
if gld is not None:
    signals.append(f"🟢 GLD {gld:.2f} > {gold_breakout} → Accumulate GDX" if gld>=gold_breakout else f"GLD {gld:.2f} — below breakout")
for s in signals:
    st.write(s)

# ----------------- Backtesting -----------------
st.header("Backtesting")
if enable_bt:
    with st.expander("View Backtest Controls"):
        st.write(f"Backtest window: {bt_start} → {bt_end} | Hold days: {hold_days} | Leverage: {leverage}x | Capital: ,")
        st.info("Historical simulation placeholder — run locally with internet to fetch data and compute multi-asset backtest.")
else:
    st.info("Backtesting disabled. Toggle in sidebar to run simulations.")

# ----------------- Aggressive Daily Playbook -----------------
if enable_playbook:
    st.header("Daily Aggressive Playbook")
    playbook = []

    # SPY puts triggered by VIX or equity drop
    if vixp is not None and vixp>=vix_panic:
        playbook.append({"Asset":"SPY","Action":"Buy ATM Put","Instrument":"Options","Strike":"ATM","Expiry":"30d","Position Size":f"{5*leverage}% capital","Risk Score":"High"})

    # TBT calls triggered by 10Y yield spike
    if tnx is not None and ten_y>=ten_y_thresh:
        playbook.append({"Asset":"TBT","Action":"Buy Call","Instrument":"ETF/Options","Strike":"ATM","Expiry":"30d","Position Size":f"{5*leverage}% capital","Risk Score":"Medium"})

    # UVXY calls triggered by VIX
    if vixp is not None and vixp>=vix_panic:
        playbook.append({"Asset":"UVXY","Action":"Buy Call","Instrument":"Options","Strike":"ATM","Expiry":"30d","Position Size":f"{3*leverage}% capital","Risk Score":"Very High"})

    # GLD calls triggered by gold breakout
    if gld is not None and gld>=gold_breakout:
        playbook.append({"Asset":"GLD","Action":"Buy Call","Instrument":"Options","Strike":"ATM","Expiry":"30d","Position Size":f"{2*leverage}% capital","Risk Score":"Medium"})

    if playbook:
        df_playbook = pd.DataFrame(playbook)
        st.dataframe(df_playbook, width=600, height=300)  # scrollable on mobile
        if st.button("Simulate Playbook"):
            st.info("Simulated equity curve placeholder — run locally with internet to fetch historical prices and options data.")
    else:
        st.write("No aggressive trades triggered today based on current thresholds.")

st.markdown("---")
st.caption("Educational dashboard: simplified aggressive scenario playbook. Not financial advice. Leverage multiplies gains and losses; high risk of drawdowns.")