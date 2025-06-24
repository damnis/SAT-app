import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

st.title("SAT-indicator")

# ---------------------------------
# Functie om SAT & Trend te berekenen
# ---------------------------------
def calculate_sat(df):
    df = df.copy()
    df["MA150"] = df["Close"].rolling(window=150).mean()
    df["MA30"] = df["Close"].rolling(window=30).mean()
    df["MA150_prev"] = df["MA150"].shift(1)
    df["MA30_prev"] = df["MA30"].shift(1)

    c = df["Close"]
    ma150 = df["MA150"]
    ma150_prev = df["MA150_prev"]
    ma30 = df["MA30"]
    ma30_prev = df["MA30_prev"]

    condlist = [
        # stage 3.1
        ((ma150 > ma150_prev) & (c > ma150) & (ma30 > c)) | ((c > ma150) & (ma30 < ma30_prev) & (ma30 > c)),
        # stage 1.1
        (ma150 < ma150_prev) & (c < ma150) & (c > ma30) & (ma30 > ma30_prev),
        # stage 3.3
        (ma150 > c) & (ma150 > ma150_prev),
        # stage 4
        (ma150 > c) & (ma150 < ma150_prev),
        # stage 1.3
        (ma150 < c) & (ma150 < ma150_prev) & (ma30 > ma30_prev),
        # stage 2
        (ma150 < c) & (ma150 > ma150_prev) & (ma30 > ma30_prev),
    ]

    choicelist = [-1, 1, -1, -2, 1, 2]

    df["Stage"] = np.select(condlist, choicelist, default=np.nan)

    # Vul lege waarden met vorige waarde
    df["Stage"] = df["Stage"].ffill()

    # Bepaal Trend als MA(25) van Stage
    df["Trend"] = df["Stage"].rolling(window=25).mean()

    return df

# ---------------------------------
# Advies en rendement
# ---------------------------------
def determine_advice(df, threshold):
    df = df.copy()
    df["TrendChange"] = df["Trend"] - df["Trend"].shift(1)
    df["Advies"] = np.nan
    df.loc[df["TrendChange"] > threshold, "Advies"] = "Kopen"
    df.loc[df["TrendChange"] < -threshold, "Advies"] = "Verkopen"
    df["Advies"] = df["Advies"].ffill()
    df["AdviesGroep"] = (df["Advies"] != df["Advies"].shift()).cumsum()

    rendementen = []
    sat_rendementen = []
    for _, groep in df.groupby("AdviesGroep"):
        start = groep["Close"].iloc[0]
        eind = groep["Close"].iloc[-1]
        advies = groep["Advies"].iloc[0]
        markt_rendement = (eind - start) / start
        sat_rendement = markt_rendement if advies == "Kopen" else -markt_rendement
        rendementen.extend([markt_rendement] * len(groep))
        sat_rendementen.extend([sat_rendement] * len(groep))

    df["Markt-%"] = rendementen
    df["SAT-%"] = sat_rendementen
    if df["Advies"].notna().any():
        huidig_advies = df["Advies"].dropna().iloc[-1]
    else:
        huidig_advies = "Niet beschikbaar"
    return df, huidig_advies

# ---------------------------------
# Invoer
# ---------------------------------
tab1, tab2 = st.tabs(["SAT Grafiek", "SAT Tabel"])

with st.sidebar:
    ticker = st.text_input("Ticker", value="AAPL")
    interval = st.selectbox("Interval", ["1d", "1wk"])
    gevoeligheid = st.slider("Gevoeligheid (advies)", 0.0, 1.0, 0.1, 0.05)

# ---------------------------------
# Data ophalen
# ---------------------------------
periode = "400d" if interval == "1d" else "400wk"
data = yf.download(ticker, period=periode, interval=interval)
data.dropna(inplace=True)

# ---------------------------------
# Berekening en advies
# ---------------------------------
data = calculate_sat(data)
data, advies = determine_advice(data, threshold=gevoeligheid)

# ---------------------------------
# Tabs
# ---------------------------------
with tab1:
    st.subheader(f"Grafiek SAT & Trend voor {ticker.upper()} ({interval})")
    fig, ax1 = plt.subplots(figsize=(12, 5))

    stage_pos = data["Stage"].clip(lower=0)
    stage_neg = data["Stage"].clip(upper=0)

    ax1.bar(data.index, stage_pos, color="green", label="Stage > 0")
    ax1.bar(data.index, stage_neg, color="red", label="Stage < 0")
    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.plot(data.index, data["Trend"], color="blue", label="Trend")

    ax1.set_ylabel("Stage / Trend")
    ax1.legend(loc="upper left")
    st.pyplot(fig)
    st.metric("Laatste Advies", advies)

with tab2:
    st.subheader("Laatste 100 rijen")
    st.dataframe(data.tail(100)[["Close", "Stage", "Trend", "Advies", "Markt-%", "SAT-%"]].round(2))
