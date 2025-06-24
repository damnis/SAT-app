import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

# -------------------------------------------
# Functie: SAT-indicator volgens originele PRT-logica
# -------------------------------------------
def calculate_prt_sat(df):
    df = df.copy()
    df["MA150"] = df["Close"].rolling(window=150).mean()
    df["MA30"] = df["Close"].rolling(window=30).mean()

    stage = []
    for i in range(len(df)):
        if i == 0:
            stage.append(np.nan)
            continue

        ma150 = df.loc[df.index[i], "MA150"]
        ma150_prev = df.loc[df.index[i - 1], "MA150"]
        ma30 = df.loc[df.index[i], "MA30"]
        ma30_prev = df.loc[df.index[i - 1], "MA30"]
        close = df.loc[df.index[i], "Close"]

        if (
            ma150 > ma150_prev and close > ma150 and ma30 > close
            or (close > ma150 and ma30 < ma30_prev and ma30 > close)
        ):
            stage.append(-1)
        elif ma150 < ma150_prev and close < ma150 and close > ma30 and ma30 > ma30_prev:
            stage.append(1)
        elif ma150 > close and ma150 > ma150_prev:
            stage.append(-1)
        elif ma150 > close and ma150 < ma150_prev:
            stage.append(-2)
        elif ma150 < close and ma150 < ma150_prev and ma30 > ma30_prev:
            stage.append(1)
        elif ma150 < close and ma150 > ma150_prev and ma30 > ma30_prev:
            stage.append(2)
        else:
            stage.append(stage[-1])  # zelfde als vorige waarde

    df["Stage"] = stage
    df["Trend"] = pd.Series(stage).rolling(window=25).mean().values

    return df

# -------------------------------------------
# App
# -------------------------------------------
st.title("SAT-indicator (volgens originele PRT-logica)")

ticker = st.text_input("Voer een ticker in", "AAPL")
interval = st.selectbox("Interval", ["1d", "1wk"], index=0)

if interval == "1d":
    period = "730d"  # ongeveer 2 jaar aan data
else:
    period = "10y"

if st.button("Bereken SAT"):
    data = yf.download(ticker, period=period, interval=interval)

    if data.empty:
        st.error("Geen data gevonden.")
    else:
        df = data[["Open", "High", "Low", "Close"]].copy()
        df = calculate_prt_sat(df)

        st.subheader("Laatste regels van data")
        st.dataframe(df.tail())

        # -----------------------------
        # Grafiek: SAT met kleur op basis van waarde
        # -----------------------------
        fig, ax1 = plt.subplots(figsize=(12, 5))

        # Kleuren voor SAT afhankelijk van positief/negatief
        colors = ["green" if val > 0 else "red" for val in df["Stage"]]
        ax1.bar(df.index, df["Stage"], color=colors, label="Stage")
        ax1.axhline(0, color="black", linewidth=1, linestyle="--")
        ax1.set_ylabel("Stage")
        ax1.set_title(f"SAT-indicator voor {ticker}")

        # Plot Trend met dezelfde schaal als Stage
        ax1.plot(df.index, df["Trend"], color="blue", label="Trend")

        ax1.legend()
        st.pyplot(fig)

        st.success("Berekening en grafiek succesvol")
