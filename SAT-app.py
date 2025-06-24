import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# --- Titel ---
st.set_page_config(page_title="SAT App", layout="wide")
st.title("📈 SAT App (Stage And Trend)")

# --- Keuze uit aandelen ---
ticker_dict = {
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "Tesla": "TSLA",
    "NVIDIA": "NVDA",
    "Meta": "META",
    "Alphabet (Google)": "GOOGL",
    "Amazon": "AMZN",
    "ASML": "ASML.AS",
    "Shell": "SHEL.AS",
    "Aegon": "AGN.AS"
}

ticker_name = st.selectbox("Kies een aandeel:", list(ticker_dict.keys()))
ticker = ticker_dict[ticker_name]

# --- Intervalkeuze ---
interval = st.selectbox("Kies interval:", ["1d", "1wk", "4h", "1h", "15m"], index=0)

# --- Periode bepalen ---
def fetch_data(ticker, interval):
    if interval == "15m":
        period = "7d"
    elif interval == "1h":
        period = "30d"
    elif interval == "4h":
        period = "60d"
    elif interval == "1d":
        period = "360d"
    else:
        period = "360wk"

    df = yf.download(ticker, interval=interval, period=period)
    return df

# --- SAT Indicator ---
def calculate_sat(df):
    df = df.copy()
    df["ma150"] = df["Close"].rolling(window=150).mean()
    df["ma30"] = df["Close"].rolling(window=30).mean()

    stage = [0]
    for i in range(1, len(df)):
        ma150 = df["ma150"].iloc[i]
        ma150_prev = df["ma150"].iloc[i - 1]
        ma30 = df["ma30"].iloc[i]
        ma30_prev = df["ma30"].iloc[i - 1]
        close = df["Close"].iloc[i]

        if (
            (ma150 > ma150_prev and close > ma150 and ma30 > close) or
            (close > ma150 and ma30 < ma30_prev and ma30 > close)
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
            stage.append(stage[-1])

    df["Stage"] = stage
    df["Trend"] = pd.Series(stage).rolling(window=25).mean().fillna(method="bfill")
    return df

# --- Ophalen en berekenen ---
df = fetch_data(ticker, interval)
df = calculate_sat(df)

# --- Slider voor bereik ---
slider_value = st.slider("Aantal dagen/perioden in beeld:", min_value=20, max_value=len(df), value=60)
df = df.tail(slider_value)

# --- Grafiek ---
st.subheader(f"Grafiek voor {ticker_name} ({interval})")
fig, ax1 = plt.subplots(figsize=(10, 4))
ax1.bar(df.index, df["Stage"], color="lightgreen", label="Stage")
ax2 = ax1.twinx()
ax2.plot(df.index, df["Trend"], color="blue", label="Trend")
ax1.set_ylabel("Stage")
ax2.set_ylabel("Trend")
fig.tight_layout()
st.pyplot(fig)

# --- Tabel ---
st.subheader("Laatste signalen")
df_show = df[["Close", "Stage", "Trend"]].dropna().tail(30).copy()
df_show["Datum"] = df_show.index.strftime("%d-%m-%Y")
df_show = df_show[["Datum", "Close", "Stage", "Trend"]]
df_show["Close"] = df_show["Close"].round(2)
df_show["Trend"] = df_show["Trend"].round(2)
st.dataframe(df_show, use_container_width=True)

st.caption("De SAT (Stage And Trend) indicator is gebaseerd op lange- en kortetermijn-gemiddelden en kan richting geven aan beleggingsbeslissingen.")
