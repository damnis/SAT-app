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
    # Bereken voortschrijdende gemiddelden
    ma150 = df["Close"].rolling(window=150).mean()
    ma30 = df["Close"].rolling(window=30).mean()

    ma150_prev = ma150.shift(1)
    ma30_prev = ma30.shift(1)
    close = df["Close"]

    stage = pd.Series(index=df.index, dtype="float")

    for i in range(len(df)):
        if i == 0:
            stage.iloc[i] = 0  # Beginwaarde
        else:
            if (
                (ma150.iloc[i] > ma150_prev.iloc[i]) and
                (close.iloc[i] > ma150.iloc[i]) and
                (
                    (ma30.iloc[i] > close.iloc[i]) or
                    ((ma30.iloc[i] < ma30_prev.iloc[i]) and (ma30.iloc[i] > close.iloc[i]))
                )
            ):
                stage.iloc[i] = -1  # Stage 3.1

            elif (
                (ma150.iloc[i] < ma150_prev.iloc[i]) and
                (close.iloc[i] < ma150.iloc[i]) and
                (close.iloc[i] > ma30.iloc[i]) and
                (ma30.iloc[i] > ma30_prev.iloc[i])
            ):
                stage.iloc[i] = 1  # Stage 1.1

            elif (
                (ma150.iloc[i] > close.iloc[i]) and
                (ma150.iloc[i] > ma150_prev.iloc[i])
            ):
                stage.iloc[i] = -1  # Stage 3.3

            elif (
                (ma150.iloc[i] > close.iloc[i]) and
                (ma150.iloc[i] < ma150_prev.iloc[i])
            ):
                stage.iloc[i] = -2  # Stage 4

            elif (
                (ma150.iloc[i] < close.iloc[i]) and
                (ma150.iloc[i] < ma150_prev.iloc[i]) and
                (ma30.iloc[i] > ma30_prev.iloc[i])
            ):
                stage.iloc[i] = 1  # Stage 1.3

            elif (
                (ma150.iloc[i] < close.iloc[i]) and
                (ma150.iloc[i] > ma150_prev.iloc[i]) and
                (ma30.iloc[i] > ma30_prev.iloc[i])
            ):
                stage.iloc[i] = 2  # Stage 2

            else:
                stage.iloc[i] = stage.iloc[i - 1]  # Zelfde stage als vorige

    # Voeg stage toe aan dataframe
    df["Stage"] = stage

    # Bereken AVSTAGE als voortschrijdend gemiddelde van de stage
    df["Trend"] = stage.rolling(window=25).mean()

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
