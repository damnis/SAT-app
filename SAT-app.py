import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# DATA OPHALEN
# -----------------------
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
    df = df[
        (df["Volume"] > 0) &
        ((df["Open"] != df["Close"]) | (df["High"] != df["Low"]))
    ]
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()]
    return df

# -----------------------
# SAT-indicator
# -----------------------
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
        ((ma150 > ma150_prev) & 
         (c > ma150) & (ma30 > c)) | ((c > ma150) & (ma30 < ma30_prev) & (ma30 > c)),
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
#def calculate_sat(df):
#    df = df.copy()
#    df["range"] = df["High"] - df["Low"]
 #   df["body"] = abs(df["Close"] - df["Open"])
 #   df["direction"] = np.where(df["Close"] > df["Open"], 1, -1)
 #   df["volatiliteit"] = df["range"].rolling(window=14).mean()
#    df["SAT"] = (
#        df["direction"] *
 #       (df["body"] / df["range"].replace(0, np.nan)) *
 #       (df["range"] / df["volatiliteit"].replace(0, np.nan))
#    )
 #   df["SAT"].fillna(0, inplace=True)
 #   return df

# -----------------------
# Advies en rendement
# -----------------------
def determine_advice(df, threshold):
    df = df.copy()
    df["Trend"] = df["SAT"].rolling(window=25).mean()
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
    if "Advies" in df.columns and df["Advies"].notna().any():
        huidig_advies = df["Advies"].dropna().iloc[-1]
    else:
        huidig_advies = "Niet beschikbaar"
    return df, huidig_advies

# -----------------------
# UI - Streamlit
# -----------------------

st.title("SAT Volatiliteitsindicator")

# --- Tickerselecties ---
# --- Volledige tickerlijsten ---
aex_tickers = {
    "ABN.AS": "ABN AMRO",
    "ADYEN.AS": "Adyen",
    "AGN.AS": "Aegon",
    "AD.AS": "Ahold Delhaize",
    "AKZA.AS": "Akzo Nobel",
    "MT.AS": "ArcelorMittal",
    "ASM.AS": "ASMI",
    "ASML.AS": "ASML",
    "ASRNL.AS": "ASR Nederland",
    "BESI.AS": "BESI",
    "DSFIR.AS": "DSM-Firmenich",
    "GLPG.AS": "Galapagos",
    "HEIA.AS": "Heineken",
    "IMCD.AS": "IMCD",
    "INGA.AS": "ING Groep",
    "TKWY.AS": "Just Eat Takeaway",
    "KPN.AS": "KPN",
    "NN.AS": "NN Group",
    "PHIA.AS": "Philips",
    "PRX.AS": "Prosus",
    "RAND.AS": "Randstad",
    "REN.AS": "Relx",
    "SHELL.AS": "Shell",
    "UNA.AS": "Unilever",
    "WKL.AS": "Wolters Kluwer"
}

dow_tickers = {
    'MMM': '3M', 'AXP': 'American Express', 'AMGN': 'Amgen', 'AAPL': 'Apple', 'BA': 'Boeing',
    'CAT': 'Caterpillar', 'CVX': 'Chevron', 'CSCO': 'Cisco', 'KO': 'Coca-Cola', 'DIS': 'Disney',
    'GS': 'Goldman Sachs', 'HD': 'Home Depot', 'HON': 'Honeywell', 'IBM': 'IBM', 'INTC': 'Intel',
    'JPM': 'JPMorgan Chase', 'JNJ': 'Johnson & Johnson', 'MCD': 'McDonaldâ€™s', 'MRK': 'Merck',
    'MSFT': 'Microsoft', 'NKE': 'Nike', 'PG': 'Procter & Gamble', 'CRM': 'Salesforce',
    'TRV': 'Travelers', 'UNH': 'UnitedHealth', 'VZ': 'Verizon', 'V': 'Visa', 'WMT': 'Walmart',
    'DOW': 'Dow', 'RTX': 'RTX Corp.', 'WBA': 'Walgreens Boots'
}
nasdaq_tickers = {
    'MSFT': 'Microsoft', 'NVDA': 'NVIDIA', 'AAPL': 'Apple', 'AMZN': 'Amazon', 'META': 'Meta',
    'NFLX': 'Netflix', 'GOOG': 'Google', 'GOOGL': 'Alphabet', 'TSLA': 'Tesla', 'CSCO': 'Cisco',
    'INTC': 'Intel', 'ADBE': 'Adobe', 'CMCSA': 'Comcast', 'PEP': 'PepsiCo', 'COST': 'Costco',
    'AVGO': 'Broadcom', 'QCOM': 'Qualcomm', 'TMUS': 'T-Mobile', 'TXN': 'Texas Instruments',
    'AMAT': 'Applied Materials'
}

ustech_tickers = {
    "SMCI": "Super Micro Computer",
    "PLTR": "Palantir",
    "SNOW": "Snowflake",
    "NVDA": "NVIDIA",
    "AMD": "AMD",
    "MDB": "MongoDB",
    "DDOG": "Datadog",
    "CRWD": "CrowdStrike",
    "ZS": "Zscaler",
    "TSLA": "Tesla",
    "AAPL": "Apple",
    "GOOGL": "Alphabet (GOOGL)",
    "MSFT": "Microsoft"
}
tab_labels = ["🇺🇸 Dow Jones", "🇺🇸 Nasdaq", "🇺🇸 US Tech", "🇳🇱 AEX"]
selected_tab = st.radio("Kies beurs", tab_labels, horizontal=True)

if selected_tab == "🇺🇸 Dow Jones":
    ticker_label = st.selectbox("Dow Jones aandeel", [f"{k} - {v}" for k, v in dow_tickers.items()])
elif selected_tab == "🇺🇸 Nasdaq":
    ticker_label = st.selectbox("Nasdaq aandeel", [f"{k} - {v}" for k, v in nasdaq_tickers.items()])
elif selected_tab == "🇺🇸 US Tech":
    ticker_label = st.selectbox("US Tech aandeel", [f"{k} - {v}" for k, v in ustech_tickers.items()])
else:
    ticker_label = st.selectbox("AEX aandeel", [f"{k} - {v}" for k, v in aex_tickers.items()])

ticker, ticker_name = ticker_label.split(" - ", 1)

# --- Interval + Slider ---
interval_optie = st.selectbox("Kies de interval", ["Dagelijks", "Wekelijks", "4-uur", "1-uur", "15-minuten"])
interval_mapping = {
    "Dagelijks": "1d",
    "Wekelijks": "1wk",
    "4-uur": "4h",
    "1-uur": "1h",
    "15-minuten": "15m"
}
interval = interval_mapping[interval_optie]
thresh = st.slider("Gevoeligheid van trendverandering", 0.005, 1.0, 0.05, step=0.005)

# --- Berekening ---
df = fetch_data(ticker, interval)
df = calculate_sat(df)
df, huidig_advies = determine_advice(df, threshold=thresh)

# --- Headeradvies ---
advies_kleur = "green" if huidig_advies == "Kopen" else "red" if huidig_advies == "Verkopen" else "gray"
st.markdown(
    f"""
    <h3>SAT-indicator en trend voor <span style='color:#3366cc'>{ticker_name}</span></h3>
    <h2 style='color:{advies_kleur}'>Huidig advies: {huidig_advies}</h2>
    """,
    unsafe_allow_html=True
)

# --- Grafiek ---
# --- Instelbare weergaveperiode ---
#st.sidebar.markdown("### Weergaveperiode")
#visible_period = st.sidebar.selectbox(
#    "Toon laatste ...",
 #   options=[30, 60, 90, 120, 160, 250],
#    index=3,  # standaard bijv. 120
  #  format_func=lambda x: f"{x} candles"
#)

# --- Beperk data voor weergave in grafiek/tabel ---
#df_filtered = df.tail(visible_period)
# Controleer en reset MultiIndex als die er is
#if isinstance(df_filtered.columns, pd.MultiIndex):
#    df_filtered.columns = ['_'.join(filter(None, col)).strip() for col in df_filtered.columns.values]

# Reset index zodat 'Date' weer gewone kolom is
#df_filtered = df_filtered.reset_index()

#st.subheader("Grafiek met SAT Indicator")
#st.write("Kolomnamen df_filtered:", df_filtered.columns)
#st.write("Index:", df_filtered.index)
#st.line_chart(df_filtered[["Close", "SAT", "Trend"]])
fig, ax1 = plt.subplots(figsize=(10, 4))
fig, ax = plt.subplots(figsize=(10, 4))

# Kleuren voor positieve/negatieve SAT
sat_colors = ["green" if v > 0 else "red" for v in df["SAT"]]
ax.bar(df.index, df["SAT"], color=sat_colors, label="SAT")

# Trend-lijn op dezelfde as (gÃ©Ã©n twinx)
ax.plot(df.index, df["Trend"], color="blue", label="Trend", linewidth=2)

# Nullijn
ax.axhline(0, color="black", linestyle="--", linewidth=1)

# Labels en legenda
ax.set_ylabel("SAT / Trend (zelfde schaal)")
ax.set_title("SAT en Trend Indicator")
ax.legend()

fig.tight_layout()
st.pyplot(fig)

#fig, ax1 = plt.subplots(figsize=(10, 4))
#ax1.bar(df.index, df["SAT"], color="orange", label="SAT")
#ax2 = ax1.twinx()
#ax2.plot(df.index, df["Trend"], color="blue", label="Trend")
#ax1.set_ylabel("SAT")
#ax2.set_ylabel("Trend")
#fig.tight_layout()
#st.pyplot(fig)

# --- Tabel ---
st.subheader("Laatste signalen en rendement")

kolommen = ["Close", "Advies", "SAT", "Trend", "Markt-%", "SAT-%"]
tabel = df[kolommen].dropna().tail(30).copy()
#tabel.index = pd.to_datetime(tabel.index, errors="coerce")
# Afronden van relevante kolommen
tabel["Close"] = tabel["Close"].round(2)
tabel["SAT"] = tabel["SAT"].round(2)
tabel["Trend"] = tabel["Trend"].round(2)

# Datumkolom en volgorde
tabel.index = pd.to_datetime(tabel.index, errors="coerce")
tabel["Datum"] = tabel.index.strftime("%d-%m-%Y")
tabel = tabel[["Datum"] + kolommen]
#tabel["Datum"] = tabel.index.strftime("%d-%m-%Y")
#tabel = tabel[["Datum"] + kolommen]
tabel["Markt-%"] = (tabel["Markt-%"].astype(float) * 100).map("{:+.2f}%".format)
tabel["SAT-%"] = (tabel["SAT-%"].astype(float) * 100).map("{:+.2f}%".format)

# HTML-rendering
html = """<style>table {border-collapse: collapse; width: 100%; font-family: Arial; font-size: 14px;}
th {background-color: #004080; color: white; padding: 6px; text-align: center;}
td {border: 1px solid #ddd; padding: 6px; text-align: right; background-color: #f9f9f9; color: #222;}
tr:nth-child(even) td {background-color: #eef2f7;} tr:hover td {background-color: #d0e4f5;}</style>
<table><thead><tr>
<th style='width:110px;'>Datum</th><th>Close</th><th>Advies</th><th>SAT</th><th>Trend</th><th>Markt-%</th><th>SAT-%</th>
</tr></thead><tbody>"""
for _, row in tabel.iterrows():
    html += "<tr>" + "".join([f"<td>{val}</td>" for val in row]) + "</tr>"
html += "</tbody></table>"
st.markdown(html, unsafe_allow_html=True)
