import io
import sys
import os
import requests
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

DATA_CSV = "comtrade_uae_from_china_8703_2023_2025.csv"

def fetch_comtrade_csv():
    # Prefer local CSV if user has downloaded it manually (avoid API 403)
    if os.path.exists(DATA_CSV):
        print(f"Found local file {DATA_CSV}, loading it instead of calling API.")
        with open(DATA_CSV, 'r', encoding='utf-8') as f:
            return f.read()

    url = (
        "https://comtrade.un.org/api/get?max=50000&type=C&freq=M&px=HS"
        "&ps=2023,2024,2025&r=784&p=156&rg=1&cc=8703&fmt=csv"
    )
    print("Requesting UN Comtrade...")
    resp = requests.get(url, timeout=60)
    if resp.status_code != 200:
        print(f"Warning: status code {resp.status_code}, saving response text to last_response.txt")
        with open("last_response.txt", "w", encoding="utf-8") as f:
            f.write(resp.text)
        resp.raise_for_status()
    return resp.text

def load_dataframe(csv_text):
    df = pd.read_csv(io.StringIO(csv_text))
    return df

def preprocess(df):
    # Try to locate period/year and quantity columns
    # Common UN Comtrade CSV columns: 'yr', 'period', 'TradeQuantity'
    if 'period' in df.columns:
        df['period'] = df['period'].astype(str)
    elif 'yr' in df.columns and 'period' not in df.columns:
        # construct period as YYYYMM if monthly available in 'yr' + 'period' pattern
        pass

    # Determine a quantity column
    qty_cols = [c for c in df.columns if 'TradeQuantity' in c or c.lower()=='tradequantity' or c.lower().endswith('quantity')]
    if not qty_cols:
        # fallback to 'TradeValue' (not ideal)
        val_cols = [c for c in df.columns if 'TradeValue' in c or c.lower()=='tradevalue']
        if not val_cols:
            raise RuntimeError('No quantity or value column found in UN Comtrade response. See saved last_response.txt')
        qty_col = val_cols[0]
        use_value = True
    else:
        qty_col = qty_cols[0]
        use_value = False

    # Build monthly date
    if 'period' in df.columns:
        # period format often YYYYMM
        df['period'] = df['period'].astype(str)
        # try to parse first 6 chars as YYYYMM
        df['ds'] = pd.to_datetime(df['period'].str.slice(0,6), format='%Y%m', errors='coerce')
    elif 'yr' in df.columns and 'period' in df.columns:
        df['ds'] = pd.to_datetime(df['yr'].astype(str) + df['period'].astype(str), format='%Y%m', errors='coerce')
    else:
        raise RuntimeError('Cannot find period/yr columns to create date index')

    df = df.dropna(subset=['ds'])

    df['y'] = pd.to_numeric(df[qty_col], errors='coerce')
    df = df.groupby('ds', as_index=False)['y'].sum().sort_values('ds')
    return df, use_value

def forecast(df):
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    m.fit(df.rename(columns={'ds':'ds','y':'y'}))
    future = m.make_future_dataframe(periods=6, freq='M')
    fcst = m.predict(future)
    return m, fcst

def plot_results(df, m, fcst):
    fig1 = m.plot(fcst)
    fig1.savefig('forecast_plot.png')
    plt.close(fig1)

    # Plot historical series
    plt.figure(figsize=(10,4))
    plt.plot(df['ds'], df['y'], marker='o')
    plt.title('Historical imports (units or value) — UAE from China (HS8703) 2023-2025')
    plt.xlabel('Date')
    plt.ylabel('Units / Value')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('historical_series.png')
    plt.close()

    print('Saved plots: historical_series.png, forecast_plot.png')

def main():
    try:
        csv_text = fetch_comtrade_csv()
        with open(DATA_CSV, 'w', encoding='utf-8') as f:
            f.write(csv_text)
        df_raw = load_dataframe(csv_text)
        print('Rows fetched:', len(df_raw))
        df, used_value = preprocess(df_raw)
        print('Prepared monthly series rows:', len(df))
        m, fcst = forecast(df)
        plot_results(df, m, fcst)
        # save forecast to csv
        fcst[['ds','yhat','yhat_lower','yhat_upper']].to_csv('forecast_6m.csv', index=False)
        print('Forecast saved to forecast_6m.csv')
    except Exception as e:
        print('Error:', e)
        sys.exit(1)

if __name__ == '__main__':
    main()
