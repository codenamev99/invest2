# main.py
import os
import io
import time
import datetime
import logging
import threading
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ==============================
# Config
# ==============================
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "YOUR_ALPHA_VANTAGE_KEY_HERE")
ALPHAVANTAGE_BASE_URL = "https://www.alphavantage.co/query"
# Free tier ~5 req/min; pace calls across threads.
AV_RATE_LIMIT_SECONDS = 13

_av_lock = threading.Lock()
_last_av_call_ts = [0.0]


# ==============================
# Helpers
# ==============================
def map_symbols_for_providers(symbol: str):
    """
    Return (yahoo_symbol, alpha_vantage_symbol).
    Yahoo uses BRK-B/BF-B style, while Alpha Vantage typically accepts BRK.B/BF.B.
    """
    yahoo_symbol = symbol.replace(".", "-")
    av_symbol = symbol  # keep dot for AV
    return yahoo_symbol, av_symbol


# ==============================
# Indicator helpers
# ==============================
def get_atr(hist, period=14):
    high_low = hist['High'] - hist['Low']
    high_close = (hist['High'] - hist['Close'].shift()).abs()
    low_close = (hist['Low'] - hist['Close'].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(period).mean()
    return atr


def get_rsi_wilder_local(hist, period=14, use_adj_close=False, exclude_last=False):
    """Local Wilder RSI fallback."""
    px_col = 'Adj Close' if use_adj_close and 'Adj Close' in hist.columns else 'Close'
    data = hist if not exclude_last else hist.iloc[:-1]

    delta = data[px_col].diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)

    roll_up = up.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False, min_periods=period).mean()

    rs = roll_up / roll_down
    rsi = 100 - (100 / (1 + rs))
    return rsi


def get_rsi_alpha_vantage(symbol, period=14, interval="daily", series_type="close"):
    """
    Fetch RSI from Alpha Vantage Technical Indicators API.
    """
    if not ALPHAVANTAGE_API_KEY or ALPHAVANTAGE_API_KEY == "YOUR_ALPHA_VANTAGE_KEY_HERE":
        raise RuntimeError("Alpha Vantage API key not set. Set ALPHAVANTAGE_API_KEY or edit the script.")

    params = {
        "function": "RSI",
        "symbol": symbol,
        "interval": interval,        # "daily", "weekly", "monthly", or intraday like "60min"
        "time_period": period,       # default 14
        "series_type": series_type,  # "close", "open", "high", "low"
        "datatype": "json",
        "apikey": ALPHAVANTAGE_API_KEY,
    }

    with _av_lock:
        now = time.time()
        wait_for = AV_RATE_LIMIT_SECONDS - (now - _last_av_call_ts[0])
        if wait_for > 0:
            time.sleep(wait_for)
        resp = requests.get(ALPHAVANTAGE_BASE_URL, params=params, timeout=30)
        _last_av_call_ts[0] = time.time()

    if resp.status_code != 200:
        raise RuntimeError(f"Alpha Vantage HTTP {resp.status_code}")

    data = resp.json()
    if "Note" in data or "Information" in data:
        raise RuntimeError(data.get("Note") or data.get("Information"))
    if "Technical Analysis: RSI" not in data:
        raise RuntimeError(data.get("Error Message") or "Unknown AV response; no RSI data.")

    ta = data["Technical Analysis: RSI"]
    if not ta:
        raise RuntimeError("Alpha Vantage returned empty RSI series.")

    latest_ts = max(ta.keys())
    rsi_str = ta[latest_ts].get("RSI")
    if rsi_str is None:
        raise RuntimeError("Alpha Vantage RSI field missing for latest bar.")
    return float(rsi_str)


# ==============================
# Ticker universe
# ==============================
def fetch_all_tickers():
    logging.info("Fetching S&P 500 tickers from Wikipedia...")
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0"}
    html = requests.get(url, headers=headers).text
    # Wrap in StringIO to silence FutureWarning
    table = pd.read_html(io.StringIO(html))[0]
    tickers = table["Symbol"].tolist()
    logging.info(f"Fetched {len(tickers)} tickers from S&P 500.")
    return tickers


# ==============================
# Analyze a single ticker
# ==============================
def analyze_ticker(symbol, ignore_thresholds=False, av_interval="daily", av_period=14, av_series_type="close"):
    try:
        yf_symbol, av_symbol = map_symbols_for_providers(symbol)

        tk = yf.Ticker(yf_symbol)
        hist = tk.history(period="6mo", interval="1d", actions=False)

        if hist is None or len(hist) < 30:
            return None

        price = hist['Close'].iloc[-1]
        if price is None or price < 5:
            return None

        # ATR %
        atr = get_atr(hist).iloc[-1]
        atr_pct = (atr / price) * 100 if pd.notna(atr) else None

        # RSI from Alpha Vantage (primary) with local fallback
        try:
            rsi = get_rsi_alpha_vantage(av_symbol, period=av_period, interval=av_interval, series_type=av_series_type)
        except Exception as av_err:
            logging.debug(f"Alpha Vantage RSI failed for {symbol}: {av_err}. Falling back to local Wilder RSI.")
            rsi = get_rsi_wilder_local(hist, period=av_period, use_adj_close=False, exclude_last=False).iloc[-1]

        # MACD (local)
        ema12 = hist['Close'].ewm(span=12, adjust=False).mean()
        ema26 = hist['Close'].ewm(span=26, adjust=False).mean()
        macd_series = ema12 - ema26
        signal_series = macd_series.ewm(span=9, adjust=False).mean()
        macd_val = float(macd_series.iloc[-1])
        signal_val = float(signal_series.iloc[-1])
        macd_gt_signal = macd_val > signal_val  # used for filtering, not for output

        # Beta (optional; may be None)
        beta = tk.info.get('beta', None)

        # Average Daily Volume
        avg_daily_vol = hist['Volume'].rolling(20).mean().iloc[-1]
        avg_daily_vol_flag = bool(pd.notna(avg_daily_vol) and avg_daily_vol > 1_000_000)  # used for filtering

        # Apply thresholds if not ignoring
        if not ignore_thresholds:
            if (not macd_gt_signal) or (rsi is None or rsi < 50 or rsi > 70) or (atr_pct is None or atr_pct <= 2) \
               or (beta is None or beta <= 1.2) or (not avg_daily_vol_flag):
                return None

        # === Output dict with requested column changes ===
        return {
            "Ticker": symbol,                                  # original symbol for display
            "Price": round(float(price), 2),
            "ATR%": round(float(atr_pct), 2) if atr_pct is not None else None,
            "RSI": round(float(rsi), 2) if rsi is not None and not pd.isna(rsi) else None,
            # Print MACD and Signal values (string keeps CSV friendly formatting)
            "MACD>Signal": f"{macd_val:.4f} / {signal_val:.4f}",
            # Print numeric 20-day avg volume instead of boolean
            "AvgDailyVol>1M": int(avg_daily_vol) if pd.notna(avg_daily_vol) else None,
            "Beta": round(float(beta), 2) if beta else None,
        }

    except Exception as e:
        logging.debug(f"Ticker {symbol} error: {e}")
    return None


# ==============================
# Run screener
# ==============================
def run_screener(tickers):
    results = []
    max_workers = 6
    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        futures = {exe.submit(analyze_ticker, t): t for t in tickers}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Scanning"):
            symbol = futures[fut]
            try:
                res = fut.result()
                if res:
                    results.append(res)
            except Exception as e:
                logging.debug(f"Error for {symbol}: {e}")
            time.sleep(0.01)  # polite delay for yfinance (not AV)

    df = pd.DataFrame(results)

    if df.empty:
        logging.info("No candidates matched filters this run.")
        return df

    if "ATR%" in df.columns:
        df = df.sort_values(by="ATR%", ascending=False, na_position="last")

    # Save results
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"candidates_{ts}.csv"
    with open(out_path, 'w') as f:
        f.write(f"# Screener Parameters:\n")
        f.write(f"# MACD > Signal (values printed as 'MACD / Signal')\n")
        f.write(f"# RSI(14) from Alpha Vantage (daily, close)\n")
        f.write(f"# ATR% > 2\n")
        f.write(f"# Beta > 1.2\n")
        f.write(f"# AvgDailyVol > 1M (value printed)\n\n")
    df.to_csv(out_path, mode='a', index=False)
    logging.info(f"Found {len(df)} candidates. Saved to {out_path}")
    return df


# ==============================
# Main
# ==============================
if __name__ == "__main__":
    print("Select mode:")
    print("1 - Scan all S&P 500 tickers")
    print("2 - Scan a specific ticker")
    choice = input("Enter 1 or 2: ").strip()

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    if choice == "1":
        tickers = fetch_all_tickers()
        all_tickers_path = f"all_scanned_tickers_{ts}.csv"
        pd.DataFrame({"Ticker": tickers}).to_csv(all_tickers_path, index=False)
        logging.info(f"Saved all scanned tickers to {all_tickers_path}")
        df = run_screener(tickers)

    elif choice == "2":
        ticker_input = input("Enter the ticker symbol (e.g., AAPL): ").strip().upper()
        res = analyze_ticker(ticker_input, ignore_thresholds=True)  # show all metrics
        if res:
            df = pd.DataFrame([res])
            out_path = f"candidates_{ticker_input}_{ts}.csv"
            with open(out_path, 'w') as f:
                f.write(f"# Screener Parameters:\n")
                f.write(f"# MACD > Signal (values printed as 'MACD / Signal')\n")
                f.write(f"# RSI(14) from Alpha Vantage (daily, close)\n")
                f.write(f"# ATR% > 2\n")
                f.write(f"# Beta > 1.2\n")
                f.write(f"# AvgDailyVol > 1M (value printed)\n\n")
            df.to_csv(out_path, mode='a', index=False)
            logging.info(f"Saved single ticker result to {out_path}")
        else:
            logging.info(f"Could not fetch metrics for {ticker_input}.")
            df = pd.DataFrame()
    else:
        print("Invalid choice. Exiting.")
        exit()

    if not df.empty:
        print(df.head(30))
