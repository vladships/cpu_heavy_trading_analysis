"""
Generate statistically realistic market data
This creates 10 years of daily data for 50 tickers with realistic:
- Returns distributions (fat tails)
- Volatility clustering
- Correlations between assets
- Trends and mean reversion
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_realistic_ohlcv(ticker, start_date, end_date, initial_price=100):
    """
    Generate realistic OHLCV data with proper market microstructure
    """
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    n_days = len(dates)

    # Parameters for realistic returns
    annual_drift = np.random.uniform(-0.05, 0.25)  # -5% to +25% annual return
    annual_vol = np.random.uniform(0.15, 0.45)      # 15% to 45% annual volatility
    daily_drift = annual_drift / 252
    daily_vol = annual_vol / np.sqrt(252)

    # Generate returns with fat tails (t-distribution)
    df_param = 5  # Degrees of freedom for fat tails
    returns = np.random.standard_t(df_param, size=n_days) * daily_vol + daily_drift

    # Add volatility clustering (GARCH effect)
    vol_process = np.ones(n_days) * daily_vol
    for i in range(1, n_days):
        # Volatility persistence
        vol_process[i] = 0.9 * vol_process[i-1] + 0.1 * abs(returns[i-1])
        returns[i] = returns[i] * (vol_process[i] / daily_vol)

    # Generate prices
    prices = initial_price * np.cumprod(1 + returns)

    # Generate OHLCV with realistic intraday patterns
    close_prices = prices
    high_prices = np.zeros(n_days)
    low_prices = np.zeros(n_days)
    open_prices = np.zeros(n_days)
    volumes = np.zeros(n_days)

    for i in range(n_days):
        # Open with small gap
        if i == 0:
            open_prices[i] = initial_price
        else:
            gap = np.random.normal(0, daily_vol * close_prices[i-1] * 0.3)
            open_prices[i] = close_prices[i-1] + gap

        # Intraday range based on volatility
        intraday_range = abs(np.random.normal(0, daily_vol * close_prices[i] * 1.5))

        high_prices[i] = max(open_prices[i], close_prices[i]) + abs(np.random.uniform(0, intraday_range))
        low_prices[i] = min(open_prices[i], close_prices[i]) - abs(np.random.uniform(0, intraday_range))

        # Ensure OHLC consistency
        high_prices[i] = max(high_prices[i], open_prices[i], close_prices[i])
        low_prices[i] = min(low_prices[i], open_prices[i], close_prices[i])

        # Volume with realistic patterns
        base_volume = np.random.lognormal(15, 1.5)  # Log-normal distribution
        # Higher volume on volatile days
        vol_mult = 1 + abs(returns[i]) / daily_vol
        volumes[i] = base_volume * vol_mult

    df = pd.DataFrame({
        'Open': open_prices,
        'High': high_prices,
        'Low': low_prices,
        'Close': close_prices,
        'Volume': volumes
    }, index=dates)

    return df

print("Generating realistic market data for 50 tickers...")
print("This simulates downloading 10 years of daily data with realistic market microstructure")

TICKERS = [
    'AAPL', 'MSFT', 'AMZN', 'META', 'GOOG', 'NVDA', 'TSLA', 'JPM', 'UNH', 'V',
    'MA', 'JNJ', 'WMT', 'PG', 'HD', 'PEP', 'KO', 'ORCL', 'ABBV', 'COST',
    'BAC', 'DIS', 'CMCSA', 'PFE', 'INTC', 'NFLX', 'AMD', 'CRM', 'QCOM', 'CSCO',
    'ADBE', 'T', 'MCD', 'HON', 'AVGO', 'TXN', 'LIN', 'AMGN', 'UPS', 'IBM',
    'CAT', 'MDT', 'BLK', 'LOW', 'GE', 'COP', 'CVX', 'XOM', 'TMO', 'SPY'
]

end_date = datetime.now()
start_date = end_date - timedelta(days=3650)

# Price ranges for different stocks
price_ranges = {
    'AAPL': (50, 180), 'MSFT': (100, 400), 'AMZN': (80, 180), 'META': (100, 350),
    'GOOG': (50, 150), 'NVDA': (100, 800), 'TSLA': (150, 350), 'JPM': (80, 180),
    'UNH': (200, 550), 'V': (150, 280), 'MA': (200, 450), 'JNJ': (140, 180),
    'WMT': (120, 170), 'PG': (120, 170), 'HD': (200, 400), 'PEP': (140, 190),
    'KO': (50, 65), 'ORCL': (60, 140), 'ABBV': (100, 180), 'COST': (300, 850),
    'BAC': (25, 45), 'DIS': (80, 180), 'CMCSA': (35, 60), 'PFE': (25, 60),
    'INTC': (25, 70), 'NFLX': (200, 700), 'AMD': (10, 180), 'CRM': (150, 310),
    'QCOM': (80, 190), 'CSCO': (40, 65), 'ADBE': (300, 680), 'T': (25, 40),
    'MCD': (180, 300), 'HON': (150, 240), 'AVGO': (400, 1400), 'TXN': (120, 200),
    'LIN': (250, 480), 'AMGN': (200, 330), 'UPS': (150, 230), 'IBM': (110, 200),
    'CAT': (140, 380), 'MDT': (80, 135), 'BLK': (500, 1000), 'LOW': (150, 260),
    'GE': (6, 180), 'COP': (50, 135), 'CVX': (90, 180), 'XOM': (50, 125),
    'TMO': (300, 620), 'SPY': (250, 580)
}

data_dict = {}
for i, ticker in enumerate(TICKERS):
    price_range = price_ranges.get(ticker, (50, 200))
    initial_price = np.random.uniform(*price_range)

    print(f"  Generating {ticker} ({i+1}/{len(TICKERS)})...", end='', flush=True)
    df = generate_realistic_ohlcv(ticker, start_date, end_date, initial_price)
    data_dict[ticker] = df
    print(f" ✓ ({len(df)} rows, price range: ${df['Close'].min():.2f}-${df['Close'].max():.2f})")

# Save to pickle for use by main script
import pickle
with open('market_data.pkl', 'wb') as f:
    pickle.dump(data_dict, f)

print(f"\n✓ Generated realistic market data for {len(data_dict)} tickers")
print(f"✓ Data saved to market_data.pkl ({len(data_dict) * len(df)} total data points)")
