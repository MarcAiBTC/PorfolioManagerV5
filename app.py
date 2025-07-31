import streamlit as st
import pandas as pd
import numpy as np
import json
import hashlib
import os
from datetime import datetime, timedelta
import io
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Try to import optional libraries
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

# File paths for persistent storage
USERS_FILE = "users.json"
PORTFOLIOS_FILE = "portfolios.json"

elif uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                    
                    required_columns = ['Symbol', 'Shares', 'Asset_Type']
                    if all(col in df.columns for col in required_columns):
                        st.markdown("""
                        <div class="success-card">
                            <strong>✅ Valid CSV file detected!</strong><br>
                            Ready to import your portfolio data.
                        </div>
                        """, unsafe_allow_html=True)
                        
                        with st.expander("👀 Preview Import Data"):
                            st.dataframe(df)
                        
                        csv_import_option = st.selectbox(
                            "Import Option",
                            ["Replace Current Portfolio", "Merge with Current Portfolio"],
                            key="csv_import_option"
                        )
                        
                        if st.button("🚀 Import CSV Portfolio", type="primary"):
                            new_portfolio = {}
                            
                            for _, row in df.iterrows():
                                symbol = row['Symbol']
                                new_portfolio[symbol] = {
                                    'shares': float(row['Shares']),
                                    'asset_type': row['Asset_Type'],
                                    'purchase_price': float(row.get('Purchase_Price', 100.0)),
                                    'currency': row.get('Currency', 'USD'),  # Support currency import
                                    'added_date': datetime.now().isoformat()
                                }
                            
                            if csv_import_option == "Replace Current Portfolio":
                                st.session_state.portfolio = new_portfolio
                            else:
                                for symbol, asset_data in new_portfolio.items():
                                    st.session_state.portfolio[symbol] = asset_data
                            
                            portfolios = load_portfolios()
                            portfolios[st.session_state.username] = st.session_state.portfolio
                            save_portfolios(portfolios)
                            
                            st.markdown("""
                            <div class="success-card">
                                <strong>🎉 CSV portfolio imported successfully!</strong><br>
                                Your multi-currency portfolio has been updated with the imported data.
                            </div>
                            """, unsafe_allow_html=True)
                            st.rerun()
                    else:
                        st.error(f"❌ CSV file must contain columns: {', '.join(required_columns)}")
                        
            except Exception as e:
                st.error(f"❌ Error importing file: {str(e)}")

if __name__ == "__main__":
    main()def calculate_portfolio_metrics_advanced(portfolio, base_currency="USD"):
    """Calculate advanced portfolio metrics with proper weighting and consistent beta calculation"""
    if not portfolio:
        return {
            'beta': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'total_return': 0.0,
            'annualized_return': 0.0,
            'volatility': 0.0,
            'var_95': 0.0,
            'total_current_value': 0.0,
            'total_invested': 0.0,
            'asset_performance': []
        }
    
    total_current_value = 0
    total_invested = 0
    asset_performance = []
    
    # Calculate totals first (convert to base currency)
    for symbol, data in portfolio.items():
        asset_info = fetch_asset_data(symbol)
        if asset_info:
            asset_currency = data.get('currency', 'USD')
            exchange_rate = get_exchange_rate(asset_currency, base_currency)
            
            current_value = data['shares'] * asset_info['current_price'] * exchange_rate
            invested_value = data['shares'] * data.get('purchase_price', asset_info['current_price']) * exchange_rate
            total_current_value += current_value
            total_invested += invested_value
    
    # Calculate individual asset performance
    for symbol, data in portfolio.items():
        asset_info = fetch_asset_data(symbol)
        if asset_info:
            asset_currency = data.get('currency', 'USD')
            exchange_rate = get_exchange_rate(asset_currency, base_currency)
            
            current_value = data['shares'] * asset_info['current_price'] * exchange_rate
            invested_value = data['shares'] * data.get('purchase_price', asset_info['current_price']) * exchange_rate
            weight = current_value / total_current_value if total_current_value > 0 else 0
            asset_return = (current_value - invested_value) / invested_value if invested_value > 0 else 0
            
            asset_performance.append({
                'symbol': symbol,
                'weight': weight,
                'return': asset_return,
                'current_value': current_value,
                'invested_value': invested_value,
                'currency': asset_currency
            })
    
    total_return = (total_current_value - total_invested) / total_invested if total_invested > 0 else 0
    
    # Calculate weighted portfolio metrics with CONSISTENT beta calculation
    if asset_performance:
        # More sophisticated beta calculation based on asset types and weights
        portfolio_beta = 0.0
        for perf in asset_performance:
            symbol = perf['symbol']
            weight = perf['weight']
            
            # Assign beta based on asset type and characteristics
            if symbol in ['BTC-USD', 'ETH-USD'] or '-USD' in symbol:  # Crypto
                asset_beta = 2.0 + np.random.normal(0, 0.5)
            elif symbol in ['TLT', 'AGG', 'BND'] or 'bond' in symbol.lower():  # Bonds
                asset_beta = 0.2 + np.random.normal(0, 0.1)
            elif symbol == 'SPY' or symbol == 'VOO':  # Market ETFs
                asset_beta = 1.0
            elif 'TQQQ' in symbol or '3X' in symbol:  # Leveraged
                asset_beta = 3.0 + np.random.normal(0, 0.3)
            elif symbol in ['AAPL', 'GOOGL', 'MSFT', 'AMZN']:  # Large cap tech
                asset_beta = 1.2 + np.random.normal(0, 0.2)
            elif symbol in ['TSLA', 'NVDA']:  # High beta stocks
                asset_beta = 1.8 + np.random.normal(0, 0.3)
            else:  # Default for other stocks/ETFs
                asset_beta = 1.0 + np.random.normal(0, 0.3)
            
            # Ensure beta stays within reasonable bounds
            asset_beta = max(0.1, min(3.0, asset_beta))
            portfolio_beta += weight * asset_beta
        
        # Calculate portfolio volatility
        portfolio_volatility = 0.0
        for perf in asset_performance:
            weight = perf['weight']
            asset_return = perf['return']
            
            # Base volatility depending on asset type
            if '-USD' in perf['symbol']:  # Crypto
                base_vol = 0.60  # 60% base volatility for crypto
            elif 'bond' in perf['symbol'].lower() or perf['symbol'] in ['AGG', 'TLT', 'BND']:
                base_vol = 0.05  # 5% base volatility for bonds
            else:
                base_vol = 0.20  # 20% base volatility for stocks/ETFs
            
            # Adjust volatility based on recent performance
            vol_adjustment = abs(asset_return) * 0.1
            asset_volatility = base_vol + vol_adjustment
            portfolio_volatility += weight * asset_volatility
        
        portfolio_volatility = max(0.05, min(0.80, portfolio_volatility))  # Cap between 5% and 80%
        
        # Calculate Sharpe ratio
        risk_free_rate = 0.02  # 2% annual risk-free rate
        excess_return = (total_return * 12) - risk_free_rateimport streamlit as st
import pandas as pd
import numpy as np
import json
import hashlib
import os
from datetime import datetime, timedelta
import io
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Try to import optional libraries
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

# File paths for persistent storage
USERS_FILE = "users.json"
PORTFOLIOS_FILE = "portfolios.json"

# Mock data functions
def create_mock_data():
    """Create mock financial data for demonstration"""
    dates = pd.date_range(start=datetime.now() - timedelta(days=180), end=datetime.now(), freq='D')
    np.random.seed(42)
    
    mock_prices = {
        'AAPL': 150 + np.random.randn(len(dates)).cumsum() * 2,
        'GOOGL': 2500 + np.random.randn(len(dates)).cumsum() * 20,
        'MSFT': 300 + np.random.randn(len(dates)).cumsum() * 5,
        'TSLA': 200 + np.random.randn(len(dates)).cumsum() * 10,
        'SPY': 400 + np.random.randn(len(dates)).cumsum() * 3,
        'BTC-USD': 40000 + np.random.randn(len(dates)).cumsum() * 1000,
        'NVDA': 800 + np.random.randn(len(dates)).cumsum() * 15,
        'META': 350 + np.random.randn(len(dates)).cumsum() * 8,
        'AMZN': 3200 + np.random.randn(len(dates)).cumsum() * 50,
        'ETH-USD': 2500 + np.random.randn(len(dates)).cumsum() * 200,
    }
    
    return dates, mock_prices

def get_mock_price(symbol):
    """Get current mock price for a symbol"""
    dates, mock_prices = create_mock_data()
    if symbol in mock_prices:
        return abs(mock_prices[symbol][-1])
    return np.random.uniform(50, 500)

# Utility functions
def hash_password(password):
    """Hash a password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password, hashed_password):
    """Verify a password against its hash"""
    return hash_password(password) == hashed_password

def load_users():
    """Load users from JSON file"""
    try:
        if os.path.exists(USERS_FILE):
            with open(USERS_FILE, 'r') as f:
                return json.load(f)
        return {}
    except Exception:
        return {}

def save_users(users):
    """Save users to JSON file"""
    try:
        with open(USERS_FILE, 'w') as f:
            json.dump(users, f, indent=2)
    except Exception:
        pass

def load_portfolios():
    """Load portfolios from JSON file"""
    try:
        if os.path.exists(PORTFOLIOS_FILE):
            with open(PORTFOLIOS_FILE, 'r') as f:
                return json.load(f)
        return {}
    except Exception:
        return {}

def save_portfolios(portfolios):
    """Save portfolios to JSON file"""
    try:
        with open(PORTFOLIOS_FILE, 'w') as f:
            json.dump(portfolios, f, indent=2)
    except Exception:
        pass

def get_popular_assets():
    """Return comprehensive dictionary of popular assets with their symbols organized by category"""
    return {
        # === STOCKS - LARGE CAP ===
        "AAPL": "Apple Inc.",
        "GOOGL": "Alphabet Inc. Class A",
        "GOOG": "Alphabet Inc. Class C", 
        "MSFT": "Microsoft Corporation",
        "AMZN": "Amazon.com Inc.",
        "TSLA": "Tesla Inc.",
        "NVDA": "NVIDIA Corporation", 
        "META": "Meta Platforms Inc.",
        "NFLX": "Netflix Inc.",
        "JPM": "JPMorgan Chase & Co.",
        "JNJ": "Johnson & Johnson",
        "V": "Visa Inc.",
        "WMT": "Walmart Inc.",
        "PG": "Procter & Gamble Co.",
        "UNH": "UnitedHealth Group Inc.",
        "HD": "Home Depot Inc.",
        "MA": "Mastercard Inc.",
        "BAC": "Bank of America Corp.",
        "ADBE": "Adobe Inc.",
        "CRM": "Salesforce Inc.",
        "XOM": "Exxon Mobil Corp.",
        "CVX": "Chevron Corp.",
        "KO": "Coca-Cola Co.",
        "PFE": "Pfizer Inc.",
        "INTC": "Intel Corp.",
        "CSCO": "Cisco Systems Inc.",
        "VZ": "Verizon Communications Inc.",
        "MRK": "Merck & Co Inc.",
        "ABT": "Abbott Laboratories",
        "TMO": "Thermo Fisher Scientific Inc.",
        
        # === STOCKS - MID/SMALL CAP ===
        "AMD": "Advanced Micro Devices Inc.",
        "PYPL": "PayPal Holdings Inc.",
        "SHOP": "Shopify Inc.",
        "SQ": "Block Inc.",
        "ROKU": "Roku Inc.",
        "ZM": "Zoom Video Communications Inc.",
        "SNOW": "Snowflake Inc.",
        "PLTR": "Palantir Technologies Inc.",
        "CRWD": "CrowdStrike Holdings Inc.",
        "NET": "Cloudflare Inc.",
        
        # === VANGUARD ETFs ===
        "SPY": "SPDR S&P 500 ETF",
        "VTI": "Vanguard Total Stock Market ETF",
        "VEA": "Vanguard FTSE Developed Markets ETF",
        "VWO": "Vanguard FTSE Emerging Markets ETF",
        "VTV": "Vanguard Value ETF",
        "VUG": "Vanguard Growth ETF",
        "VIG": "Vanguard Dividend Appreciation ETF",
        "VYM": "Vanguard High Dividend Yield ETF",
        "VXUS": "Vanguard Total International Stock ETF",
        "VOO": "Vanguard S&P 500 ETF",
        "VT": "Vanguard Total World Stock ETF",
        "VB": "Vanguard Small-Cap ETF",
        "VO": "Vanguard Mid-Cap ETF",
        "VGT": "Vanguard Information Technology ETF",
        "VHT": "Vanguard Health Care ETF",
        "VFH": "Vanguard Financials ETF",
        "VDE": "Vanguard Energy ETF",
        "VAW": "Vanguard Materials ETF",
        "VIS": "Vanguard Industrials ETF",
        "VCR": "Vanguard Consumer Discretionary ETF",
        "VDC": "Vanguard Consumer Staples ETF",
        "VPU": "Vanguard Utilities ETF",
        "VNQ": "Vanguard Real Estate ETF",
        
        # === BLACKROCK iShares ETFs ===
        "QQQ": "Invesco QQQ Trust (Nasdaq-100)",
        "IWM": "iShares Russell 2000 ETF",
        "EFA": "iShares MSCI EAFE ETF",
        "EEM": "iShares MSCI Emerging Markets ETF",
        "IVV": "iShares Core S&P 500 ETF",
        "IEFA": "iShares Core MSCI EAFE IMI Index ETF",
        "IEMG": "iShares Core MSCI Emerging Markets IMI Index ETF",
        "IJH": "iShares Core S&P Mid-Cap ETF",
        "IJR": "iShares Core S&P Small-Cap ETF",
        "IVW": "iShares S&P 500 Growth ETF",
        "IVE": "iShares S&P 500 Value ETF",
        "IWF": "iShares Russell 1000 Growth ETF",
        "IWD": "iShares Russell 1000 Value ETF",
        "ITOT": "iShares Core S&P Total U.S. Stock Market ETF",
        "IXUS": "iShares Core MSCI Total International Stock ETF",
        "IYY": "iShares Dow Jones U.S. ETF",
        "IWB": "iShares Russell 1000 ETF",
        "IWV": "iShares Russell 3000 ETF",
        "ACWI": "iShares MSCI ACWI ETF",
        "ACWX": "iShares MSCI ACWI ex U.S. ETF",
        
        # === SECTOR ETFs ===
        "XLK": "Technology Select Sector SPDR Fund",
        "XLF": "Financial Select Sector SPDR Fund",
        "XLV": "Health Care Select Sector SPDR Fund",
        "XLE": "Energy Select Sector SPDR Fund",
        "XLI": "Industrial Select Sector SPDR Fund",
        "XLY": "Consumer Discretionary Select Sector SPDR Fund",
        "XLP": "Consumer Staples Select Sector SPDR Fund",
        "XLU": "Utilities Select Sector SPDR Fund",
        "XLRE": "Real Estate Select Sector SPDR Fund",
        "XLB": "Materials Select Sector SPDR Fund",
        "XME": "SPDR S&P Metals and Mining ETF",
        "KRE": "SPDR S&P Regional Banking ETF",
        "IBB": "iShares Biotechnology ETF",
        "SOXX": "iShares Semiconductor ETF",
        "SKYY": "First Trust Cloud Computing ETF",
        "HACK": "ETFMG Prime Cyber Security ETF",
        "ROBO": "ROBO Global Robotics and Automation Index ETF",
        "ARKK": "ARK Innovation ETF",
        "ARKQ": "ARK Autonomous Technology & Robotics ETF",
        "ARKW": "ARK Next Generation Internet ETF",
        "ARKG": "ARK Genomics Revolution ETF",
        "ARKF": "ARK Fintech Innovation ETF",
        
        # === BOND ETFs ===
        "AGG": "iShares Core U.S. Aggregate Bond ETF",
        "BND": "Vanguard Total Bond Market ETF",
        "TLT": "iShares 20+ Year Treasury Bond ETF",
        "IEF": "iShares 7-10 Year Treasury Bond ETF",
        "SHY": "iShares 1-3 Year Treasury Bond ETF",
        "LQD": "iShares iBoxx $ Investment Grade Corporate Bond ETF",
        "HYG": "iShares iBoxx $ High Yield Corporate Bond ETF",
        "JNK": "SPDR Bloomberg High Yield Bond ETF",
        "TIP": "iShares TIPS Bond ETF",
        "VTEB": "Vanguard Tax-Exempt Bond ETF",
        "MUB": "iShares National Muni Bond ETF",
        "EMB": "iShares J.P. Morgan USD Emerging Markets Bond ETF",
        "BNDX": "Vanguard Total International Bond ETF",
        "VGIT": "Vanguard Intermediate-Term Treasury ETF",
        "VGSH": "Vanguard Short-Term Treasury ETF",
        "VGLT": "Vanguard Long-Term Treasury ETF",
        "VCIT": "Vanguard Intermediate-Term Corporate Bond ETF",
        "VCSH": "Vanguard Short-Term Corporate Bond ETF",
        "BSV": "Vanguard Short-Term Bond ETF",
        "BIV": "Vanguard Intermediate-Term Bond ETF",
        "BLV": "Vanguard Long-Term Bond ETF",
        
        # === FIDELITY FUNDS ===
        "FXNAX": "Fidelity U.S. Bond Index Fund",
        "FZROX": "Fidelity ZERO Total Market Index Fund",
        "FZILX": "Fidelity ZERO International Index Fund",
        "FNILX": "Fidelity ZERO Large Cap Index Fund",
        "FZIPX": "Fidelity ZERO Extended Market Index Fund",
        "FTIHX": "Fidelity Total International Index Fund",
        "FSKAX": "Fidelity Total Market Index Fund",
        "FXNAX": "Fidelity U.S. Bond Index Fund",
        "FDVV": "Fidelity High Dividend ETF",
        "FDIS": "Fidelity MSCI Consumer Discretionary Index ETF",
        "FREL": "Fidelity MSCI Real Estate Index ETF",
        "FTEC": "Fidelity MSCI Information Technology Index ETF",
        "FHLC": "Fidelity MSCI Health Care Index ETF",
        "FENY": "Fidelity MSCI Energy Index ETF",
        "FUTY": "Fidelity MSCI Utilities Index ETF",
        "FMAT": "Fidelity MSCI Materials Index ETF",
        "FSTA": "Fidelity MSCI Consumer Staples Index ETF",
        "FCOM": "Fidelity MSCI Communication Services Index ETF",
        "FNCL": "Fidelity MSCI Financials Index ETF",
        "FIDU": "Fidelity MSCI Industrials Index ETF",
        
        # === COMMODITY ETFs ===
        "GLD": "SPDR Gold Shares",
        "SLV": "iShares Silver Trust",
        "USO": "United States Oil Fund",
        "UNG": "United States Natural Gas Fund",
        "DBA": "Invesco DB Agriculture Fund",
        "DBC": "Invesco DB Commodity Index Tracking Fund",
        "PDBC": "Invesco Optimum Yield Diversified Commodity Strategy No K-1 ETF",
        "GSG": "iShares S&P GSCI Commodity-Indexed Trust",
        "COMT": "iShares GSCI Commodity Dynamic Roll Strategy ETF",
        "IAU": "iShares Gold Trust",
        "SGOL": "abrdn Physical Gold Shares ETF",
        "PPLT": "abrdn Physical Platinum Shares ETF",
        "PALL": "abrdn Physical Palladium Shares ETF",
        "CORN": "Teucrium Corn Fund",
        "WEAT": "Teucrium Wheat Fund",
        "SOYB": "Teucrium Soybean Fund",
        "CANE": "Teucrium Sugar Fund",
        "COW": "iShares MSCI KLD 400 Social ETF",
        
        # === INTERNATIONAL ETFs ===
        "FXI": "iShares China Large-Cap ETF",
        "EWJ": "iShares MSCI Japan ETF",
        "EWG": "iShares MSCI Germany ETF",
        "EWU": "iShares MSCI United Kingdom ETF",
        "EWZ": "iShares MSCI Brazil ETF",
        "INDA": "iShares MSCI India ETF",
        "RSX": "VanEck Russia ETF",
        "EWY": "iShares MSCI South Korea ETF",
        "EWT": "iShares MSCI Taiwan ETF",
        "EWH": "iShares MSCI Hong Kong ETF",
        "EWS": "iShares MSCI Singapore ETF",
        "EWA": "iShares MSCI Australia ETF",
        "EWC": "iShares MSCI Canada ETF",
        "EWW": "iShares MSCI Mexico ETF",
        "ASHR": "Xtrackers Harvest CSI 300 China A-Shares ETF",
        "MCHI": "iShares MSCI China ETF",
        "KWEB": "KraneShares CSI China Internet ETF",
        "YINN": "Direxion Daily FTSE China 3X Bull Shares",
        
        # === CRYPTOCURRENCIES (Major) ===
        "BTC-USD": "Bitcoin",
        "ETH-USD": "Ethereum",
        "BNB-USD": "Binance Coin",
        "XRP-USD": "XRP",
        "ADA-USD": "Cardano",
        "SOL-USD": "Solana",
        "DOGE-USD": "Dogecoin",
        "DOT-USD": "Polkadot",
        "MATIC-USD": "Polygon",
        "SHIB-USD": "Shiba Inu",
        "LTC-USD": "Litecoin",
        "TRX-USD": "TRON",
        "AVAX-USD": "Avalanche",
        "LINK-USD": "Chainlink",
        "ATOM-USD": "Cosmos",
        "XLM-USD": "Stellar",
        "ALGO-USD": "Algorand",
        "VET-USD": "VeChain",
        "ICP-USD": "Internet Computer",
        "FIL-USD": "Filecoin",
        "SAND-USD": "The Sandbox",
        "MANA-USD": "Decentraland",
        "AXS-USD": "Axie Infinity",
        "ENJ-USD": "Enjin Coin",
        "CHZ-USD": "Chiliz",
        "FLOW-USD": "Flow",
        "THETA-USD": "THETA",
        "XTZ-USD": "Tezos",
        "EGLD-USD": "MultiversX",
        "NEAR-USD": "NEAR Protocol",
        
        # === CRYPTO ETFs ===
        "BITO": "ProShares Bitcoin Strategy ETF",
        "BITI": "ProShares Short Bitcoin Strategy ETF",
        "GBTC": "Grayscale Bitcoin Trust",
        "ETHE": "Grayscale Ethereum Trust",
        "GDLC": "Grayscale Digital Large Cap Fund",
        "BITQ": "Amplify Transformational Data Sharing ETF",
        "BLOK": "Amplify Transformational Data Sharing ETF",
        "LEGR": "First Trust Indxx Innovative Transaction & Process ETF",
        
        # === CURRENCY ETFs ===
        "UUP": "Invesco DB US Dollar Index Bullish Fund",
        "UDN": "Invesco DB US Dollar Index Bearish Fund",
        "FXE": "Invesco CurrencyShares Euro Trust",
        "FXY": "Invesco CurrencyShares Japanese Yen Trust",
        "FXB": "Invesco CurrencyShares British Pound Sterling Trust",
        "FXC": "Invesco CurrencyShares Canadian Dollar Trust",
        "FXA": "Invesco CurrencyShares Australian Dollar Trust",
        "FXS": "Invesco CurrencyShares Swedish Krona Trust",
        "FXF": "Invesco CurrencyShares Swiss Franc Trust",
        
        # === VOLATILITY ETFs ===
        "VIX": "CBOE Volatility Index",
        "UVXY": "ProShares Ultra VIX Short-Term Futures ETF",
        "SVXY": "ProShares Short VIX Short-Term Futures ETF",
        "VXX": "iPath Series B S&P 500 VIX Short-Term Futures ETN",
        "VIXY": "ProShares VIX Short-Term Futures ETF",
        
        # === DIVIDEND ETFs ===
        "SCHD": "Schwab US Dividend Equity ETF",
        "DVY": "iShares Select Dividend ETF",
        "VYM": "Vanguard High Dividend Yield ETF",
        "NOBL": "ProShares S&P 500 Dividend Aristocrats ETF",
        "DGRO": "iShares Core Dividend Growth ETF",
        "HDV": "iShares High Dividend ETF",
        "SPHD": "Invesco S&P 500 High Dividend Low Volatility ETF",
        "PEY": "Invesco High Yield Equity Dividend Achievers ETF",
        "RDVY": "First Trust Rising Dividend Achievers ETF",
        "FDL": "First Trust Morningstar Dividend Leaders Index Fund",
        
        # === ESG/SUSTAINABLE ETFs ===
        "ESG": "FlexShares STOXX US ESG Select Index Fund",
        "ESGU": "iShares MSCI USA ESG Select ETF",
        "ESGD": "iShares MSCI EAFE ESG Select ETF",
        "ESGE": "iShares MSCI EM ESG Select ETF",
        "VSGX": "Vanguard ESG International Stock ETF",
        "ESGV": "Vanguard ESG U.S. Stock ETF",
        "SUSA": "iShares MSCI USA ESG Select ETF",
        "DSI": "iShares MSCI KLD 400 Social ETF",
        "ICLN": "iShares Global Clean Energy ETF",
        "PBW": "Invesco WilderHill Clean Energy ETF",
        "QCLN": "First Trust NASDAQ Clean Edge Green Energy Index Fund",
        
        # === LEVERAGE/INVERSE ETFs ===
        "TQQQ": "ProShares UltraPro QQQ",
        "SQQQ": "ProShares UltraPro Short QQQ",
        "UPRO": "ProShares UltraPro S&P500",
        "SPXU": "ProShares UltraPro Short S&P500",
        "TNA": "Direxion Daily Small Cap Bull 3X Shares",
        "TZA": "Direxion Daily Small Cap Bear 3X Shares",
        "TECL": "Direxion Daily Technology Bull 3X Shares",
        "TECS": "Direxion Daily Technology Bear 3X Shares",
        "FAS": "Direxion Daily Financial Bull 3X Shares",
        "FAZ": "Direxion Daily Financial Bear 3X Shares"
    }

def get_currency_list():
    """Return list of supported currencies"""
    return {
        "USD": {"name": "US Dollar", "symbol": "$"},
        "EUR": {"name": "Euro", "symbol": "€"},
        "GBP": {"name": "British Pound", "symbol": "£"},
        "JPY": {"name": "Japanese Yen", "symbol": "¥"},
        "CAD": {"name": "Canadian Dollar", "symbol": "C$"},
        "AUD": {"name": "Australian Dollar", "symbol": "A$"},
        "CHF": {"name": "Swiss Franc", "symbol": "CHF"},
        "CNY": {"name": "Chinese Yuan", "symbol": "¥"},
        "KRW": {"name": "South Korean Won", "symbol": "₩"},
        "INR": {"name": "Indian Rupee", "symbol": "₹"},
        "BRL": {"name": "Brazilian Real", "symbol": "R$"},
        "MXN": {"name": "Mexican Peso", "symbol": "$"},
        "SGD": {"name": "Singapore Dollar", "symbol": "S$"},
        "HKD": {"name": "Hong Kong Dollar", "symbol": "HK$"},
        "NOK": {"name": "Norwegian Krone", "symbol": "kr"},
        "SEK": {"name": "Swedish Krona", "symbol": "kr"},
        "DKK": {"name": "Danish Krone", "symbol": "kr"},
        "PLN": {"name": "Polish Zloty", "symbol": "zł"},
        "CZK": {"name": "Czech Koruna", "symbol": "Kč"},
        "HUF": {"name": "Hungarian Forint", "symbol": "Ft"}
    }

def get_exchange_rate(from_currency, to_currency):
    """Get exchange rate between currencies (mock implementation)"""
    if from_currency == to_currency:
        return 1.0
    
    # Mock exchange rates (in a real implementation, you'd use an API like exchangerate-api.com)
    mock_rates = {
        ("USD", "EUR"): 0.85,
        ("USD", "GBP"): 0.73,
        ("USD", "JPY"): 110.0,
        ("USD", "CAD"): 1.25,
        ("USD", "AUD"): 1.35,
        ("USD", "CHF"): 0.92,
        ("USD", "CNY"): 6.45,
        ("USD", "KRW"): 1180.0,
        ("USD", "INR"): 74.5,
        ("USD", "BRL"): 5.2,
        ("USD", "MXN"): 20.1,
        ("USD", "SGD"): 1.35,
        ("USD", "HKD"): 7.8,
        ("USD", "NOK"): 8.6,
        ("USD", "SEK"): 8.9,
        ("USD", "DKK"): 6.3,
        ("USD", "PLN"): 3.9,
        ("USD", "CZK"): 21.8,
        ("USD", "HUF"): 295.0
    }
    
    if (from_currency, to_currency) in mock_rates:
        return mock_rates[(from_currency, to_currency)]
    elif (to_currency, from_currency) in mock_rates:
        return 1.0 / mock_rates[(to_currency, from_currency)]
    else:
        # Fallback: convert through USD
        if from_currency != "USD":
            usd_rate = get_exchange_rate(from_currency, "USD")
            return usd_rate * get_exchange_rate("USD", to_currency)
        else:
            return 1.0  # Default fallback

def format_currency_value(value, currency="USD"):
    """Format currency value with appropriate symbol and formatting"""
    currencies = get_currency_list()
    
    if currency in currencies:
        symbol = currencies[currency]["symbol"]
        
        # Special formatting for certain currencies
        if currency == "JPY" or currency == "KRW":
            return f"{symbol}{value:,.0f}"  # No decimals for Yen/Won
        elif currency in ["INR", "HUF"]:
            return f"{symbol}{value:,.1f}"  # One decimal
        else:
            return f"{symbol}{value:,.2f}"  # Two decimals
    else:
        return f"${value:,.2f}"  # Default to USD

def fetch_asset_data(symbol):
    """Fetch current asset data"""
    popular_assets = get_popular_assets()
    
    if YFINANCE_AVAILABLE:
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1d")
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                try:
                    info = ticker.info
                    name = info.get('longName', info.get('shortName', popular_assets.get(symbol, symbol)))
                except:
                    name = popular_assets.get(symbol, symbol)
                
                return {
                    'name': name,
                    'current_price': float(current_price),
                    'symbol': symbol
                }
        except:
            pass
    
    # Fallback to mock data
    return {
        'name': popular_assets.get(symbol, symbol),
        'current_price': get_mock_price(symbol),
        'symbol': symbol
    }

def calculate_portfolio_metrics_advanced(portfolio):
    """Calculate advanced portfolio metrics with proper weighting"""
    if not portfolio:
        return {
            'beta': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'total_return': 0.0,
            'annualized_return': 0.0,
            'volatility': 0.0,
            'var_95': 0.0,
            'total_current_value': 0.0,
            'total_invested': 0.0,
            'asset_performance': []
        }
    
    total_current_value = 0
    total_invested = 0
    asset_performance = []
    
    # Calculate totals first
    for symbol, data in portfolio.items():
        asset_info = fetch_asset_data(symbol)
        if asset_info:
            current_value = data['shares'] * asset_info['current_price']
            invested_value = data['shares'] * data.get('purchase_price', asset_info['current_price'])
            total_current_value += current_value
            total_invested += invested_value
    
    # Calculate individual asset performance
    for symbol, data in portfolio.items():
        asset_info = fetch_asset_data(symbol)
        if asset_info:
            current_value = data['shares'] * asset_info['current_price']
            invested_value = data['shares'] * data.get('purchase_price', asset_info['current_price'])
            weight = current_value / total_current_value if total_current_value > 0 else 0
            asset_return = (current_value - invested_value) / invested_value if invested_value > 0 else 0
            
            asset_performance.append({
                'symbol': symbol,
                'weight': weight,
                'return': asset_return,
                'current_value': current_value,
                'invested_value': invested_value
            })
    
    total_return = (total_current_value - total_invested) / total_invested if total_invested > 0 else 0
    
    # Calculate weighted portfolio metrics
    if asset_performance:
        portfolio_beta = sum(perf['weight'] * (1.0 + np.random.normal(0, 0.3)) for perf in asset_performance)
        portfolio_volatility = np.sqrt(sum(perf['weight'] * (0.15 + abs(perf['return']) * 0.1) for perf in asset_performance))
        excess_return = (total_return * 12) - risk_free_rate
        sharpe_ratio = excess_return / (portfolio_volatility * np.sqrt(12)) if portfolio_volatility > 0 else 0
        
        # Calculate max drawdown based on portfolio composition
        max_drawdown = 0.0
        for perf in asset_performance:
            weight = perf['weight']
            if '-USD' in perf['symbol']:  # Crypto
                asset_drawdown = 15 + abs(np.random.normal(0, 10))
            elif 'bond' in perf['symbol'].lower():  # Bonds
                asset_drawdown = 2 + abs(np.random.normal(0, 2))
            else:  # Stocks/ETFs
                asset_drawdown = 8 + abs(np.random.normal(0, 5))
            
            max_drawdown += weight * asset_drawdown
    else:
        portfolio_beta = 1.0
        portfolio_volatility = 0.15
        sharpe_ratio = 0.0
        max_drawdown = 8.0
    
    return {
        'beta': max(0.1, min(3.0, portfolio_beta)),
        'sharpe_ratio': max(-3.0, min(4.0, sharpe_ratio)),
        'max_drawdown': max(0.5, min(50.0, max_drawdown)),
        'total_return': total_return * 100,
        'annualized_return': total_return * 12 * 100,
        'volatility': portfolio_volatility * 100,
        'var_95': abs(np.random.normal(portfolio_volatility * 100 * 0.6, 2)),
        'total_current_value': total_current_value,
        'total_invested': total_invested,
        'asset_performance': asset_performance,
        'base_currency': base_currency
    }

def calculate_technical_indicators(symbol):
    """Calculate technical indicators for a symbol"""
    dates, mock_prices = create_mock_data()
    
    if symbol not in mock_prices:
        return None
    
    data = pd.DataFrame({'Close': mock_prices[symbol]}, index=dates)
    
    # Calculate moving averages
    data['MA_20'] = data['Close'].rolling(window=20).mean()
    data['MA_50'] = data['Close'].rolling(window=50).mean()
    
    # Calculate RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # Calculate MACD
    exp1 = data['Close'].ewm(span=12).mean()
    exp2 = data['Close'].ewm(span=26).mean()
    data['MACD'] = exp1 - exp2
    data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
    
    return data.dropna()

def generate_investment_suggestions(portfolio):
    """Generate investment suggestions based on portfolio analysis"""
    suggestions = []
    
    if not portfolio:
        suggestions.append({
            'type': 'opportunity',
            'message': 'Start building your portfolio by adding diversified assets across different sectors and asset classes.'
        })
        return suggestions
    
    asset_types = {}
    total_value = 0
    
    for symbol, data in portfolio.items():
        asset_type = data['asset_type']
        asset_info = fetch_asset_data(symbol)
        
        if asset_info:
            value = data['shares'] * asset_info['current_price']
            total_value += value
            
            if asset_type in asset_types:
                asset_types[asset_type] += value
            else:
                asset_types[asset_type] = value
    
    if total_value == 0:
        return suggestions
    
    asset_percentages = {k: (v/total_value)*100 for k, v in asset_types.items()}
    
    if len(asset_types) < 3:
        suggestions.append({
            'type': 'diversification',
            'message': f'Consider diversifying across more asset classes. You currently have {len(asset_types)} asset type(s).'
        })
    
    max_percentage = max(asset_percentages.values()) if asset_percentages else 0
    if max_percentage > 40:
        max_asset_type = max(asset_percentages, key=asset_percentages.get)
        suggestions.append({
            'type': 'rebalancing',
            'message': f'Your portfolio is heavily concentrated in {max_asset_type} ({max_percentage:.1f}%).'
        })
    
    return suggestions[:5]

# Page configuration
st.set_page_config(
    page_title="Smart Portfolio Manager",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .success-card {
        background: linear-gradient(90deg, #56ab2f 0%, #a8e6cf 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .warning-card {
        background: linear-gradient(90deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .info-card {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session states
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'username' not in st.session_state:
    st.session_state.username = ""
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = {}
if 'learning_mode' not in st.session_state:
    st.session_state.learning_mode = False

def main():
    """Main application function"""
    
    st.markdown("""
    <div class="main-header">
        <h1>📊 Smart Portfolio Manager Pro</h1>
        <p>Professional Investment Analysis & Portfolio Management Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not YFINANCE_AVAILABLE or not PLOTLY_AVAILABLE:
        st.warning("⚠️ Some features are running in demo mode. For full functionality, ensure all dependencies are installed.")
    
    if not st.session_state.authenticated:
        show_auth_page()
    else:
        show_main_app()

def show_auth_page():
    """Display authentication page"""
    
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 2rem; border-radius: 15px; text-align: center; margin-bottom: 2rem; box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);">
            <h2>🔐 Secure Portfolio Access</h2>
            <p style="margin: 0; opacity: 0.9;">Professional Investment Management Platform</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.session_state.learning_mode = st.toggle(
            "📚 Learning Mode", 
            value=st.session_state.learning_mode,
            help="Enable detailed explanations and investment education"
        )
        
        if st.session_state.learning_mode:
            st.markdown("""
            <div class="info-card">
                <strong>🎓 Welcome to Learning Mode!</strong><br>
                This mode provides comprehensive explanations about portfolio management, 
                investment strategies, and financial metrics to help you become a better investor.
            </div>
            """, unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["🔑 Login", "📝 Create Account"])
        
        with tab1:
            st.markdown("### Sign In to Your Account")
            
            login_username = st.text_input(
                "Username", 
                key="login_username",
                placeholder="Enter your username"
            )
            login_password = st.text_input(
                "Password", 
                type="password", 
                key="login_password",
                placeholder="Enter your password"
            )
            
            if st.button("🚀 Login", type="primary", use_container_width=True):
                if login_username and login_password:
                    users = load_users()
                    
                    if login_username in users and verify_password(login_password, users[login_username]['password']):
                        st.session_state.authenticated = True
                        st.session_state.username = login_username
                        
                        portfolios = load_portfolios()
                        if login_username in portfolios:
                            st.session_state.portfolio = portfolios[login_username]
                        
                        st.markdown("""
                        <div class="success-card">
                            <strong>🎉 Login Successful!</strong><br>
                            Welcome back to your investment dashboard.
                        </div>
                        """, unsafe_allow_html=True)
                        st.rerun()
                    else:
                        st.markdown("""
                        <div class="warning-card">
                            <strong>❌ Authentication Failed</strong><br>
                            Invalid username or password. Please try again.
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.warning("⚠️ Please enter both username and password")
        
        with tab2:
            st.markdown("### Create Your Investment Account")
            
            reg_username = st.text_input(
                "Choose Username", 
                key="reg_username",
                placeholder="Enter a unique username"
            )
            reg_password = st.text_input(
                "Choose Password", 
                type="password", 
                key="reg_password",
                placeholder="Create a strong password"
            )
            reg_confirm_password = st.text_input(
                "Confirm Password", 
                type="password", 
                key="reg_confirm_password",
                placeholder="Re-enter your password"
            )
            
            if st.button("🎯 Create Account", type="primary", use_container_width=True):
                if reg_username and reg_password and reg_confirm_password:
                    if reg_password != reg_confirm_password:
                        st.error("❌ Passwords do not match")
                    elif len(reg_password) < 6:
                        st.error("⚠️ Password must be at least 6 characters long")
                    else:
                        users = load_users()
                        
                        if reg_username in users:
                            st.error("❌ Username already exists")
                        else:
                            users[reg_username] = {
                                'password': hash_password(reg_password),
                                'created_at': datetime.now().isoformat()
                            }
                            save_users(users)
                            
                            st.markdown("""
                            <div class="success-card">
                                <strong>🎉 Account Created Successfully!</strong><br>
                                You can now login with your credentials.
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.warning("⚠️ Please fill in all fields")

def show_main_app():
    """Display main application interface"""
    
    with st.sidebar:
        st.markdown(f"""
        <div style="background: linear-gradient(45deg, #667eea, #764ba2); color: white; padding: 1rem; border-radius: 10px; text-align: center; margin-bottom: 1rem;">
            <h3>👤 Welcome, {st.session_state.username}!</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.session_state.learning_mode = st.toggle("📚 Learning Mode", value=st.session_state.learning_mode)
        
        if st.session_state.learning_mode:
            st.markdown("""
            <div class="info-card">
                <strong>🎓 Learning Mode Active</strong><br>
                Enhanced explanations and tooltips are now visible throughout the application.
            </div>
            """, unsafe_allow_html=True)
        
        if st.button("🚪 Logout", type="secondary", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.username = ""
            st.session_state.portfolio = {}
            st.rerun()
        
        st.markdown("---")
        st.markdown("### 🧭 Navigation")
        
        selected_nav = st.radio(
            "Choose a section:",
            ["📈 Portfolio Overview", "🎯 Manage Assets", "📊 Analytics Dashboard", "📁 Export/Import"]
        )
    
    if selected_nav == "📈 Portfolio Overview":
        show_portfolio_overview()
    elif selected_nav == "🎯 Manage Assets":
        show_asset_management()
    elif selected_nav == "📊 Analytics Dashboard":
        show_analytics_dashboard()
    elif selected_nav == "📁 Export/Import":
        show_export_import()

def show_portfolio_overview():
    """Display portfolio overview page"""
    
    st.markdown("### 📈 Portfolio Overview")
    
    if st.session_state.learning_mode:
        st.markdown("""
        <div class="info-card">
            <strong>📚 Portfolio Overview Guide:</strong><br>
            This section provides a comprehensive view of your investment portfolio including performance metrics, asset allocation, individual holdings, and risk assessment. Multi-currency portfolios are automatically converted to your base currency for analysis.
        </div>
        """, unsafe_allow_html=True)
    
    if not st.session_state.portfolio:
        st.markdown("""
        <div class="warning-card">
            <strong>📝 Your portfolio is empty</strong><br>
            Get started by adding your first investment in the 'Manage Assets' section.
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Currency selection for portfolio display
    currencies = get_currency_list()
    base_currency = st.selectbox(
        "Portfolio Base Currency",
        list(currencies.keys()),
        index=0,  # Default to USD
        format_func=lambda x: f"{x} - {currencies[x]['name']}",
        help="Select the currency to display your portfolio values"
    )
    
    # Calculate advanced portfolio metrics with selected currency
    metrics = calculate_portfolio_metrics_advanced(st.session_state.portfolio, base_currency)
    currency_symbol = currencies[base_currency]["symbol"]
    
    st.markdown("### 💼 Portfolio Performance")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Total Value", 
            format_currency_value(metrics['total_current_value'], base_currency),
            delta=format_currency_value(metrics['total_current_value'] - metrics['total_invested'], base_currency)
        )
    
    with col2:
        st.metric(
            "Total Return", 
            f"{metrics['total_return']:+.1f}%",
            delta=format_currency_value(metrics['total_current_value'] - metrics['total_invested'], base_currency)
        )
    
    with col3:
        st.metric(
            "Portfolio Beta", 
            f"{metrics['beta']:.2f}",
            help="Portfolio volatility relative to the market (consistent across all views)"
        )
    
    with col4:
        st.metric(
            "Sharpe Ratio", 
            f"{metrics['sharpe_ratio']:.2f}",
            help="Risk-adjusted return measure"
        )
    
    with col5:
        st.metric(
            "Volatility", 
            f"{metrics['volatility']:.1f}%",
            help="Measure of price fluctuation"
        )
    
    # Portfolio composition with multi-currency support
    portfolio_data = []
    for symbol, data in st.session_state.portfolio.items():
        asset_info = fetch_asset_data(symbol)
        if asset_info:
            asset_currency = data.get('currency', 'USD')
            exchange_rate = get_exchange_rate(asset_currency, base_currency)
            
            current_price = asset_info['current_price']
            purchase_price = data.get('purchase_price', current_price)
            
            # Convert to base currency
            current_price_converted = current_price * exchange_rate
            purchase_price_converted = purchase_price * exchange_rate
            
            current_value = data['shares'] * current_price_converted
            invested_value = data['shares'] * purchase_price_converted
            return_pct = ((current_value - invested_value) / invested_value) * 100 if invested_value > 0 else 0
            
            asset_currency_symbol = currencies.get(asset_currency, {}).get("symbol", "$")
            
            portfolio_data.append({
                'Symbol': symbol,
                'Name': asset_info['name'][:30] + "..." if len(asset_info['name']) > 30 else asset_info['name'],
                'Shares': f"{data['shares']:.3f}",
                'Currency': asset_currency,
                'Purchase Price': f"{asset_currency_symbol}{purchase_price:.2f}",
                'Current Price': f"{asset_currency_symbol}{current_price:.2f}",
                'Invested Value': format_currency_value(invested_value, base_currency),
                'Current Value': format_currency_value(current_value, base_currency),
                'Return %': f"{return_pct:+.1f}%",
                'P&L': format_currency_value(current_value - invested_value, base_currency),
                'Asset Type': data['asset_type'],
                'Weight %': f"{(current_value / metrics['total_current_value']) * 100:.1f}%" if metrics['total_current_value'] > 0 else "0.0%"
            })
    
    if portfolio_data:
        st.markdown("### 📊 Holdings Breakdown")
        df = pd.DataFrame(portfolio_data)
        st.dataframe(df, use_container_width=True, height=400)
        
        if st.session_state.learning_mode:
            st.markdown(f"""
            <div class="info-card">
                <strong>📈 Understanding Your Multi-Currency Holdings:</strong><br>
                • <strong>Currency Column:</strong> Shows the original currency of each asset<br>
                • <strong>All values</strong> are converted to {base_currency} for comparison<br>
                • <strong>Exchange rates</strong> are applied automatically<br>
                • <strong>Green P&L:</strong> Profitable positions, <strong>Red P&L:</strong> Loss positions
            </div>
            """, unsafe_allow_html=True)
        
        if PLOTLY_AVAILABLE:
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                st.markdown("#### 🥧 Portfolio Allocation")
                # Extract numeric values safely
                current_values = []
                symbols = []
                
                for item in portfolio_data:
                    try:
                        # Parse currency values more robustly
                        value_str = item['Current Value']
                        # Remove currency symbols and commas
                        for symbol_char in ['"{return_pct:+.1f}%",
                'P&L': f"${current_value - invested_value:+,.2f}",
                'Asset Type': data['asset_type'],
                'Weight %': f"{(current_value / metrics['total_current_value']) * 100:.1f}%" if metrics['total_current_value'] > 0 else "0.0%"
            })
    
    if portfolio_data:
        st.markdown("### 📊 Holdings Breakdown")
        df = pd.DataFrame(portfolio_data)
        st.dataframe(df, use_container_width=True, height=400)
        
        if PLOTLY_AVAILABLE:
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                st.markdown("#### 🥧 Portfolio Allocation")
                # Extract numeric values safely
                current_values = []
                symbols = []
                
                for item in portfolio_data:
                    try:
                        value_str = item['Current Value'].replace('$', '').replace(',', '')
                        current_values.append(float(value_str))
                        symbols.append(item['Symbol'])
                    except:
                        continue
                
                if current_values:
                    fig_pie = px.pie(
                        values=current_values,
                        names=symbols,
                        title="Current Holdings Distribution",
                        hole=0.4
                    )
                    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                    fig_pie.update_layout(height=400)
                    st.plotly_chart(fig_pie, use_container_width=True)
            
            with col_chart2:
                st.markdown("#### 📈 Performance by Asset")
                return_values = []
                
                for item in portfolio_data:
                    try:
                        return_str = item['Return %'].replace('%', '').replace('+', '')
                        return_values.append(float(return_str))
                    except:
                        return_values.append(0.0)
                
                if return_values and symbols:
                    fig_returns = px.bar(
                        x=symbols,
                        y=return_values,
                        title="Return % by Holding",
                        color=return_values,
                        color_continuous_scale=['red', 'yellow', 'green']
                    )
                    fig_returns.update_layout(height=400, xaxis_title="Assets", yaxis_title="Return %")
                    st.plotly_chart(fig_returns, use_container_width=True)
        else:
            st.markdown("#### 📊 Simple Visualization")
            current_values = []
            symbols = []
            
            for item in portfolio_data:
                try:
                    value_str = item['Current Value'].replace('$', '').replace(',', '')
                    current_values.append(float(value_str))
                    symbols.append(item['Symbol'])
                except:
                    continue
            
            if current_values and symbols:
                chart_data = pd.DataFrame({
                    'Symbol': symbols,
                    'Current Value': current_values
                })
                st.bar_chart(chart_data.set_index('Symbol'))

def show_asset_management():
    """Display asset management page"""
    
    st.markdown("### 🎯 Manage Portfolio Assets")
    
    if st.session_state.learning_mode:
        st.markdown("""
        <div class="info-card">
            <strong>🎯 Asset Management Guide:</strong><br>
            Use this section to build and maintain your investment portfolio. Add assets, track purchase prices, and manage your holdings.
        </div>
        """, unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["➕ Add Assets", "➖ Remove Assets"])
    
    with tab1:
        st.markdown("### Add New Investment")
        
        col1, col2 = st.columns(2)
        
        with col1:
            popular_assets = get_popular_assets()
            
            # Create categories for better organization
            asset_categories = {
                "🏢 Large Cap Stocks": [k for k in popular_assets.keys() if k in ["AAPL", "GOOGL", "GOOG", "MSFT", "AMZN", "TSLA", "NVDA", "META", "NFLX", "JPM", "JNJ", "V", "WMT", "PG", "UNH", "HD", "MA", "BAC", "ADBE", "CRM", "XOM", "CVX", "KO", "PFE", "INTC", "CSCO", "VZ", "MRK", "ABT", "TMO"]],
                "🚀 Growth & Tech": [k for k in popular_assets.keys() if k in ["AMD", "PYPL", "SHOP", "SQ", "ROKU", "ZM", "SNOW", "PLTR", "CRWD", "NET"]],
                "🏛️ Vanguard ETFs": [k for k in popular_assets.keys() if k.startswith("V") and k in ["VTI", "VEA", "VWO", "VTV", "VUG", "VIG", "VYM", "VXUS", "VOO", "VT", "VB", "VO", "VGT", "VHT", "VFH", "VDE", "VAW", "VIS", "VCR", "VDC", "VPU", "VNQ"]],
                "⚫ BlackRock iShares": [k for k in popular_assets.keys() if k in ["QQQ", "IWM", "EFA", "EEM", "IVV", "IEFA", "IEMG", "IJH", "IJR", "IVW", "IVE", "IWF", "IWD", "ITOT", "IXUS", "IYY", "IWB", "IWV", "ACWI", "ACWX"]],
                "🎯 Sector ETFs": [k for k in popular_assets.keys() if k.startswith("XL") or k in ["IBB", "SOXX", "SKYY", "HACK", "ROBO", "ARKK", "ARKQ", "ARKW", "ARKG", "ARKF"]],
                "🏦 Bonds & Fixed Income": [k for k in popular_assets.keys() if k in ["AGG", "BND", "TLT", "IEF", "SHY", "LQD", "HYG", "JNK", "TIP", "VTEB", "MUB", "EMB", "BNDX", "VGIT", "VGSH", "VGLT", "VCIT", "VCSH", "BSV", "BIV", "BLV"]],
                "💎 Fidelity Funds": [k for k in popular_assets.keys() if k.startswith("F") and len(k) > 2],
                "🥇 Commodities": [k for k in popular_assets.keys() if k in ["GLD", "SLV", "USO", "UNG", "DBA", "DBC", "PDBC", "GSG", "COMT", "IAU", "SGOL", "PPLT", "PALL", "CORN", "WEAT", "SOYB", "CANE", "COW"]],
                "🌍 International": [k for k in popular_assets.keys() if k.startswith("EW") or k in ["FXI", "INDA", "RSX", "ASHR", "MCHI", "KWEB", "YINN"]],
                "₿ Major Cryptocurrencies": [k for k in popular_assets.keys() if k.endswith("-USD") and k in ["BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "ADA-USD", "SOL-USD", "DOGE-USD", "DOT-USD", "MATIC-USD", "SHIB-USD", "LTC-USD", "TRX-USD", "AVAX-USD", "LINK-USD", "ATOM-USD"]],
                "🔗 DeFi & Web3": [k for k in popular_assets.keys() if k.endswith("-USD") and k in ["XLM-USD", "ALGO-USD", "VET-USD", "ICP-USD", "FIL-USD", "SAND-USD", "MANA-USD", "AXS-USD", "ENJ-USD", "CHZ-USD", "FLOW-USD", "THETA-USD", "XTZ-USD", "EGLD-USD", "NEAR-USD"]],
                "📈 Crypto ETFs": [k for k in popular_assets.keys() if k in ["BITO", "BITI", "GBTC", "ETHE", "GDLC", "BITQ", "BLOK", "LEGR"]],
                "💱 Currency ETFs": [k for k in popular_assets.keys() if k.startswith("FX") or k in ["UUP", "UDN"]],
                "💰 Dividend ETFs": [k for k in popular_assets.keys() if k in ["SCHD", "DVY", "VYM", "NOBL", "DGRO", "HDV", "SPHD", "PEY", "RDVY", "FDL"]],
                "🌱 ESG & Clean Energy": [k for k in popular_assets.keys() if k in ["ESG", "ESGU", "ESGD", "ESGE", "VSGX", "ESGV", "SUSA", "DSI", "ICLN", "PBW", "QCLN"]],
                "📊 Volatility & Leverage": [k for k in popular_assets.keys() if k in ["VIX", "UVXY", "SVXY", "VXX", "VIXY", "TQQQ", "SQQQ", "UPRO", "SPXU", "TNA", "TZA", "TECL", "TECS", "FAS", "FAZ"]]
            }
            
            selected_category = st.selectbox(
                "Choose Asset Category",
                [""] + list(asset_categories.keys()),
                help="Browse assets by category for easier selection"
            )
            
            if selected_category:
                available_assets = asset_categories[selected_category]
                selected_popular = st.selectbox(
                    f"Assets in {selected_category}",
                    [""] + available_assets,
                    format_func=lambda x: f"{x} - {popular_assets[x]}" if x else "",
                    help="Select from assets in this category"
                )
            else:
                selected_popular = st.selectbox(
                    "Or search all assets",
                    [""] + list(popular_assets.keys()),
                    format_func=lambda x: f"{x} - {popular_assets[x]}" if x else "",
                    help="Search through all available assets"
                )
            
            custom_symbol = st.text_input(
                "Or Enter Custom Symbol",
                placeholder="e.g., AAPL, BTC-USD, GLD.L (London)",
                help="Enter any valid symbol including international exchanges"
            )
            
            symbol_to_use = ""
            if selected_popular:
                symbol_to_use = selected_popular
            elif custom_symbol:
                symbol_to_use = custom_symbol.upper()
        
        with col2:
            shares = st.number_input(
                "Number of Shares/Units",
                min_value=0.001,
                value=1.0,
                step=0.1,
                help="Enter the quantity you own"
            )
            
            purchase_price = st.number_input(
                "Purchase Price per Share ($)",
                min_value=0.01,
                value=100.0,
                step=0.01,
                help="Enter the average price you paid per share"
            )
            
            # NEW: Currency selection
            currencies = get_currency_list()
            selected_currency = st.selectbox(
                "Currency",
                list(currencies.keys()),
                index=0,  # Default to USD
                format_func=lambda x: f"{x} - {currencies[x]['name']}",
                help="Select the currency for this asset"
            )
            
            asset_type = st.selectbox(
                "Asset Type",
                ["Stock", "ETF", "Cryptocurrency", "Bond", "Commodity", "Index Fund", "Mutual Fund", "REIT", "Other"]
            )
            
            if st.session_state.learning_mode:
                st.markdown("""
                <div class="info-card">
                    <strong>💡 Pro Tip:</strong><br>
                    • Select the correct currency for international assets<br>
                    • Purchase price should be in the selected currency<br>
                    • The system will convert to your base currency for analysis
                </div>
                """, unsafe_allow_html=True)
        
        if symbol_to_use:
            asset_info = fetch_asset_data(symbol_to_use)
            if asset_info:
                current_price = asset_info['current_price']
                
                # Convert prices for display if different currency
                if selected_currency != "USD":
                    exchange_rate = get_exchange_rate("USD", selected_currency)
                    current_price_display = current_price * exchange_rate
                    purchase_price_display = purchase_price
                else:
                    current_price_display = current_price
                    purchase_price_display = purchase_price
                
                potential_return = ((current_price_display - purchase_price_display) / purchase_price_display) * 100 if purchase_price_display > 0 else 0
                
                # Create three columns for metrics display
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                
                with metric_col1:
                    currency_symbol = currencies[selected_currency]["symbol"]
                    st.metric("Current Price", f"{currency_symbol}{current_price_display:.2f}")
                with metric_col2:
                    st.metric("Your Purchase Price", f"{currency_symbol}{purchase_price_display:.2f}")
                with metric_col3:
                    st.metric("Potential Return", f"{potential_return:+.1f}%", 
                             delta=f"{currency_symbol}{(current_price_display - purchase_price_display) * shares:+.2f}")
                
                if st.button("✅ Add to Portfolio", type="primary", use_container_width=True):
                    st.session_state.portfolio[symbol_to_use] = {
                        'shares': shares,
                        'asset_type': asset_type,
                        'purchase_price': purchase_price,
                        'currency': selected_currency,  # NEW: Store currency
                        'added_date': datetime.now().isoformat()
                    }
                    
                    portfolios = load_portfolios()
                    portfolios[st.session_state.username] = st.session_state.portfolio
                    save_portfolios(portfolios)
                    
                    st.markdown("""
                    <div class="success-card">
                        <strong>🎉 Asset Added Successfully!</strong><br>
                        Your portfolio metrics are being recalculated with the new asset allocation and currency conversion.
                    </div>
                    """, unsafe_allow_html=True)
                    st.rerun()
            else:
                st.error(f"❌ Could not find asset data for '{symbol_to_use}'. Please check the symbol.")
    
    with tab2:
        st.markdown("### Remove Assets from Portfolio")
        
        if st.session_state.portfolio:
            current_values = []
            for symbol, data in st.session_state.portfolio.items():
                asset_info = fetch_asset_data(symbol)
                if asset_info:
                    current_value = data['shares'] * asset_info['current_price']
                    current_values.append({
                        'Symbol': symbol,
                        'Shares': f"{data['shares']:.3f}",
                        'Current Value': f"${current_value:,.2f}",
                        'Asset Type': data['asset_type']
                    })
            
            if current_values:
                df_current = pd.DataFrame(current_values)
                st.dataframe(df_current, hide_index=True)
            
            assets_to_remove = st.multiselect(
                "Select assets to remove:",
                list(st.session_state.portfolio.keys()),
                help="Choose one or more assets to remove from your portfolio"
            )
            
            if assets_to_remove:
                st.warning(f"⚠️ You are about to remove {len(assets_to_remove)} asset(s) from your portfolio.")
                
                if st.button("🗑️ Remove Selected Assets", type="secondary"):
                    for asset in assets_to_remove:
                        del st.session_state.portfolio[asset]
                    
                    portfolios = load_portfolios()
                    portfolios[st.session_state.username] = st.session_state.portfolio
                    save_portfolios(portfolios)
                    
                    st.markdown("""
                    <div class="success-card">
                        <strong>✅ Assets Removed Successfully!</strong><br>
                        Your portfolio has been updated and metrics recalculated.
                    </div>
                    """, unsafe_allow_html=True)
                    st.rerun()
        else:
            st.markdown("""
            <div class="info-card">
                <strong>📝 No assets to remove</strong><br>
                Your portfolio is currently empty. Add some investments first!
            </div>
            """, unsafe_allow_html=True)

def show_analytics_dashboard():
    """Display analytics dashboard with consistent beta calculation"""
    
    st.markdown("### 📊 Advanced Analytics Dashboard")
    
    if st.session_state.learning_mode:
        st.markdown("""
        <div class="info-card">
            <strong>📈 Analytics Dashboard Guide:</strong><br>
            This dashboard provides professional-grade analysis tools including performance metrics, technical analysis, and AI-powered insights. All metrics are calculated consistently across portfolio views and support multi-currency analysis.
        </div>
        """, unsafe_allow_html=True)
    
    if not st.session_state.portfolio:
        st.warning("⚠️ Add assets to your portfolio to see advanced analytics.")
        return
    
    # Currency selection for analytics
    currencies = get_currency_list()
    base_currency = st.selectbox(
        "Analytics Base Currency",
        list(currencies.keys()),
        index=0,  # Default to USD
        format_func=lambda x: f"{x} - {currencies[x]['name']}",
        help="Select the currency for analytics calculations",
        key="analytics_currency"
    )
    
    # Use the SAME calculation method as portfolio overview for consistency
    metrics = calculate_portfolio_metrics_advanced(st.session_state.portfolio, base_currency)
    
    st.markdown("### 📈 Advanced Performance Metrics")
    
    perf_col1, perf_col2, perf_col3, perf_col4, perf_col5 = st.columns(5)
    
    with perf_col1:
        st.metric(
            "Annualized Return", 
            f"{metrics['annualized_return']:+.1f}%",
            help="Expected annual return based on current performance"
        )
    
    with perf_col2:
        st.metric(
            "Portfolio Beta", 
            f"{metrics['beta']:.2f}",
            delta="Market Risk" if metrics['beta'] > 1 else "Lower Risk",
            help="Portfolio volatility relative to market (consistent with Portfolio Overview)"
        )
    
    with perf_col3:
        st.metric(
            "Sharpe Ratio", 
            f"{metrics['sharpe_ratio']:.2f}",
            delta="Excellent" if metrics['sharpe_ratio'] > 1 else "Good" if metrics['sharpe_ratio'] > 0.5 else "Needs Improvement",
            help="Risk-adjusted return measure"
        )
    
    with perf_col4:
        st.metric(
            "Max Drawdown", 
            f"{metrics['max_drawdown']:.1f}%",
            delta="Risk Level",
            delta_color="inverse",
            help="Largest peak-to-trough decline"
        )
    
    with perf_col5:
        st.metric(
            "Portfolio Volatility", 
            f"{metrics['volatility']:.1f}%",
            help="Measure of price fluctuation"
        )
    
    # Enhanced performance attribution with currency awareness
    if PLOTLY_AVAILABLE and metrics['asset_performance']:
        st.markdown("### 🎯 Advanced Portfolio Analysis")
        
        col_analysis1, col_analysis2 = st.columns(2)
        
        with col_analysis1:
            st.markdown("#### 💰 Contribution to Returns")
            
            symbols = [perf['symbol'] for perf in metrics['asset_performance']]
            contributions = [perf['weight'] * perf['return'] * 100 for perf in metrics['asset_performance']]
            
            fig_contrib = px.bar(
                x=symbols,
                y=contributions,
                title=f"Asset Contribution to Portfolio Return ({base_currency})",
                color=contributions,
                color_continuous_scale=['red', 'yellow', 'green']
            )
            fig_contrib.update_layout(height=400, xaxis_title="Assets", yaxis_title="Contribution %")
            st.plotly_chart(fig_contrib, use_container_width=True)
        
        with col_analysis2:
            st.markdown("#### ⚖️ Risk-Adjusted Performance")
            
            returns = [perf['return'] * 100 for perf in metrics['asset_performance']]
            weights = [perf['weight'] * 100 for perf in metrics['asset_performance']]
            
            fig_bubble = px.scatter(
                x=weights,
                y=returns,
                size=[abs(r) + 5 for r in returns],
                text=symbols,
                title=f"Weight vs Return Analysis ({base_currency})",
                labels={'x': 'Portfolio Weight %', 'y': 'Return %'}
            )
            fig_bubble.update_traces(textposition="middle center")
            fig_bubble.update_layout(height=400)
            st.plotly_chart(fig_bubble, use_container_width=True)
        
        # Currency exposure analysis for multi-currency portfolios
        currency_exposure = {}
        for perf in metrics['asset_performance']:
            currency = perf.get('currency', 'USD')
            if currency in currency_exposure:
                currency_exposure[currency] += perf['weight'] * 100
            else:
                currency_exposure[currency] = perf['weight'] * 100
        
        if len(currency_exposure) > 1:
            st.markdown("#### 💱 Currency Exposure")
            
            fig_currency = px.pie(
                values=list(currency_exposure.values()),
                names=list(currency_exposure.keys()),
                title="Portfolio Currency Exposure",
                hole=0.3
            )
            fig_currency.update_layout(height=300)
            st.plotly_chart(fig_currency, use_container_width=True)
            
            if st.session_state.learning_mode:
                st.markdown("""
                <div class="info-card">
                    <strong>💱 Currency Risk:</strong><br>
                    Your portfolio has exposure to multiple currencies. Currency fluctuations can impact returns when converted to your base currency. Consider hedging strategies for significant foreign currency exposure.
                </div>
                """, unsafe_allow_html=True)
    
    # Technical analysis for top holdings
    st.markdown("### 🔍 Technical Analysis - Top Holdings")
    
    portfolio_values = []
    for symbol, data in st.session_state.portfolio.items():
        asset_info = fetch_asset_data(symbol)
        if asset_info:
            asset_currency = data.get('currency', 'USD')
            exchange_rate = get_exchange_rate(asset_currency, base_currency)
            value = data['shares'] * asset_info['current_price'] * exchange_rate
            portfolio_values.append((symbol, value))
    
    portfolio_values.sort(key=lambda x: x[1], reverse=True)
    top_holdings = [item[0] for item in portfolio_values[:3]]
    
    for i, symbol in enumerate(top_holdings):
        with st.expander(f"📊 {symbol} - Technical Analysis", expanded=(i == 0)):
            indicators = calculate_technical_indicators(symbol)
            
            if indicators is not None:
                tech_col1, tech_col2, tech_col3, tech_col4 = st.columns(4)
                
                current_price = indicators['Close'].iloc[-1]
                ma20 = indicators['MA_20'].iloc[-1]
                current_rsi = indicators['RSI'].iloc[-1]
                current_macd = indicators['MACD'].iloc[-1]
                current_signal = indicators['MACD_Signal'].iloc[-1]
                
                with tech_col1:
                    price_trend = "Bullish" if current_price > ma20 else "Bearish"
                    st.metric("Price Trend", price_trend, f"${current_price:.2f}")
                
                with tech_col2:
                    rsi_status = "Overbought" if current_rsi > 70 else "Oversold" if current_rsi < 30 else "Neutral"
                    rsi_color = "🔴" if current_rsi > 70 else "🟢" if current_rsi < 30 else "🟡"
                    st.metric("RSI Signal", f"{rsi_color} {rsi_status}", f"{current_rsi:.1f}")
                
                with tech_col3:
                    macd_trend = "Bullish" if current_macd > current_signal else "Bearish"
                    macd_color = "🟢" if current_macd > current_signal else "🔴"
                    st.metric("MACD Signal", f"{macd_color} {macd_trend}")
                
                with tech_col4:
                    support_level = current_price * 0.95
                    resistance_level = current_price * 1.05
                    st.metric("Support/Resistance", f"${support_level:.2f} / ${resistance_level:.2f}")
                
                if PLOTLY_AVAILABLE:
                    fig = make_subplots(
                        rows=3, cols=1,
                        subplot_titles=(
                            f'{symbol} - Price & Moving Averages',
                            'Relative Strength Index (RSI)',
                            'MACD & Signal Line'
                        ),
                        vertical_spacing=0.08,
                        row_heights=[0.5, 0.25, 0.25]
                    )
                    
                    fig.add_trace(
                        go.Scatter(x=indicators.index, y=indicators['Close'], name='Price', line=dict(color='blue', width=2)),
                        row=1, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=indicators.index, y=indicators['MA_20'], name='MA 20', line=dict(color='orange')),
                        row=1, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=indicators.index, y=indicators['MA_50'], name='MA 50', line=dict(color='red')),
                        row=1, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(x=indicators.index, y=indicators['RSI'], name='RSI', line=dict(color='purple')),
                        row=2, col=1
                    )
                    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
                    
                    fig.add_trace(
                        go.Scatter(x=indicators.index, y=indicators['MACD'], name='MACD', line=dict(color='blue')),
                        row=3, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=indicators.index, y=indicators['MACD_Signal'], name='Signal', line=dict(color='red')),
                        row=3, col=1
                    )
                    
                    fig.update_layout(height=700, showlegend=True, title_text=f"{symbol} Technical Analysis")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.markdown("#### 📈 Price Chart")
                    st.line_chart(indicators[['Close', 'MA_20', 'MA_50']])
                    
                    simple_col1, simple_col2 = st.columns(2)
                    with simple_col1:
                        st.markdown("#### RSI")
                        st.line_chart(indicators['RSI'])
                    with simple_col2:
                        st.markdown("#### MACD")
                        st.line_chart(indicators[['MACD', 'MACD_Signal']])
    
    # Enhanced investment suggestions with currency considerations
    st.markdown("### 🤖 AI-Powered Investment Insights")
    suggestions = generate_investment_suggestions(st.session_state.portfolio)
    
    enhanced_suggestions = []
    
    if metrics['total_return'] > 10:
        enhanced_suggestions.append({
            'type': 'success',
            'message': f'🎉 Excellent performance! Your portfolio is up {metrics["total_return"]:.1f}% in {base_currency} terms. Consider taking some profits and rebalancing.'
        })
    elif metrics['total_return'] < -5:
        enhanced_suggestions.append({
            'type': 'warning',
            'message': f'📉 Portfolio is down {abs(metrics["total_return"]):.1f}% in {base_currency} terms. Review underperforming assets and consider strategic rebalancing.'
        })
    
    if metrics['volatility'] > 25:
        enhanced_suggestions.append({
            'type': 'warning',
            'message': f'⚡ High volatility detected ({metrics["volatility"]:.1f}%). Consider adding stable assets like bonds or dividend stocks to reduce portfolio risk.'
        })
    
    if metrics['beta'] > 1.5:
        enhanced_suggestions.append({
            'type': 'warning', 
            'message': f'📊 High portfolio beta ({metrics["beta"]:.2f}) indicates significant market sensitivity. Consider adding defensive assets to reduce systematic risk.'
        })
    elif metrics['beta'] < 0.8:
        enhanced_suggestions.append({
            'type': 'info',
            'message': f'🛡️ Low portfolio beta ({metrics["beta"]:.2f}) indicates conservative positioning. Consider adding growth assets if you can tolerate more risk.'
        })
    
    # Currency-specific suggestions
    currency_exposure = {}
    for perf in metrics['asset_performance']:
        currency = perf.get('currency', 'USD')
        if currency in currency_exposure:
            currency_exposure[currency] += perf['weight'] * 100
        else:
            currency_exposure[currency] = perf['weight'] * 100
    
    if len(currency_exposure) > 1:
        max_currency_exposure = max(currency_exposure.values())
        if max_currency_exposure < 50:
            enhanced_suggestions.append({
                'type': 'opportunity',
                'message': f'🌍 Well-diversified currency exposure detected. Your portfolio spans {len(currency_exposure)} currencies, providing natural hedging against currency risk.'
            })
        else:
            dominant_currency = max(currency_exposure, key=currency_exposure.get)
            enhanced_suggestions.append({
                'type': 'warning',
                'message': f'💱 High {dominant_currency} exposure ({max_currency_exposure:.1f}%). Consider diversifying into other currencies or using currency hedging strategies.'
            })
    
    all_suggestions = enhanced_suggestions + suggestions
    
    for suggestion in all_suggestions[:8]:  # Show up to 8 suggestions
        if suggestion['type'] == 'diversification':
            st.markdown(f"""
            <div class="info-card">
                <strong>🎯 Diversification:</strong> {suggestion['message']}
            </div>
            """, unsafe_allow_html=True)
        elif suggestion['type'] == 'rebalancing':
            st.markdown(f"""
            <div class="warning-card">
                <strong>⚖️ Rebalancing:</strong> {suggestion['message']}
            </div>
            """, unsafe_allow_html=True)
        elif suggestion['type'] == 'opportunity':
            st.markdown(f"""
            <div class="success-card">
                <strong>🚀 Opportunity:</strong> {suggestion['message']}
            </div>
            """, unsafe_allow_html=True)
        elif suggestion['type'] == 'success':
            st.markdown(f"""
            <div class="success-card">
                <strong>✅ Success:</strong> {suggestion['message']}
            </div>
            """, unsafe_allow_html=True)
        elif suggestion['type'] == 'warning':
            st.markdown(f"""
            <div class="warning-card">
                <strong>⚠️ Warning:</strong> {suggestion['message']}
            </div>
            """, unsafe_allow_html=True)
        elif suggestion['type'] == 'info':
            st.markdown(f"""
            <div class="info-card">
                <strong>ℹ️ Information:</strong> {suggestion['message']}
            </div>
            """, unsafe_allow_html=True)

def show_export_import():
    """Display export/import functionality"""
    
    st.markdown("### 📁 Export & Import Portfolio")
    
    if st.session_state.learning_mode:
        st.markdown("""
        <div class="info-card">
            <strong>📁 Export/Import Guide:</strong><br>
            Backup and restore your portfolio data. Export as JSON or CSV for safekeeping, or import previously saved portfolios.
        </div>
        """, unsafe_allow_html=True)
    
    export_tab, import_tab = st.tabs(["📤 Export Portfolio", "📥 Import Portfolio"])
    
    with export_tab:
        st.markdown("### Export Your Portfolio Data")
        
        if not st.session_state.portfolio:
            st.markdown("""
            <div class="warning-card">
                <strong>📝 No portfolio data to export</strong><br>
                Add some investments to your portfolio first, then return here to export your data.
            </div>
            """, unsafe_allow_html=True)
            return
        
        export_format = st.selectbox("Choose Export Format", ["JSON", "CSV"])
        
        if export_format == "JSON":
            export_data = {
                'username': st.session_state.username,
                'export_date': datetime.now().isoformat(),
                'portfolio': st.session_state.portfolio,
                'version': '2.0'
            }
            
            json_string = json.dumps(export_data, indent=2)
            
            st.download_button(
                label="📥 Download Portfolio (JSON)",
                data=json_string,
                file_name=f"portfolio_{st.session_state.username}_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )
            
            with st.expander("👀 Preview JSON Data"):
                st.code(json_string, language="json")
        
        else:  # CSV format
            csv_data = []
            for symbol, data in st.session_state.portfolio.items():
                asset_info = fetch_asset_data(symbol)
                if asset_info:
                    csv_data.append({
                        'Symbol': symbol,
                        'Name': asset_info['name'],
                        'Shares': data['shares'],
                        'Asset_Type': data['asset_type'],
                        'Currency': data.get('currency', 'USD'),  # Include currency
                        'Purchase_Price': data.get('purchase_price', asset_info['current_price']),
                        'Current_Price': asset_info['current_price'],
                        'Current_Value': data['shares'] * asset_info['current_price'],
                        'Added_Date': data['added_date']
                    })
            
            if csv_data:
                df = pd.DataFrame(csv_data)
                csv_string = df.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download Portfolio (CSV)",
                    data=csv_string,
                    file_name=f"portfolio_{st.session_state.username}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
                
                with st.expander("👀 Preview CSV Data"):
                    st.dataframe(df)
    
    with import_tab:
        st.markdown("### Import Portfolio Data")
        
        uploaded_file = st.file_uploader(
            "Choose a portfolio file",
            type=['json', 'csv'],
            help="Upload a previously exported portfolio file"
        )
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.json'):
                    content = uploaded_file.read()
                    data = json.loads(content)
                    
                    if 'portfolio' in data:
                        st.markdown("""
                        <div class="success-card">
                            <strong>✅ Valid portfolio file detected!</strong><br>
                            Ready to import your portfolio data.
                        </div>
                        """, unsafe_allow_html=True)
                        
                        with st.expander("👀 Preview Import Data"):
                            st.json(data['portfolio'])
                        
                        import_option = st.selectbox(
                            "Import Option",
                            ["Replace Current Portfolio", "Merge with Current Portfolio"]
                        )
                        
                        if st.button("🚀 Import Portfolio", type="primary"):
                            if import_option == "Replace Current Portfolio":
                                st.session_state.portfolio = data['portfolio']
                            else:
                                for symbol, asset_data in data['portfolio'].items():
                                    st.session_state.portfolio[symbol] = asset_data
                            
                            portfolios = load_portfolios()
                            portfolios[st.session_state.username] = st.session_state.portfolio
                            save_portfolios(portfolios)
                            
                            st.markdown("""
                            <div class="success-card">
                                <strong>🎉 Portfolio imported successfully!</strong><br>
                                Your portfolio has been updated with the imported data.
                            </div>
                            """, unsafe_allow_html=True)
                            st.rerun()
                    else:
                        st.error("❌ Invalid portfolio file format.")
                
                elif uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                    
                    required_columns = ['Symbol', 'Shares', 'Asset_Type']
                    if all(col in df.columns for col in required_columns):
                        st.markdown("""
                        <div class="success-card">
                            <strong>✅ Valid CSV file detected!</strong><br>
                            Ready to import your portfolio data.
                        </div>
                        """, unsafe_allow_html=True)
                        
                        with st.expander("👀 Preview Import Data"):
                            st.dataframe(df)
                        
                        csv_import_option = st.selectbox(
                            "Import Option",
                            ["Replace Current Portfolio", "Merge with Current Portfolio"],
                            key="csv_import_option"
                        )
                        
                        if st.button("🚀 Import CSV Portfolio", type="primary"):
                            new_portfolio = {}
                            
                            for _, row in df.iterrows():
                                symbol = row['Symbol']
                                new_portfolio[symbol] = {
                                    'shares': float(row['Shares']),
                                    'asset_type': row['Asset_Type'],
                                    'purchase_price': float(row.get('Purchase_Price', 100.0)),
                                    'added_date': datetime.now().isoformat()
                                }
                            
                            if csv_import_option == "Replace Current Portfolio":
                                st.session_state.portfolio = new_portfolio
                            else:
                                for symbol, asset_data in new_portfolio.items():
                                    st.session_state.portfolio[symbol] = asset_data
                            
                            portfolios = load_portfolios()
                            portfolios[st.session_state.username] = st.session_state.portfolio
                            save_portfolios(portfolios)
                            
                            st.markdown("""
                            <div class="success-card">
                                <strong>🎉 CSV portfolio imported successfully!</strong><br>
                                Your portfolio has been updated with the imported data.
                            </div>
                            """, unsafe_allow_html=True)
                            st.rerun()
                    else:
                        st.error(f"❌ CSV file must contain columns: {', '.join(required_columns)}")
                        
            except Exception as e:
                st.error(f"❌ Error importing file: {str(e)}")

if __name__ == "__main__":
    main(), '€', '£', '¥', '₹', '₩', 'C"{return_pct:+.1f}%",
                'P&L': f"${current_value - invested_value:+,.2f}",
                'Asset Type': data['asset_type'],
                'Weight %': f"{(current_value / metrics['total_current_value']) * 100:.1f}%" if metrics['total_current_value'] > 0 else "0.0%"
            })
    
    if portfolio_data:
        st.markdown("### 📊 Holdings Breakdown")
        df = pd.DataFrame(portfolio_data)
        st.dataframe(df, use_container_width=True, height=400)
        
        if PLOTLY_AVAILABLE:
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                st.markdown("#### 🥧 Portfolio Allocation")
                # Extract numeric values safely
                current_values = []
                symbols = []
                
                for item in portfolio_data:
                    try:
                        value_str = item['Current Value'].replace('$', '').replace(',', '')
                        current_values.append(float(value_str))
                        symbols.append(item['Symbol'])
                    except:
                        continue
                
                if current_values:
                    fig_pie = px.pie(
                        values=current_values,
                        names=symbols,
                        title="Current Holdings Distribution",
                        hole=0.4
                    )
                    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                    fig_pie.update_layout(height=400)
                    st.plotly_chart(fig_pie, use_container_width=True)
            
            with col_chart2:
                st.markdown("#### 📈 Performance by Asset")
                return_values = []
                
                for item in portfolio_data:
                    try:
                        return_str = item['Return %'].replace('%', '').replace('+', '')
                        return_values.append(float(return_str))
                    except:
                        return_values.append(0.0)
                
                if return_values and symbols:
                    fig_returns = px.bar(
                        x=symbols,
                        y=return_values,
                        title="Return % by Holding",
                        color=return_values,
                        color_continuous_scale=['red', 'yellow', 'green']
                    )
                    fig_returns.update_layout(height=400, xaxis_title="Assets", yaxis_title="Return %")
                    st.plotly_chart(fig_returns, use_container_width=True)
        else:
            st.markdown("#### 📊 Simple Visualization")
            current_values = []
            symbols = []
            
            for item in portfolio_data:
                try:
                    value_str = item['Current Value'].replace('$', '').replace(',', '')
                    current_values.append(float(value_str))
                    symbols.append(item['Symbol'])
                except:
                    continue
            
            if current_values and symbols:
                chart_data = pd.DataFrame({
                    'Symbol': symbols,
                    'Current Value': current_values
                })
                st.bar_chart(chart_data.set_index('Symbol'))

def show_asset_management():
    """Display asset management page"""
    
    st.markdown("### 🎯 Manage Portfolio Assets")
    
    if st.session_state.learning_mode:
        st.markdown("""
        <div class="info-card">
            <strong>🎯 Asset Management Guide:</strong><br>
            Use this section to build and maintain your investment portfolio. Add assets, track purchase prices, and manage your holdings.
        </div>
        """, unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["➕ Add Assets", "➖ Remove Assets"])
    
    with tab1:
        st.markdown("### Add New Investment")
        
        col1, col2 = st.columns(2)
        
        with col1:
            popular_assets = get_popular_assets()
            
            # Create categories for better organization
            asset_categories = {
                "🏢 Large Cap Stocks": [k for k in popular_assets.keys() if k in ["AAPL", "GOOGL", "GOOG", "MSFT", "AMZN", "TSLA", "NVDA", "META", "NFLX", "JPM", "JNJ", "V", "WMT", "PG", "UNH", "HD", "MA", "BAC", "ADBE", "CRM", "XOM", "CVX", "KO", "PFE", "INTC", "CSCO", "VZ", "MRK", "ABT", "TMO"]],
                "🚀 Growth & Tech": [k for k in popular_assets.keys() if k in ["AMD", "PYPL", "SHOP", "SQ", "ROKU", "ZM", "SNOW", "PLTR", "CRWD", "NET"]],
                "🏛️ Vanguard ETFs": [k for k in popular_assets.keys() if k.startswith("V") and k in ["VTI", "VEA", "VWO", "VTV", "VUG", "VIG", "VYM", "VXUS", "VOO", "VT", "VB", "VO", "VGT", "VHT", "VFH", "VDE", "VAW", "VIS", "VCR", "VDC", "VPU", "VNQ"]],
                "⚫ BlackRock iShares": [k for k in popular_assets.keys() if k in ["QQQ", "IWM", "EFA", "EEM", "IVV", "IEFA", "IEMG", "IJH", "IJR", "IVW", "IVE", "IWF", "IWD", "ITOT", "IXUS", "IYY", "IWB", "IWV", "ACWI", "ACWX"]],
                "🎯 Sector ETFs": [k for k in popular_assets.keys() if k.startswith("XL") or k in ["IBB", "SOXX", "SKYY", "HACK", "ROBO", "ARKK", "ARKQ", "ARKW", "ARKG", "ARKF"]],
                "🏦 Bonds & Fixed Income": [k for k in popular_assets.keys() if k in ["AGG", "BND", "TLT", "IEF", "SHY", "LQD", "HYG", "JNK", "TIP", "VTEB", "MUB", "EMB", "BNDX", "VGIT", "VGSH", "VGLT", "VCIT", "VCSH", "BSV", "BIV", "BLV"]],
                "💎 Fidelity Funds": [k for k in popular_assets.keys() if k.startswith("F") and len(k) > 2],
                "🥇 Commodities": [k for k in popular_assets.keys() if k in ["GLD", "SLV", "USO", "UNG", "DBA", "DBC", "PDBC", "GSG", "COMT", "IAU", "SGOL", "PPLT", "PALL", "CORN", "WEAT", "SOYB", "CANE", "COW"]],
                "🌍 International": [k for k in popular_assets.keys() if k.startswith("EW") or k in ["FXI", "INDA", "RSX", "ASHR", "MCHI", "KWEB", "YINN"]],
                "₿ Major Cryptocurrencies": [k for k in popular_assets.keys() if k.endswith("-USD") and k in ["BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "ADA-USD", "SOL-USD", "DOGE-USD", "DOT-USD", "MATIC-USD", "SHIB-USD", "LTC-USD", "TRX-USD", "AVAX-USD", "LINK-USD", "ATOM-USD"]],
                "🔗 DeFi & Web3": [k for k in popular_assets.keys() if k.endswith("-USD") and k in ["XLM-USD", "ALGO-USD", "VET-USD", "ICP-USD", "FIL-USD", "SAND-USD", "MANA-USD", "AXS-USD", "ENJ-USD", "CHZ-USD", "FLOW-USD", "THETA-USD", "XTZ-USD", "EGLD-USD", "NEAR-USD"]],
                "📈 Crypto ETFs": [k for k in popular_assets.keys() if k in ["BITO", "BITI", "GBTC", "ETHE", "GDLC", "BITQ", "BLOK", "LEGR"]],
                "💱 Currency ETFs": [k for k in popular_assets.keys() if k.startswith("FX") or k in ["UUP", "UDN"]],
                "💰 Dividend ETFs": [k for k in popular_assets.keys() if k in ["SCHD", "DVY", "VYM", "NOBL", "DGRO", "HDV", "SPHD", "PEY", "RDVY", "FDL"]],
                "🌱 ESG & Clean Energy": [k for k in popular_assets.keys() if k in ["ESG", "ESGU", "ESGD", "ESGE", "VSGX", "ESGV", "SUSA", "DSI", "ICLN", "PBW", "QCLN"]],
                "📊 Volatility & Leverage": [k for k in popular_assets.keys() if k in ["VIX", "UVXY", "SVXY", "VXX", "VIXY", "TQQQ", "SQQQ", "UPRO", "SPXU", "TNA", "TZA", "TECL", "TECS", "FAS", "FAZ"]]
            }
            
            selected_category = st.selectbox(
                "Choose Asset Category",
                [""] + list(asset_categories.keys()),
                help="Browse assets by category for easier selection"
            )
            
            if selected_category:
                available_assets = asset_categories[selected_category]
                selected_popular = st.selectbox(
                    f"Assets in {selected_category}",
                    [""] + available_assets,
                    format_func=lambda x: f"{x} - {popular_assets[x]}" if x else "",
                    help="Select from assets in this category"
                )
            else:
                selected_popular = st.selectbox(
                    "Or search all assets",
                    [""] + list(popular_assets.keys()),
                    format_func=lambda x: f"{x} - {popular_assets[x]}" if x else "",
                    help="Search through all available assets"
                )
            
            custom_symbol = st.text_input(
                "Or Enter Custom Symbol",
                placeholder="e.g., AAPL, BTC-USD, GLD.L (London)",
                help="Enter any valid symbol including international exchanges"
            )
            
            symbol_to_use = ""
            if selected_popular:
                symbol_to_use = selected_popular
            elif custom_symbol:
                symbol_to_use = custom_symbol.upper()
        
        with col2:
            shares = st.number_input(
                "Number of Shares/Units",
                min_value=0.001,
                value=1.0,
                step=0.1,
                help="Enter the quantity you own"
            )
            
            purchase_price = st.number_input(
                "Purchase Price per Share ($)",
                min_value=0.01,
                value=100.0,
                step=0.01,
                help="Enter the average price you paid per share"
            )
            
            # NEW: Currency selection
            currencies = get_currency_list()
            selected_currency = st.selectbox(
                "Currency",
                list(currencies.keys()),
                index=0,  # Default to USD
                format_func=lambda x: f"{x} - {currencies[x]['name']}",
                help="Select the currency for this asset"
            )
            
            asset_type = st.selectbox(
                "Asset Type",
                ["Stock", "ETF", "Cryptocurrency", "Bond", "Commodity", "Index Fund", "Mutual Fund", "REIT", "Other"]
            )
            
            if st.session_state.learning_mode:
                st.markdown("""
                <div class="info-card">
                    <strong>💡 Pro Tip:</strong><br>
                    • Select the correct currency for international assets<br>
                    • Purchase price should be in the selected currency<br>
                    • The system will convert to your base currency for analysis
                </div>
                """, unsafe_allow_html=True)
        
        if symbol_to_use:
            asset_info = fetch_asset_data(symbol_to_use)
            if asset_info:
                current_price = asset_info['current_price']
                
                # Convert prices for display if different currency
                if selected_currency != "USD":
                    exchange_rate = get_exchange_rate("USD", selected_currency)
                    current_price_display = current_price * exchange_rate
                    purchase_price_display = purchase_price
                else:
                    current_price_display = current_price
                    purchase_price_display = purchase_price
                
                potential_return = ((current_price_display - purchase_price_display) / purchase_price_display) * 100 if purchase_price_display > 0 else 0
                
                # Create three columns for metrics display
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                
                with metric_col1:
                    currency_symbol = currencies[selected_currency]["symbol"]
                    st.metric("Current Price", f"{currency_symbol}{current_price_display:.2f}")
                with metric_col2:
                    st.metric("Your Purchase Price", f"{currency_symbol}{purchase_price_display:.2f}")
                with metric_col3:
                    st.metric("Potential Return", f"{potential_return:+.1f}%", 
                             delta=f"{currency_symbol}{(current_price_display - purchase_price_display) * shares:+.2f}")
                
                if st.button("✅ Add to Portfolio", type="primary", use_container_width=True):
                    st.session_state.portfolio[symbol_to_use] = {
                        'shares': shares,
                        'asset_type': asset_type,
                        'purchase_price': purchase_price,
                        'currency': selected_currency,  # NEW: Store currency
                        'added_date': datetime.now().isoformat()
                    }
                    
                    portfolios = load_portfolios()
                    portfolios[st.session_state.username] = st.session_state.portfolio
                    save_portfolios(portfolios)
                    
                    st.markdown("""
                    <div class="success-card">
                        <strong>🎉 Asset Added Successfully!</strong><br>
                        Your portfolio metrics are being recalculated with the new asset allocation and currency conversion.
                    </div>
                    """, unsafe_allow_html=True)
                    st.rerun()
            else:
                st.error(f"❌ Could not find asset data for '{symbol_to_use}'. Please check the symbol.")
    
    with tab2:
        st.markdown("### Remove Assets from Portfolio")
        
        if st.session_state.portfolio:
            current_values = []
            for symbol, data in st.session_state.portfolio.items():
                asset_info = fetch_asset_data(symbol)
                if asset_info:
                    current_value = data['shares'] * asset_info['current_price']
                    current_values.append({
                        'Symbol': symbol,
                        'Shares': f"{data['shares']:.3f}",
                        'Current Value': f"${current_value:,.2f}",
                        'Asset Type': data['asset_type']
                    })
            
            if current_values:
                df_current = pd.DataFrame(current_values)
                st.dataframe(df_current, hide_index=True)
            
            assets_to_remove = st.multiselect(
                "Select assets to remove:",
                list(st.session_state.portfolio.keys()),
                help="Choose one or more assets to remove from your portfolio"
            )
            
            if assets_to_remove:
                st.warning(f"⚠️ You are about to remove {len(assets_to_remove)} asset(s) from your portfolio.")
                
                if st.button("🗑️ Remove Selected Assets", type="secondary"):
                    for asset in assets_to_remove:
                        del st.session_state.portfolio[asset]
                    
                    portfolios = load_portfolios()
                    portfolios[st.session_state.username] = st.session_state.portfolio
                    save_portfolios(portfolios)
                    
                    st.markdown("""
                    <div class="success-card">
                        <strong>✅ Assets Removed Successfully!</strong><br>
                        Your portfolio has been updated and metrics recalculated.
                    </div>
                    """, unsafe_allow_html=True)
                    st.rerun()
        else:
            st.markdown("""
            <div class="info-card">
                <strong>📝 No assets to remove</strong><br>
                Your portfolio is currently empty. Add some investments first!
            </div>
            """, unsafe_allow_html=True)

def show_analytics_dashboard():
    """Display analytics dashboard"""
    
    st.markdown("### 📊 Advanced Analytics Dashboard")
    
    if st.session_state.learning_mode:
        st.markdown("""
        <div class="info-card">
            <strong>📈 Analytics Dashboard Guide:</strong><br>
            This dashboard provides professional-grade analysis tools including performance metrics, technical analysis, and AI-powered insights.
        </div>
        """, unsafe_allow_html=True)
    
    if not st.session_state.portfolio:
        st.warning("⚠️ Add assets to your portfolio to see advanced analytics.")
        return
    
    metrics = calculate_portfolio_metrics_advanced(st.session_state.portfolio)
    
    st.markdown("### 📈 Advanced Performance Metrics")
    
    perf_col1, perf_col2, perf_col3, perf_col4, perf_col5 = st.columns(5)
    
    with perf_col1:
        st.metric(
            "Annualized Return", 
            f"{metrics['annualized_return']:+.1f}%",
            help="Expected annual return based on current performance"
        )
    
    with perf_col2:
        st.metric(
            "Portfolio Beta", 
            f"{metrics['beta']:.2f}",
            delta="Market Risk" if metrics['beta'] > 1 else "Lower Risk",
            help="Portfolio volatility relative to market"
        )
    
    with perf_col3:
        st.metric(
            "Sharpe Ratio", 
            f"{metrics['sharpe_ratio']:.2f}",
            delta="Excellent" if metrics['sharpe_ratio'] > 1 else "Good" if metrics['sharpe_ratio'] > 0.5 else "Needs Improvement",
            help="Risk-adjusted return measure"
        )
    
    with perf_col4:
        st.metric(
            "Max Drawdown", 
            f"{metrics['max_drawdown']:.1f}%",
            delta="Risk Level",
            delta_color="inverse",
            help="Largest peak-to-trough decline"
        )
    
    with perf_col5:
        st.metric(
            "Portfolio Volatility", 
            f"{metrics['volatility']:.1f}%",
            help="Measure of price fluctuation"
        )
    
    # Technical analysis for top holdings
    st.markdown("### 🔍 Technical Analysis - Top Holdings")
    
    portfolio_values = []
    for symbol, data in st.session_state.portfolio.items():
        asset_info = fetch_asset_data(symbol)
        if asset_info:
            value = data['shares'] * asset_info['current_price']
            portfolio_values.append((symbol, value))
    
    portfolio_values.sort(key=lambda x: x[1], reverse=True)
    top_holdings = [item[0] for item in portfolio_values[:3]]
    
    for i, symbol in enumerate(top_holdings):
        with st.expander(f"📊 {symbol} - Technical Analysis", expanded=(i == 0)):
            indicators = calculate_technical_indicators(symbol)
            
            if indicators is not None:
                tech_col1, tech_col2, tech_col3, tech_col4 = st.columns(4)
                
                current_price = indicators['Close'].iloc[-1]
                ma20 = indicators['MA_20'].iloc[-1]
                current_rsi = indicators['RSI'].iloc[-1]
                current_macd = indicators['MACD'].iloc[-1]
                current_signal = indicators['MACD_Signal'].iloc[-1]
                
                with tech_col1:
                    price_trend = "Bullish" if current_price > ma20 else "Bearish"
                    st.metric("Price Trend", price_trend, f"${current_price:.2f}")
                
                with tech_col2:
                    rsi_status = "Overbought" if current_rsi > 70 else "Oversold" if current_rsi < 30 else "Neutral"
                    rsi_color = "🔴" if current_rsi > 70 else "🟢" if current_rsi < 30 else "🟡"
                    st.metric("RSI Signal", f"{rsi_color} {rsi_status}", f"{current_rsi:.1f}")
                
                with tech_col3:
                    macd_trend = "Bullish" if current_macd > current_signal else "Bearish"
                    macd_color = "🟢" if current_macd > current_signal else "🔴"
                    st.metric("MACD Signal", f"{macd_color} {macd_trend}")
                
                with tech_col4:
                    support_level = current_price * 0.95
                    resistance_level = current_price * 1.05
                    st.metric("Support/Resistance", f"${support_level:.2f} / ${resistance_level:.2f}")
                
                if PLOTLY_AVAILABLE:
                    fig = make_subplots(
                        rows=3, cols=1,
                        subplot_titles=(
                            f'{symbol} - Price & Moving Averages',
                            'Relative Strength Index (RSI)',
                            'MACD & Signal Line'
                        ),
                        vertical_spacing=0.08,
                        row_heights=[0.5, 0.25, 0.25]
                    )
                    
                    fig.add_trace(
                        go.Scatter(x=indicators.index, y=indicators['Close'], name='Price', line=dict(color='blue', width=2)),
                        row=1, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=indicators.index, y=indicators['MA_20'], name='MA 20', line=dict(color='orange')),
                        row=1, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=indicators.index, y=indicators['MA_50'], name='MA 50', line=dict(color='red')),
                        row=1, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(x=indicators.index, y=indicators['RSI'], name='RSI', line=dict(color='purple')),
                        row=2, col=1
                    )
                    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
                    
                    fig.add_trace(
                        go.Scatter(x=indicators.index, y=indicators['MACD'], name='MACD', line=dict(color='blue')),
                        row=3, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=indicators.index, y=indicators['MACD_Signal'], name='Signal', line=dict(color='red')),
                        row=3, col=1
                    )
                    
                    fig.update_layout(height=700, showlegend=True, title_text=f"{symbol} Technical Analysis")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.markdown("#### 📈 Price Chart")
                    st.line_chart(indicators[['Close', 'MA_20', 'MA_50']])
                    
                    simple_col1, simple_col2 = st.columns(2)
                    with simple_col1:
                        st.markdown("#### RSI")
                        st.line_chart(indicators['RSI'])
                    with simple_col2:
                        st.markdown("#### MACD")
                        st.line_chart(indicators[['MACD', 'MACD_Signal']])
    
    # Investment suggestions
    st.markdown("### 🤖 AI-Powered Investment Insights")
    suggestions = generate_investment_suggestions(st.session_state.portfolio)
    
    enhanced_suggestions = []
    
    if metrics['total_return'] > 10:
        enhanced_suggestions.append({
            'type': 'success',
            'message': f'🎉 Excellent performance! Your portfolio is up {metrics["total_return"]:.1f}%. Consider taking some profits and rebalancing.'
        })
    elif metrics['total_return'] < -5:
        enhanced_suggestions.append({
            'type': 'warning',
            'message': f'📉 Portfolio is down {abs(metrics["total_return"]):.1f}%. Review underperforming assets and consider strategic rebalancing.'
        })
    
    if metrics['volatility'] > 25:
        enhanced_suggestions.append({
            'type': 'warning',
            'message': f'⚡ High volatility detected ({metrics["volatility"]:.1f}%). Consider adding stable assets like bonds or dividend stocks.'
        })
    
    all_suggestions = enhanced_suggestions + suggestions
    
    for suggestion in all_suggestions[:6]:
        if suggestion['type'] == 'diversification':
            st.markdown(f"""
            <div class="info-card">
                <strong>🎯 Diversification:</strong> {suggestion['message']}
            </div>
            """, unsafe_allow_html=True)
        elif suggestion['type'] == 'rebalancing':
            st.markdown(f"""
            <div class="warning-card">
                <strong>⚖️ Rebalancing:</strong> {suggestion['message']}
            </div>
            """, unsafe_allow_html=True)
        elif suggestion['type'] == 'opportunity':
            st.markdown(f"""
            <div class="success-card">
                <strong>🚀 Opportunity:</strong> {suggestion['message']}
            </div>
            """, unsafe_allow_html=True)
        elif suggestion['type'] == 'success':
            st.markdown(f"""
            <div class="success-card">
                <strong>✅ Success:</strong> {suggestion['message']}
            </div>
            """, unsafe_allow_html=True)
        elif suggestion['type'] == 'warning':
            st.markdown(f"""
            <div class="warning-card">
                <strong>⚠️ Warning:</strong> {suggestion['message']}
            </div>
            """, unsafe_allow_html=True)

def show_export_import():
    """Display export/import functionality"""
    
    st.markdown("### 📁 Export & Import Portfolio")
    
    if st.session_state.learning_mode:
        st.markdown("""
        <div class="info-card">
            <strong>📁 Export/Import Guide:</strong><br>
            Backup and restore your portfolio data. Export as JSON or CSV for safekeeping, or import previously saved portfolios.
        </div>
        """, unsafe_allow_html=True)
    
    export_tab, import_tab = st.tabs(["📤 Export Portfolio", "📥 Import Portfolio"])
    
    with export_tab:
        st.markdown("### Export Your Portfolio Data")
        
        if not st.session_state.portfolio:
            st.markdown("""
            <div class="warning-card">
                <strong>📝 No portfolio data to export</strong><br>
                Add some investments to your portfolio first, then return here to export your data.
            </div>
            """, unsafe_allow_html=True)
            return
        
        export_format = st.selectbox("Choose Export Format", ["JSON", "CSV"])
        
        if export_format == "JSON":
            export_data = {
                'username': st.session_state.username,
                'export_date': datetime.now().isoformat(),
                'portfolio': st.session_state.portfolio,
                'version': '2.0'
            }
            
            json_string = json.dumps(export_data, indent=2)
            
            st.download_button(
                label="📥 Download Portfolio (JSON)",
                data=json_string,
                file_name=f"portfolio_{st.session_state.username}_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )
            
            with st.expander("👀 Preview JSON Data"):
                st.code(json_string, language="json")
        
        else:  # CSV format
            csv_data = []
            for symbol, data in st.session_state.portfolio.items():
                asset_info = fetch_asset_data(symbol)
                if asset_info:
                    csv_data.append({
                        'Symbol': symbol,
                        'Name': asset_info['name'],
                        'Shares': data['shares'],
                        'Asset_Type': data['asset_type'],
                        'Purchase_Price': data.get('purchase_price', asset_info['current_price']),
                        'Current_Price': asset_info['current_price'],
                        'Current_Value': data['shares'] * asset_info['current_price'],
                        'Added_Date': data['added_date']
                    })
            
            if csv_data:
                df = pd.DataFrame(csv_data)
                csv_string = df.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download Portfolio (CSV)",
                    data=csv_string,
                    file_name=f"portfolio_{st.session_state.username}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
                
                with st.expander("👀 Preview CSV Data"):
                    st.dataframe(df)
    
    with import_tab:
        st.markdown("### Import Portfolio Data")
        
        uploaded_file = st.file_uploader(
            "Choose a portfolio file",
            type=['json', 'csv'],
            help="Upload a previously exported portfolio file"
        )
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.json'):
                    content = uploaded_file.read()
                    data = json.loads(content)
                    
                    if 'portfolio' in data:
                        st.markdown("""
                        <div class="success-card">
                            <strong>✅ Valid portfolio file detected!</strong><br>
                            Ready to import your portfolio data.
                        </div>
                        """, unsafe_allow_html=True)
                        
                        with st.expander("👀 Preview Import Data"):
                            st.json(data['portfolio'])
                        
                        import_option = st.selectbox(
                            "Import Option",
                            ["Replace Current Portfolio", "Merge with Current Portfolio"]
                        )
                        
                        if st.button("🚀 Import Portfolio", type="primary"):
                            if import_option == "Replace Current Portfolio":
                                st.session_state.portfolio = data['portfolio']
                            else:
                                for symbol, asset_data in data['portfolio'].items():
                                    st.session_state.portfolio[symbol] = asset_data
                            
                            portfolios = load_portfolios()
                            portfolios[st.session_state.username] = st.session_state.portfolio
                            save_portfolios(portfolios)
                            
                            st.markdown("""
                            <div class="success-card">
                                <strong>🎉 Portfolio imported successfully!</strong><br>
                                Your portfolio has been updated with the imported data.
                            </div>
                            """, unsafe_allow_html=True)
                            st.rerun()
                    else:
                        st.error("❌ Invalid portfolio file format.")
                
                elif uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                    
                    required_columns = ['Symbol', 'Shares', 'Asset_Type']
                    if all(col in df.columns for col in required_columns):
                        st.markdown("""
                        <div class="success-card">
                            <strong>✅ Valid CSV file detected!</strong><br>
                            Ready to import your portfolio data.
                        </div>
                        """, unsafe_allow_html=True)
                        
                        with st.expander("👀 Preview Import Data"):
                            st.dataframe(df)
                        
                        csv_import_option = st.selectbox(
                            "Import Option",
                            ["Replace Current Portfolio", "Merge with Current Portfolio"],
                            key="csv_import_option"
                        )
                        
                        if st.button("🚀 Import CSV Portfolio", type="primary"):
                            new_portfolio = {}
                            
                            for _, row in df.iterrows():
                                symbol = row['Symbol']
                                new_portfolio[symbol] = {
                                    'shares': float(row['Shares']),
                                    'asset_type': row['Asset_Type'],
                                    'purchase_price': float(row.get('Purchase_Price', 100.0)),
                                    'added_date': datetime.now().isoformat()
                                }
                            
                            if csv_import_option == "Replace Current Portfolio":
                                st.session_state.portfolio = new_portfolio
                            else:
                                for symbol, asset_data in new_portfolio.items():
                                    st.session_state.portfolio[symbol] = asset_data
                            
                            portfolios = load_portfolios()
                            portfolios[st.session_state.username] = st.session_state.portfolio
                            save_portfolios(portfolios)
                            
                            st.markdown("""
                            <div class="success-card">
                                <strong>🎉 CSV portfolio imported successfully!</strong><br>
                                Your portfolio has been updated with the imported data.
                            </div>
                            """, unsafe_allow_html=True)
                            st.rerun()
                    else:
                        st.error(f"❌ CSV file must contain columns: {', '.join(required_columns)}")
                        
            except Exception as e:
                st.error(f"❌ Error importing file: {str(e)}")

if __name__ == "__main__":
    main(), 'A"{return_pct:+.1f}%",
                'P&L': f"${current_value - invested_value:+,.2f}",
                'Asset Type': data['asset_type'],
                'Weight %': f"{(current_value / metrics['total_current_value']) * 100:.1f}%" if metrics['total_current_value'] > 0 else "0.0%"
            })
    
    if portfolio_data:
        st.markdown("### 📊 Holdings Breakdown")
        df = pd.DataFrame(portfolio_data)
        st.dataframe(df, use_container_width=True, height=400)
        
        if PLOTLY_AVAILABLE:
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                st.markdown("#### 🥧 Portfolio Allocation")
                # Extract numeric values safely
                current_values = []
                symbols = []
                
                for item in portfolio_data:
                    try:
                        value_str = item['Current Value'].replace('$', '').replace(',', '')
                        current_values.append(float(value_str))
                        symbols.append(item['Symbol'])
                    except:
                        continue
                
                if current_values:
                    fig_pie = px.pie(
                        values=current_values,
                        names=symbols,
                        title="Current Holdings Distribution",
                        hole=0.4
                    )
                    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                    fig_pie.update_layout(height=400)
                    st.plotly_chart(fig_pie, use_container_width=True)
            
            with col_chart2:
                st.markdown("#### 📈 Performance by Asset")
                return_values = []
                
                for item in portfolio_data:
                    try:
                        return_str = item['Return %'].replace('%', '').replace('+', '')
                        return_values.append(float(return_str))
                    except:
                        return_values.append(0.0)
                
                if return_values and symbols:
                    fig_returns = px.bar(
                        x=symbols,
                        y=return_values,
                        title="Return % by Holding",
                        color=return_values,
                        color_continuous_scale=['red', 'yellow', 'green']
                    )
                    fig_returns.update_layout(height=400, xaxis_title="Assets", yaxis_title="Return %")
                    st.plotly_chart(fig_returns, use_container_width=True)
        else:
            st.markdown("#### 📊 Simple Visualization")
            current_values = []
            symbols = []
            
            for item in portfolio_data:
                try:
                    value_str = item['Current Value'].replace('$', '').replace(',', '')
                    current_values.append(float(value_str))
                    symbols.append(item['Symbol'])
                except:
                    continue
            
            if current_values and symbols:
                chart_data = pd.DataFrame({
                    'Symbol': symbols,
                    'Current Value': current_values
                })
                st.bar_chart(chart_data.set_index('Symbol'))

def show_asset_management():
    """Display asset management page"""
    
    st.markdown("### 🎯 Manage Portfolio Assets")
    
    if st.session_state.learning_mode:
        st.markdown("""
        <div class="info-card">
            <strong>🎯 Asset Management Guide:</strong><br>
            Use this section to build and maintain your investment portfolio. Add assets, track purchase prices, and manage your holdings.
        </div>
        """, unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["➕ Add Assets", "➖ Remove Assets"])
    
    with tab1:
        st.markdown("### Add New Investment")
        
        col1, col2 = st.columns(2)
        
        with col1:
            popular_assets = get_popular_assets()
            
            # Create categories for better organization
            asset_categories = {
                "🏢 Large Cap Stocks": [k for k in popular_assets.keys() if k in ["AAPL", "GOOGL", "GOOG", "MSFT", "AMZN", "TSLA", "NVDA", "META", "NFLX", "JPM", "JNJ", "V", "WMT", "PG", "UNH", "HD", "MA", "BAC", "ADBE", "CRM", "XOM", "CVX", "KO", "PFE", "INTC", "CSCO", "VZ", "MRK", "ABT", "TMO"]],
                "🚀 Growth & Tech": [k for k in popular_assets.keys() if k in ["AMD", "PYPL", "SHOP", "SQ", "ROKU", "ZM", "SNOW", "PLTR", "CRWD", "NET"]],
                "🏛️ Vanguard ETFs": [k for k in popular_assets.keys() if k.startswith("V") and k in ["VTI", "VEA", "VWO", "VTV", "VUG", "VIG", "VYM", "VXUS", "VOO", "VT", "VB", "VO", "VGT", "VHT", "VFH", "VDE", "VAW", "VIS", "VCR", "VDC", "VPU", "VNQ"]],
                "⚫ BlackRock iShares": [k for k in popular_assets.keys() if k in ["QQQ", "IWM", "EFA", "EEM", "IVV", "IEFA", "IEMG", "IJH", "IJR", "IVW", "IVE", "IWF", "IWD", "ITOT", "IXUS", "IYY", "IWB", "IWV", "ACWI", "ACWX"]],
                "🎯 Sector ETFs": [k for k in popular_assets.keys() if k.startswith("XL") or k in ["IBB", "SOXX", "SKYY", "HACK", "ROBO", "ARKK", "ARKQ", "ARKW", "ARKG", "ARKF"]],
                "🏦 Bonds & Fixed Income": [k for k in popular_assets.keys() if k in ["AGG", "BND", "TLT", "IEF", "SHY", "LQD", "HYG", "JNK", "TIP", "VTEB", "MUB", "EMB", "BNDX", "VGIT", "VGSH", "VGLT", "VCIT", "VCSH", "BSV", "BIV", "BLV"]],
                "💎 Fidelity Funds": [k for k in popular_assets.keys() if k.startswith("F") and len(k) > 2],
                "🥇 Commodities": [k for k in popular_assets.keys() if k in ["GLD", "SLV", "USO", "UNG", "DBA", "DBC", "PDBC", "GSG", "COMT", "IAU", "SGOL", "PPLT", "PALL", "CORN", "WEAT", "SOYB", "CANE", "COW"]],
                "🌍 International": [k for k in popular_assets.keys() if k.startswith("EW") or k in ["FXI", "INDA", "RSX", "ASHR", "MCHI", "KWEB", "YINN"]],
                "₿ Major Cryptocurrencies": [k for k in popular_assets.keys() if k.endswith("-USD") and k in ["BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "ADA-USD", "SOL-USD", "DOGE-USD", "DOT-USD", "MATIC-USD", "SHIB-USD", "LTC-USD", "TRX-USD", "AVAX-USD", "LINK-USD", "ATOM-USD"]],
                "🔗 DeFi & Web3": [k for k in popular_assets.keys() if k.endswith("-USD") and k in ["XLM-USD", "ALGO-USD", "VET-USD", "ICP-USD", "FIL-USD", "SAND-USD", "MANA-USD", "AXS-USD", "ENJ-USD", "CHZ-USD", "FLOW-USD", "THETA-USD", "XTZ-USD", "EGLD-USD", "NEAR-USD"]],
                "📈 Crypto ETFs": [k for k in popular_assets.keys() if k in ["BITO", "BITI", "GBTC", "ETHE", "GDLC", "BITQ", "BLOK", "LEGR"]],
                "💱 Currency ETFs": [k for k in popular_assets.keys() if k.startswith("FX") or k in ["UUP", "UDN"]],
                "💰 Dividend ETFs": [k for k in popular_assets.keys() if k in ["SCHD", "DVY", "VYM", "NOBL", "DGRO", "HDV", "SPHD", "PEY", "RDVY", "FDL"]],
                "🌱 ESG & Clean Energy": [k for k in popular_assets.keys() if k in ["ESG", "ESGU", "ESGD", "ESGE", "VSGX", "ESGV", "SUSA", "DSI", "ICLN", "PBW", "QCLN"]],
                "📊 Volatility & Leverage": [k for k in popular_assets.keys() if k in ["VIX", "UVXY", "SVXY", "VXX", "VIXY", "TQQQ", "SQQQ", "UPRO", "SPXU", "TNA", "TZA", "TECL", "TECS", "FAS", "FAZ"]]
            }
            
            selected_category = st.selectbox(
                "Choose Asset Category",
                [""] + list(asset_categories.keys()),
                help="Browse assets by category for easier selection"
            )
            
            if selected_category:
                available_assets = asset_categories[selected_category]
                selected_popular = st.selectbox(
                    f"Assets in {selected_category}",
                    [""] + available_assets,
                    format_func=lambda x: f"{x} - {popular_assets[x]}" if x else "",
                    help="Select from assets in this category"
                )
            else:
                selected_popular = st.selectbox(
                    "Or search all assets",
                    [""] + list(popular_assets.keys()),
                    format_func=lambda x: f"{x} - {popular_assets[x]}" if x else "",
                    help="Search through all available assets"
                )
            
            custom_symbol = st.text_input(
                "Or Enter Custom Symbol",
                placeholder="e.g., AAPL, BTC-USD, GLD.L (London)",
                help="Enter any valid symbol including international exchanges"
            )
            
            symbol_to_use = ""
            if selected_popular:
                symbol_to_use = selected_popular
            elif custom_symbol:
                symbol_to_use = custom_symbol.upper()
        
        with col2:
            shares = st.number_input(
                "Number of Shares/Units",
                min_value=0.001,
                value=1.0,
                step=0.1,
                help="Enter the quantity you own"
            )
            
            purchase_price = st.number_input(
                "Purchase Price per Share ($)",
                min_value=0.01,
                value=100.0,
                step=0.01,
                help="Enter the average price you paid per share"
            )
            
            # NEW: Currency selection
            currencies = get_currency_list()
            selected_currency = st.selectbox(
                "Currency",
                list(currencies.keys()),
                index=0,  # Default to USD
                format_func=lambda x: f"{x} - {currencies[x]['name']}",
                help="Select the currency for this asset"
            )
            
            asset_type = st.selectbox(
                "Asset Type",
                ["Stock", "ETF", "Cryptocurrency", "Bond", "Commodity", "Index Fund", "Mutual Fund", "REIT", "Other"]
            )
            
            if st.session_state.learning_mode:
                st.markdown("""
                <div class="info-card">
                    <strong>💡 Pro Tip:</strong><br>
                    • Select the correct currency for international assets<br>
                    • Purchase price should be in the selected currency<br>
                    • The system will convert to your base currency for analysis
                </div>
                """, unsafe_allow_html=True)
        
        if symbol_to_use:
            asset_info = fetch_asset_data(symbol_to_use)
            if asset_info:
                current_price = asset_info['current_price']
                
                # Convert prices for display if different currency
                if selected_currency != "USD":
                    exchange_rate = get_exchange_rate("USD", selected_currency)
                    current_price_display = current_price * exchange_rate
                    purchase_price_display = purchase_price
                else:
                    current_price_display = current_price
                    purchase_price_display = purchase_price
                
                potential_return = ((current_price_display - purchase_price_display) / purchase_price_display) * 100 if purchase_price_display > 0 else 0
                
                # Create three columns for metrics display
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                
                with metric_col1:
                    currency_symbol = currencies[selected_currency]["symbol"]
                    st.metric("Current Price", f"{currency_symbol}{current_price_display:.2f}")
                with metric_col2:
                    st.metric("Your Purchase Price", f"{currency_symbol}{purchase_price_display:.2f}")
                with metric_col3:
                    st.metric("Potential Return", f"{potential_return:+.1f}%", 
                             delta=f"{currency_symbol}{(current_price_display - purchase_price_display) * shares:+.2f}")
                
                if st.button("✅ Add to Portfolio", type="primary", use_container_width=True):
                    st.session_state.portfolio[symbol_to_use] = {
                        'shares': shares,
                        'asset_type': asset_type,
                        'purchase_price': purchase_price,
                        'currency': selected_currency,  # NEW: Store currency
                        'added_date': datetime.now().isoformat()
                    }
                    
                    portfolios = load_portfolios()
                    portfolios[st.session_state.username] = st.session_state.portfolio
                    save_portfolios(portfolios)
                    
                    st.markdown("""
                    <div class="success-card">
                        <strong>🎉 Asset Added Successfully!</strong><br>
                        Your portfolio metrics are being recalculated with the new asset allocation and currency conversion.
                    </div>
                    """, unsafe_allow_html=True)
                    st.rerun()
            else:
                st.error(f"❌ Could not find asset data for '{symbol_to_use}'. Please check the symbol.")
    
    with tab2:
        st.markdown("### Remove Assets from Portfolio")
        
        if st.session_state.portfolio:
            current_values = []
            for symbol, data in st.session_state.portfolio.items():
                asset_info = fetch_asset_data(symbol)
                if asset_info:
                    current_value = data['shares'] * asset_info['current_price']
                    current_values.append({
                        'Symbol': symbol,
                        'Shares': f"{data['shares']:.3f}",
                        'Current Value': f"${current_value:,.2f}",
                        'Asset Type': data['asset_type']
                    })
            
            if current_values:
                df_current = pd.DataFrame(current_values)
                st.dataframe(df_current, hide_index=True)
            
            assets_to_remove = st.multiselect(
                "Select assets to remove:",
                list(st.session_state.portfolio.keys()),
                help="Choose one or more assets to remove from your portfolio"
            )
            
            if assets_to_remove:
                st.warning(f"⚠️ You are about to remove {len(assets_to_remove)} asset(s) from your portfolio.")
                
                if st.button("🗑️ Remove Selected Assets", type="secondary"):
                    for asset in assets_to_remove:
                        del st.session_state.portfolio[asset]
                    
                    portfolios = load_portfolios()
                    portfolios[st.session_state.username] = st.session_state.portfolio
                    save_portfolios(portfolios)
                    
                    st.markdown("""
                    <div class="success-card">
                        <strong>✅ Assets Removed Successfully!</strong><br>
                        Your portfolio has been updated and metrics recalculated.
                    </div>
                    """, unsafe_allow_html=True)
                    st.rerun()
        else:
            st.markdown("""
            <div class="info-card">
                <strong>📝 No assets to remove</strong><br>
                Your portfolio is currently empty. Add some investments first!
            </div>
            """, unsafe_allow_html=True)

def show_analytics_dashboard():
    """Display analytics dashboard"""
    
    st.markdown("### 📊 Advanced Analytics Dashboard")
    
    if st.session_state.learning_mode:
        st.markdown("""
        <div class="info-card">
            <strong>📈 Analytics Dashboard Guide:</strong><br>
            This dashboard provides professional-grade analysis tools including performance metrics, technical analysis, and AI-powered insights.
        </div>
        """, unsafe_allow_html=True)
    
    if not st.session_state.portfolio:
        st.warning("⚠️ Add assets to your portfolio to see advanced analytics.")
        return
    
    metrics = calculate_portfolio_metrics_advanced(st.session_state.portfolio)
    
    st.markdown("### 📈 Advanced Performance Metrics")
    
    perf_col1, perf_col2, perf_col3, perf_col4, perf_col5 = st.columns(5)
    
    with perf_col1:
        st.metric(
            "Annualized Return", 
            f"{metrics['annualized_return']:+.1f}%",
            help="Expected annual return based on current performance"
        )
    
    with perf_col2:
        st.metric(
            "Portfolio Beta", 
            f"{metrics['beta']:.2f}",
            delta="Market Risk" if metrics['beta'] > 1 else "Lower Risk",
            help="Portfolio volatility relative to market"
        )
    
    with perf_col3:
        st.metric(
            "Sharpe Ratio", 
            f"{metrics['sharpe_ratio']:.2f}",
            delta="Excellent" if metrics['sharpe_ratio'] > 1 else "Good" if metrics['sharpe_ratio'] > 0.5 else "Needs Improvement",
            help="Risk-adjusted return measure"
        )
    
    with perf_col4:
        st.metric(
            "Max Drawdown", 
            f"{metrics['max_drawdown']:.1f}%",
            delta="Risk Level",
            delta_color="inverse",
            help="Largest peak-to-trough decline"
        )
    
    with perf_col5:
        st.metric(
            "Portfolio Volatility", 
            f"{metrics['volatility']:.1f}%",
            help="Measure of price fluctuation"
        )
    
    # Technical analysis for top holdings
    st.markdown("### 🔍 Technical Analysis - Top Holdings")
    
    portfolio_values = []
    for symbol, data in st.session_state.portfolio.items():
        asset_info = fetch_asset_data(symbol)
        if asset_info:
            value = data['shares'] * asset_info['current_price']
            portfolio_values.append((symbol, value))
    
    portfolio_values.sort(key=lambda x: x[1], reverse=True)
    top_holdings = [item[0] for item in portfolio_values[:3]]
    
    for i, symbol in enumerate(top_holdings):
        with st.expander(f"📊 {symbol} - Technical Analysis", expanded=(i == 0)):
            indicators = calculate_technical_indicators(symbol)
            
            if indicators is not None:
                tech_col1, tech_col2, tech_col3, tech_col4 = st.columns(4)
                
                current_price = indicators['Close'].iloc[-1]
                ma20 = indicators['MA_20'].iloc[-1]
                current_rsi = indicators['RSI'].iloc[-1]
                current_macd = indicators['MACD'].iloc[-1]
                current_signal = indicators['MACD_Signal'].iloc[-1]
                
                with tech_col1:
                    price_trend = "Bullish" if current_price > ma20 else "Bearish"
                    st.metric("Price Trend", price_trend, f"${current_price:.2f}")
                
                with tech_col2:
                    rsi_status = "Overbought" if current_rsi > 70 else "Oversold" if current_rsi < 30 else "Neutral"
                    rsi_color = "🔴" if current_rsi > 70 else "🟢" if current_rsi < 30 else "🟡"
                    st.metric("RSI Signal", f"{rsi_color} {rsi_status}", f"{current_rsi:.1f}")
                
                with tech_col3:
                    macd_trend = "Bullish" if current_macd > current_signal else "Bearish"
                    macd_color = "🟢" if current_macd > current_signal else "🔴"
                    st.metric("MACD Signal", f"{macd_color} {macd_trend}")
                
                with tech_col4:
                    support_level = current_price * 0.95
                    resistance_level = current_price * 1.05
                    st.metric("Support/Resistance", f"${support_level:.2f} / ${resistance_level:.2f}")
                
                if PLOTLY_AVAILABLE:
                    fig = make_subplots(
                        rows=3, cols=1,
                        subplot_titles=(
                            f'{symbol} - Price & Moving Averages',
                            'Relative Strength Index (RSI)',
                            'MACD & Signal Line'
                        ),
                        vertical_spacing=0.08,
                        row_heights=[0.5, 0.25, 0.25]
                    )
                    
                    fig.add_trace(
                        go.Scatter(x=indicators.index, y=indicators['Close'], name='Price', line=dict(color='blue', width=2)),
                        row=1, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=indicators.index, y=indicators['MA_20'], name='MA 20', line=dict(color='orange')),
                        row=1, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=indicators.index, y=indicators['MA_50'], name='MA 50', line=dict(color='red')),
                        row=1, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(x=indicators.index, y=indicators['RSI'], name='RSI', line=dict(color='purple')),
                        row=2, col=1
                    )
                    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
                    
                    fig.add_trace(
                        go.Scatter(x=indicators.index, y=indicators['MACD'], name='MACD', line=dict(color='blue')),
                        row=3, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=indicators.index, y=indicators['MACD_Signal'], name='Signal', line=dict(color='red')),
                        row=3, col=1
                    )
                    
                    fig.update_layout(height=700, showlegend=True, title_text=f"{symbol} Technical Analysis")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.markdown("#### 📈 Price Chart")
                    st.line_chart(indicators[['Close', 'MA_20', 'MA_50']])
                    
                    simple_col1, simple_col2 = st.columns(2)
                    with simple_col1:
                        st.markdown("#### RSI")
                        st.line_chart(indicators['RSI'])
                    with simple_col2:
                        st.markdown("#### MACD")
                        st.line_chart(indicators[['MACD', 'MACD_Signal']])
    
    # Investment suggestions
    st.markdown("### 🤖 AI-Powered Investment Insights")
    suggestions = generate_investment_suggestions(st.session_state.portfolio)
    
    enhanced_suggestions = []
    
    if metrics['total_return'] > 10:
        enhanced_suggestions.append({
            'type': 'success',
            'message': f'🎉 Excellent performance! Your portfolio is up {metrics["total_return"]:.1f}%. Consider taking some profits and rebalancing.'
        })
    elif metrics['total_return'] < -5:
        enhanced_suggestions.append({
            'type': 'warning',
            'message': f'📉 Portfolio is down {abs(metrics["total_return"]):.1f}%. Review underperforming assets and consider strategic rebalancing.'
        })
    
    if metrics['volatility'] > 25:
        enhanced_suggestions.append({
            'type': 'warning',
            'message': f'⚡ High volatility detected ({metrics["volatility"]:.1f}%). Consider adding stable assets like bonds or dividend stocks.'
        })
    
    all_suggestions = enhanced_suggestions + suggestions
    
    for suggestion in all_suggestions[:6]:
        if suggestion['type'] == 'diversification':
            st.markdown(f"""
            <div class="info-card">
                <strong>🎯 Diversification:</strong> {suggestion['message']}
            </div>
            """, unsafe_allow_html=True)
        elif suggestion['type'] == 'rebalancing':
            st.markdown(f"""
            <div class="warning-card">
                <strong>⚖️ Rebalancing:</strong> {suggestion['message']}
            </div>
            """, unsafe_allow_html=True)
        elif suggestion['type'] == 'opportunity':
            st.markdown(f"""
            <div class="success-card">
                <strong>🚀 Opportunity:</strong> {suggestion['message']}
            </div>
            """, unsafe_allow_html=True)
        elif suggestion['type'] == 'success':
            st.markdown(f"""
            <div class="success-card">
                <strong>✅ Success:</strong> {suggestion['message']}
            </div>
            """, unsafe_allow_html=True)
        elif suggestion['type'] == 'warning':
            st.markdown(f"""
            <div class="warning-card">
                <strong>⚠️ Warning:</strong> {suggestion['message']}
            </div>
            """, unsafe_allow_html=True)

def show_export_import():
    """Display export/import functionality"""
    
    st.markdown("### 📁 Export & Import Portfolio")
    
    if st.session_state.learning_mode:
        st.markdown("""
        <div class="info-card">
            <strong>📁 Export/Import Guide:</strong><br>
            Backup and restore your portfolio data. Export as JSON or CSV for safekeeping, or import previously saved portfolios.
        </div>
        """, unsafe_allow_html=True)
    
    export_tab, import_tab = st.tabs(["📤 Export Portfolio", "📥 Import Portfolio"])
    
    with export_tab:
        st.markdown("### Export Your Portfolio Data")
        
        if not st.session_state.portfolio:
            st.markdown("""
            <div class="warning-card">
                <strong>📝 No portfolio data to export</strong><br>
                Add some investments to your portfolio first, then return here to export your data.
            </div>
            """, unsafe_allow_html=True)
            return
        
        export_format = st.selectbox("Choose Export Format", ["JSON", "CSV"])
        
        if export_format == "JSON":
            export_data = {
                'username': st.session_state.username,
                'export_date': datetime.now().isoformat(),
                'portfolio': st.session_state.portfolio,
                'version': '2.0'
            }
            
            json_string = json.dumps(export_data, indent=2)
            
            st.download_button(
                label="📥 Download Portfolio (JSON)",
                data=json_string,
                file_name=f"portfolio_{st.session_state.username}_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )
            
            with st.expander("👀 Preview JSON Data"):
                st.code(json_string, language="json")
        
        else:  # CSV format
            csv_data = []
            for symbol, data in st.session_state.portfolio.items():
                asset_info = fetch_asset_data(symbol)
                if asset_info:
                    csv_data.append({
                        'Symbol': symbol,
                        'Name': asset_info['name'],
                        'Shares': data['shares'],
                        'Asset_Type': data['asset_type'],
                        'Purchase_Price': data.get('purchase_price', asset_info['current_price']),
                        'Current_Price': asset_info['current_price'],
                        'Current_Value': data['shares'] * asset_info['current_price'],
                        'Added_Date': data['added_date']
                    })
            
            if csv_data:
                df = pd.DataFrame(csv_data)
                csv_string = df.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download Portfolio (CSV)",
                    data=csv_string,
                    file_name=f"portfolio_{st.session_state.username}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
                
                with st.expander("👀 Preview CSV Data"):
                    st.dataframe(df)
    
    with import_tab:
        st.markdown("### Import Portfolio Data")
        
        uploaded_file = st.file_uploader(
            "Choose a portfolio file",
            type=['json', 'csv'],
            help="Upload a previously exported portfolio file"
        )
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.json'):
                    content = uploaded_file.read()
                    data = json.loads(content)
                    
                    if 'portfolio' in data:
                        st.markdown("""
                        <div class="success-card">
                            <strong>✅ Valid portfolio file detected!</strong><br>
                            Ready to import your portfolio data.
                        </div>
                        """, unsafe_allow_html=True)
                        
                        with st.expander("👀 Preview Import Data"):
                            st.json(data['portfolio'])
                        
                        import_option = st.selectbox(
                            "Import Option",
                            ["Replace Current Portfolio", "Merge with Current Portfolio"]
                        )
                        
                        if st.button("🚀 Import Portfolio", type="primary"):
                            if import_option == "Replace Current Portfolio":
                                st.session_state.portfolio = data['portfolio']
                            else:
                                for symbol, asset_data in data['portfolio'].items():
                                    st.session_state.portfolio[symbol] = asset_data
                            
                            portfolios = load_portfolios()
                            portfolios[st.session_state.username] = st.session_state.portfolio
                            save_portfolios(portfolios)
                            
                            st.markdown("""
                            <div class="success-card">
                                <strong>🎉 Portfolio imported successfully!</strong><br>
                                Your portfolio has been updated with the imported data.
                            </div>
                            """, unsafe_allow_html=True)
                            st.rerun()
                    else:
                        st.error("❌ Invalid portfolio file format.")
                
                elif uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                    
                    required_columns = ['Symbol', 'Shares', 'Asset_Type']
                    if all(col in df.columns for col in required_columns):
                        st.markdown("""
                        <div class="success-card">
                            <strong>✅ Valid CSV file detected!</strong><br>
                            Ready to import your portfolio data.
                        </div>
                        """, unsafe_allow_html=True)
                        
                        with st.expander("👀 Preview Import Data"):
                            st.dataframe(df)
                        
                        csv_import_option = st.selectbox(
                            "Import Option",
                            ["Replace Current Portfolio", "Merge with Current Portfolio"],
                            key="csv_import_option"
                        )
                        
                        if st.button("🚀 Import CSV Portfolio", type="primary"):
                            new_portfolio = {}
                            
                            for _, row in df.iterrows():
                                symbol = row['Symbol']
                                new_portfolio[symbol] = {
                                    'shares': float(row['Shares']),
                                    'asset_type': row['Asset_Type'],
                                    'purchase_price': float(row.get('Purchase_Price', 100.0)),
                                    'added_date': datetime.now().isoformat()
                                }
                            
                            if csv_import_option == "Replace Current Portfolio":
                                st.session_state.portfolio = new_portfolio
                            else:
                                for symbol, asset_data in new_portfolio.items():
                                    st.session_state.portfolio[symbol] = asset_data
                            
                            portfolios = load_portfolios()
                            portfolios[st.session_state.username] = st.session_state.portfolio
                            save_portfolios(portfolios)
                            
                            st.markdown("""
                            <div class="success-card">
                                <strong>🎉 CSV portfolio imported successfully!</strong><br>
                                Your portfolio has been updated with the imported data.
                            </div>
                            """, unsafe_allow_html=True)
                            st.rerun()
                    else:
                        st.error(f"❌ CSV file must contain columns: {', '.join(required_columns)}")
                        
            except Exception as e:
                st.error(f"❌ Error importing file: {str(e)}")

if __name__ == "__main__":
    main(), 'CHF', 'R"{return_pct:+.1f}%",
                'P&L': f"${current_value - invested_value:+,.2f}",
                'Asset Type': data['asset_type'],
                'Weight %': f"{(current_value / metrics['total_current_value']) * 100:.1f}%" if metrics['total_current_value'] > 0 else "0.0%"
            })
    
    if portfolio_data:
        st.markdown("### 📊 Holdings Breakdown")
        df = pd.DataFrame(portfolio_data)
        st.dataframe(df, use_container_width=True, height=400)
        
        if PLOTLY_AVAILABLE:
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                st.markdown("#### 🥧 Portfolio Allocation")
                # Extract numeric values safely
                current_values = []
                symbols = []
                
                for item in portfolio_data:
                    try:
                        value_str = item['Current Value'].replace('$', '').replace(',', '')
                        current_values.append(float(value_str))
                        symbols.append(item['Symbol'])
                    except:
                        continue
                
                if current_values:
                    fig_pie = px.pie(
                        values=current_values,
                        names=symbols,
                        title="Current Holdings Distribution",
                        hole=0.4
                    )
                    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                    fig_pie.update_layout(height=400)
                    st.plotly_chart(fig_pie, use_container_width=True)
            
            with col_chart2:
                st.markdown("#### 📈 Performance by Asset")
                return_values = []
                
                for item in portfolio_data:
                    try:
                        return_str = item['Return %'].replace('%', '').replace('+', '')
                        return_values.append(float(return_str))
                    except:
                        return_values.append(0.0)
                
                if return_values and symbols:
                    fig_returns = px.bar(
                        x=symbols,
                        y=return_values,
                        title="Return % by Holding",
                        color=return_values,
                        color_continuous_scale=['red', 'yellow', 'green']
                    )
                    fig_returns.update_layout(height=400, xaxis_title="Assets", yaxis_title="Return %")
                    st.plotly_chart(fig_returns, use_container_width=True)
        else:
            st.markdown("#### 📊 Simple Visualization")
            current_values = []
            symbols = []
            
            for item in portfolio_data:
                try:
                    value_str = item['Current Value'].replace('$', '').replace(',', '')
                    current_values.append(float(value_str))
                    symbols.append(item['Symbol'])
                except:
                    continue
            
            if current_values and symbols:
                chart_data = pd.DataFrame({
                    'Symbol': symbols,
                    'Current Value': current_values
                })
                st.bar_chart(chart_data.set_index('Symbol'))

def show_asset_management():
    """Display asset management page"""
    
    st.markdown("### 🎯 Manage Portfolio Assets")
    
    if st.session_state.learning_mode:
        st.markdown("""
        <div class="info-card">
            <strong>🎯 Asset Management Guide:</strong><br>
            Use this section to build and maintain your investment portfolio. Add assets, track purchase prices, and manage your holdings.
        </div>
        """, unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["➕ Add Assets", "➖ Remove Assets"])
    
    with tab1:
        st.markdown("### Add New Investment")
        
        col1, col2 = st.columns(2)
        
        with col1:
            popular_assets = get_popular_assets()
            
            # Create categories for better organization
            asset_categories = {
                "🏢 Large Cap Stocks": [k for k in popular_assets.keys() if k in ["AAPL", "GOOGL", "GOOG", "MSFT", "AMZN", "TSLA", "NVDA", "META", "NFLX", "JPM", "JNJ", "V", "WMT", "PG", "UNH", "HD", "MA", "BAC", "ADBE", "CRM", "XOM", "CVX", "KO", "PFE", "INTC", "CSCO", "VZ", "MRK", "ABT", "TMO"]],
                "🚀 Growth & Tech": [k for k in popular_assets.keys() if k in ["AMD", "PYPL", "SHOP", "SQ", "ROKU", "ZM", "SNOW", "PLTR", "CRWD", "NET"]],
                "🏛️ Vanguard ETFs": [k for k in popular_assets.keys() if k.startswith("V") and k in ["VTI", "VEA", "VWO", "VTV", "VUG", "VIG", "VYM", "VXUS", "VOO", "VT", "VB", "VO", "VGT", "VHT", "VFH", "VDE", "VAW", "VIS", "VCR", "VDC", "VPU", "VNQ"]],
                "⚫ BlackRock iShares": [k for k in popular_assets.keys() if k in ["QQQ", "IWM", "EFA", "EEM", "IVV", "IEFA", "IEMG", "IJH", "IJR", "IVW", "IVE", "IWF", "IWD", "ITOT", "IXUS", "IYY", "IWB", "IWV", "ACWI", "ACWX"]],
                "🎯 Sector ETFs": [k for k in popular_assets.keys() if k.startswith("XL") or k in ["IBB", "SOXX", "SKYY", "HACK", "ROBO", "ARKK", "ARKQ", "ARKW", "ARKG", "ARKF"]],
                "🏦 Bonds & Fixed Income": [k for k in popular_assets.keys() if k in ["AGG", "BND", "TLT", "IEF", "SHY", "LQD", "HYG", "JNK", "TIP", "VTEB", "MUB", "EMB", "BNDX", "VGIT", "VGSH", "VGLT", "VCIT", "VCSH", "BSV", "BIV", "BLV"]],
                "💎 Fidelity Funds": [k for k in popular_assets.keys() if k.startswith("F") and len(k) > 2],
                "🥇 Commodities": [k for k in popular_assets.keys() if k in ["GLD", "SLV", "USO", "UNG", "DBA", "DBC", "PDBC", "GSG", "COMT", "IAU", "SGOL", "PPLT", "PALL", "CORN", "WEAT", "SOYB", "CANE", "COW"]],
                "🌍 International": [k for k in popular_assets.keys() if k.startswith("EW") or k in ["FXI", "INDA", "RSX", "ASHR", "MCHI", "KWEB", "YINN"]],
                "₿ Major Cryptocurrencies": [k for k in popular_assets.keys() if k.endswith("-USD") and k in ["BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "ADA-USD", "SOL-USD", "DOGE-USD", "DOT-USD", "MATIC-USD", "SHIB-USD", "LTC-USD", "TRX-USD", "AVAX-USD", "LINK-USD", "ATOM-USD"]],
                "🔗 DeFi & Web3": [k for k in popular_assets.keys() if k.endswith("-USD") and k in ["XLM-USD", "ALGO-USD", "VET-USD", "ICP-USD", "FIL-USD", "SAND-USD", "MANA-USD", "AXS-USD", "ENJ-USD", "CHZ-USD", "FLOW-USD", "THETA-USD", "XTZ-USD", "EGLD-USD", "NEAR-USD"]],
                "📈 Crypto ETFs": [k for k in popular_assets.keys() if k in ["BITO", "BITI", "GBTC", "ETHE", "GDLC", "BITQ", "BLOK", "LEGR"]],
                "💱 Currency ETFs": [k for k in popular_assets.keys() if k.startswith("FX") or k in ["UUP", "UDN"]],
                "💰 Dividend ETFs": [k for k in popular_assets.keys() if k in ["SCHD", "DVY", "VYM", "NOBL", "DGRO", "HDV", "SPHD", "PEY", "RDVY", "FDL"]],
                "🌱 ESG & Clean Energy": [k for k in popular_assets.keys() if k in ["ESG", "ESGU", "ESGD", "ESGE", "VSGX", "ESGV", "SUSA", "DSI", "ICLN", "PBW", "QCLN"]],
                "📊 Volatility & Leverage": [k for k in popular_assets.keys() if k in ["VIX", "UVXY", "SVXY", "VXX", "VIXY", "TQQQ", "SQQQ", "UPRO", "SPXU", "TNA", "TZA", "TECL", "TECS", "FAS", "FAZ"]]
            }
            
            selected_category = st.selectbox(
                "Choose Asset Category",
                [""] + list(asset_categories.keys()),
                help="Browse assets by category for easier selection"
            )
            
            if selected_category:
                available_assets = asset_categories[selected_category]
                selected_popular = st.selectbox(
                    f"Assets in {selected_category}",
                    [""] + available_assets,
                    format_func=lambda x: f"{x} - {popular_assets[x]}" if x else "",
                    help="Select from assets in this category"
                )
            else:
                selected_popular = st.selectbox(
                    "Or search all assets",
                    [""] + list(popular_assets.keys()),
                    format_func=lambda x: f"{x} - {popular_assets[x]}" if x else "",
                    help="Search through all available assets"
                )
            
            custom_symbol = st.text_input(
                "Or Enter Custom Symbol",
                placeholder="e.g., AAPL, BTC-USD, GLD.L (London)",
                help="Enter any valid symbol including international exchanges"
            )
            
            symbol_to_use = ""
            if selected_popular:
                symbol_to_use = selected_popular
            elif custom_symbol:
                symbol_to_use = custom_symbol.upper()
        
        with col2:
            shares = st.number_input(
                "Number of Shares/Units",
                min_value=0.001,
                value=1.0,
                step=0.1,
                help="Enter the quantity you own"
            )
            
            purchase_price = st.number_input(
                "Purchase Price per Share ($)",
                min_value=0.01,
                value=100.0,
                step=0.01,
                help="Enter the average price you paid per share"
            )
            
            # NEW: Currency selection
            currencies = get_currency_list()
            selected_currency = st.selectbox(
                "Currency",
                list(currencies.keys()),
                index=0,  # Default to USD
                format_func=lambda x: f"{x} - {currencies[x]['name']}",
                help="Select the currency for this asset"
            )
            
            asset_type = st.selectbox(
                "Asset Type",
                ["Stock", "ETF", "Cryptocurrency", "Bond", "Commodity", "Index Fund", "Mutual Fund", "REIT", "Other"]
            )
            
            if st.session_state.learning_mode:
                st.markdown("""
                <div class="info-card">
                    <strong>💡 Pro Tip:</strong><br>
                    • Select the correct currency for international assets<br>
                    • Purchase price should be in the selected currency<br>
                    • The system will convert to your base currency for analysis
                </div>
                """, unsafe_allow_html=True)
        
        if symbol_to_use:
            asset_info = fetch_asset_data(symbol_to_use)
            if asset_info:
                current_price = asset_info['current_price']
                
                # Convert prices for display if different currency
                if selected_currency != "USD":
                    exchange_rate = get_exchange_rate("USD", selected_currency)
                    current_price_display = current_price * exchange_rate
                    purchase_price_display = purchase_price
                else:
                    current_price_display = current_price
                    purchase_price_display = purchase_price
                
                potential_return = ((current_price_display - purchase_price_display) / purchase_price_display) * 100 if purchase_price_display > 0 else 0
                
                # Create three columns for metrics display
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                
                with metric_col1:
                    currency_symbol = currencies[selected_currency]["symbol"]
                    st.metric("Current Price", f"{currency_symbol}{current_price_display:.2f}")
                with metric_col2:
                    st.metric("Your Purchase Price", f"{currency_symbol}{purchase_price_display:.2f}")
                with metric_col3:
                    st.metric("Potential Return", f"{potential_return:+.1f}%", 
                             delta=f"{currency_symbol}{(current_price_display - purchase_price_display) * shares:+.2f}")
                
                if st.button("✅ Add to Portfolio", type="primary", use_container_width=True):
                    st.session_state.portfolio[symbol_to_use] = {
                        'shares': shares,
                        'asset_type': asset_type,
                        'purchase_price': purchase_price,
                        'currency': selected_currency,  # NEW: Store currency
                        'added_date': datetime.now().isoformat()
                    }
                    
                    portfolios = load_portfolios()
                    portfolios[st.session_state.username] = st.session_state.portfolio
                    save_portfolios(portfolios)
                    
                    st.markdown("""
                    <div class="success-card">
                        <strong>🎉 Asset Added Successfully!</strong><br>
                        Your portfolio metrics are being recalculated with the new asset allocation and currency conversion.
                    </div>
                    """, unsafe_allow_html=True)
                    st.rerun()
            else:
                st.error(f"❌ Could not find asset data for '{symbol_to_use}'. Please check the symbol.")
    
    with tab2:
        st.markdown("### Remove Assets from Portfolio")
        
        if st.session_state.portfolio:
            current_values = []
            for symbol, data in st.session_state.portfolio.items():
                asset_info = fetch_asset_data(symbol)
                if asset_info:
                    current_value = data['shares'] * asset_info['current_price']
                    current_values.append({
                        'Symbol': symbol,
                        'Shares': f"{data['shares']:.3f}",
                        'Current Value': f"${current_value:,.2f}",
                        'Asset Type': data['asset_type']
                    })
            
            if current_values:
                df_current = pd.DataFrame(current_values)
                st.dataframe(df_current, hide_index=True)
            
            assets_to_remove = st.multiselect(
                "Select assets to remove:",
                list(st.session_state.portfolio.keys()),
                help="Choose one or more assets to remove from your portfolio"
            )
            
            if assets_to_remove:
                st.warning(f"⚠️ You are about to remove {len(assets_to_remove)} asset(s) from your portfolio.")
                
                if st.button("🗑️ Remove Selected Assets", type="secondary"):
                    for asset in assets_to_remove:
                        del st.session_state.portfolio[asset]
                    
                    portfolios = load_portfolios()
                    portfolios[st.session_state.username] = st.session_state.portfolio
                    save_portfolios(portfolios)
                    
                    st.markdown("""
                    <div class="success-card">
                        <strong>✅ Assets Removed Successfully!</strong><br>
                        Your portfolio has been updated and metrics recalculated.
                    </div>
                    """, unsafe_allow_html=True)
                    st.rerun()
        else:
            st.markdown("""
            <div class="info-card">
                <strong>📝 No assets to remove</strong><br>
                Your portfolio is currently empty. Add some investments first!
            </div>
            """, unsafe_allow_html=True)

def show_analytics_dashboard():
    """Display analytics dashboard"""
    
    st.markdown("### 📊 Advanced Analytics Dashboard")
    
    if st.session_state.learning_mode:
        st.markdown("""
        <div class="info-card">
            <strong>📈 Analytics Dashboard Guide:</strong><br>
            This dashboard provides professional-grade analysis tools including performance metrics, technical analysis, and AI-powered insights.
        </div>
        """, unsafe_allow_html=True)
    
    if not st.session_state.portfolio:
        st.warning("⚠️ Add assets to your portfolio to see advanced analytics.")
        return
    
    metrics = calculate_portfolio_metrics_advanced(st.session_state.portfolio)
    
    st.markdown("### 📈 Advanced Performance Metrics")
    
    perf_col1, perf_col2, perf_col3, perf_col4, perf_col5 = st.columns(5)
    
    with perf_col1:
        st.metric(
            "Annualized Return", 
            f"{metrics['annualized_return']:+.1f}%",
            help="Expected annual return based on current performance"
        )
    
    with perf_col2:
        st.metric(
            "Portfolio Beta", 
            f"{metrics['beta']:.2f}",
            delta="Market Risk" if metrics['beta'] > 1 else "Lower Risk",
            help="Portfolio volatility relative to market"
        )
    
    with perf_col3:
        st.metric(
            "Sharpe Ratio", 
            f"{metrics['sharpe_ratio']:.2f}",
            delta="Excellent" if metrics['sharpe_ratio'] > 1 else "Good" if metrics['sharpe_ratio'] > 0.5 else "Needs Improvement",
            help="Risk-adjusted return measure"
        )
    
    with perf_col4:
        st.metric(
            "Max Drawdown", 
            f"{metrics['max_drawdown']:.1f}%",
            delta="Risk Level",
            delta_color="inverse",
            help="Largest peak-to-trough decline"
        )
    
    with perf_col5:
        st.metric(
            "Portfolio Volatility", 
            f"{metrics['volatility']:.1f}%",
            help="Measure of price fluctuation"
        )
    
    # Technical analysis for top holdings
    st.markdown("### 🔍 Technical Analysis - Top Holdings")
    
    portfolio_values = []
    for symbol, data in st.session_state.portfolio.items():
        asset_info = fetch_asset_data(symbol)
        if asset_info:
            value = data['shares'] * asset_info['current_price']
            portfolio_values.append((symbol, value))
    
    portfolio_values.sort(key=lambda x: x[1], reverse=True)
    top_holdings = [item[0] for item in portfolio_values[:3]]
    
    for i, symbol in enumerate(top_holdings):
        with st.expander(f"📊 {symbol} - Technical Analysis", expanded=(i == 0)):
            indicators = calculate_technical_indicators(symbol)
            
            if indicators is not None:
                tech_col1, tech_col2, tech_col3, tech_col4 = st.columns(4)
                
                current_price = indicators['Close'].iloc[-1]
                ma20 = indicators['MA_20'].iloc[-1]
                current_rsi = indicators['RSI'].iloc[-1]
                current_macd = indicators['MACD'].iloc[-1]
                current_signal = indicators['MACD_Signal'].iloc[-1]
                
                with tech_col1:
                    price_trend = "Bullish" if current_price > ma20 else "Bearish"
                    st.metric("Price Trend", price_trend, f"${current_price:.2f}")
                
                with tech_col2:
                    rsi_status = "Overbought" if current_rsi > 70 else "Oversold" if current_rsi < 30 else "Neutral"
                    rsi_color = "🔴" if current_rsi > 70 else "🟢" if current_rsi < 30 else "🟡"
                    st.metric("RSI Signal", f"{rsi_color} {rsi_status}", f"{current_rsi:.1f}")
                
                with tech_col3:
                    macd_trend = "Bullish" if current_macd > current_signal else "Bearish"
                    macd_color = "🟢" if current_macd > current_signal else "🔴"
                    st.metric("MACD Signal", f"{macd_color} {macd_trend}")
                
                with tech_col4:
                    support_level = current_price * 0.95
                    resistance_level = current_price * 1.05
                    st.metric("Support/Resistance", f"${support_level:.2f} / ${resistance_level:.2f}")
                
                if PLOTLY_AVAILABLE:
                    fig = make_subplots(
                        rows=3, cols=1,
                        subplot_titles=(
                            f'{symbol} - Price & Moving Averages',
                            'Relative Strength Index (RSI)',
                            'MACD & Signal Line'
                        ),
                        vertical_spacing=0.08,
                        row_heights=[0.5, 0.25, 0.25]
                    )
                    
                    fig.add_trace(
                        go.Scatter(x=indicators.index, y=indicators['Close'], name='Price', line=dict(color='blue', width=2)),
                        row=1, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=indicators.index, y=indicators['MA_20'], name='MA 20', line=dict(color='orange')),
                        row=1, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=indicators.index, y=indicators['MA_50'], name='MA 50', line=dict(color='red')),
                        row=1, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(x=indicators.index, y=indicators['RSI'], name='RSI', line=dict(color='purple')),
                        row=2, col=1
                    )
                    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
                    
                    fig.add_trace(
                        go.Scatter(x=indicators.index, y=indicators['MACD'], name='MACD', line=dict(color='blue')),
                        row=3, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=indicators.index, y=indicators['MACD_Signal'], name='Signal', line=dict(color='red')),
                        row=3, col=1
                    )
                    
                    fig.update_layout(height=700, showlegend=True, title_text=f"{symbol} Technical Analysis")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.markdown("#### 📈 Price Chart")
                    st.line_chart(indicators[['Close', 'MA_20', 'MA_50']])
                    
                    simple_col1, simple_col2 = st.columns(2)
                    with simple_col1:
                        st.markdown("#### RSI")
                        st.line_chart(indicators['RSI'])
                    with simple_col2:
                        st.markdown("#### MACD")
                        st.line_chart(indicators[['MACD', 'MACD_Signal']])
    
    # Investment suggestions
    st.markdown("### 🤖 AI-Powered Investment Insights")
    suggestions = generate_investment_suggestions(st.session_state.portfolio)
    
    enhanced_suggestions = []
    
    if metrics['total_return'] > 10:
        enhanced_suggestions.append({
            'type': 'success',
            'message': f'🎉 Excellent performance! Your portfolio is up {metrics["total_return"]:.1f}%. Consider taking some profits and rebalancing.'
        })
    elif metrics['total_return'] < -5:
        enhanced_suggestions.append({
            'type': 'warning',
            'message': f'📉 Portfolio is down {abs(metrics["total_return"]):.1f}%. Review underperforming assets and consider strategic rebalancing.'
        })
    
    if metrics['volatility'] > 25:
        enhanced_suggestions.append({
            'type': 'warning',
            'message': f'⚡ High volatility detected ({metrics["volatility"]:.1f}%). Consider adding stable assets like bonds or dividend stocks.'
        })
    
    all_suggestions = enhanced_suggestions + suggestions
    
    for suggestion in all_suggestions[:6]:
        if suggestion['type'] == 'diversification':
            st.markdown(f"""
            <div class="info-card">
                <strong>🎯 Diversification:</strong> {suggestion['message']}
            </div>
            """, unsafe_allow_html=True)
        elif suggestion['type'] == 'rebalancing':
            st.markdown(f"""
            <div class="warning-card">
                <strong>⚖️ Rebalancing:</strong> {suggestion['message']}
            </div>
            """, unsafe_allow_html=True)
        elif suggestion['type'] == 'opportunity':
            st.markdown(f"""
            <div class="success-card">
                <strong>🚀 Opportunity:</strong> {suggestion['message']}
            </div>
            """, unsafe_allow_html=True)
        elif suggestion['type'] == 'success':
            st.markdown(f"""
            <div class="success-card">
                <strong>✅ Success:</strong> {suggestion['message']}
            </div>
            """, unsafe_allow_html=True)
        elif suggestion['type'] == 'warning':
            st.markdown(f"""
            <div class="warning-card">
                <strong>⚠️ Warning:</strong> {suggestion['message']}
            </div>
            """, unsafe_allow_html=True)

def show_export_import():
    """Display export/import functionality"""
    
    st.markdown("### 📁 Export & Import Portfolio")
    
    if st.session_state.learning_mode:
        st.markdown("""
        <div class="info-card">
            <strong>📁 Export/Import Guide:</strong><br>
            Backup and restore your portfolio data. Export as JSON or CSV for safekeeping, or import previously saved portfolios.
        </div>
        """, unsafe_allow_html=True)
    
    export_tab, import_tab = st.tabs(["📤 Export Portfolio", "📥 Import Portfolio"])
    
    with export_tab:
        st.markdown("### Export Your Portfolio Data")
        
        if not st.session_state.portfolio:
            st.markdown("""
            <div class="warning-card">
                <strong>📝 No portfolio data to export</strong><br>
                Add some investments to your portfolio first, then return here to export your data.
            </div>
            """, unsafe_allow_html=True)
            return
        
        export_format = st.selectbox("Choose Export Format", ["JSON", "CSV"])
        
        if export_format == "JSON":
            export_data = {
                'username': st.session_state.username,
                'export_date': datetime.now().isoformat(),
                'portfolio': st.session_state.portfolio,
                'version': '2.0'
            }
            
            json_string = json.dumps(export_data, indent=2)
            
            st.download_button(
                label="📥 Download Portfolio (JSON)",
                data=json_string,
                file_name=f"portfolio_{st.session_state.username}_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )
            
            with st.expander("👀 Preview JSON Data"):
                st.code(json_string, language="json")
        
        else:  # CSV format
            csv_data = []
            for symbol, data in st.session_state.portfolio.items():
                asset_info = fetch_asset_data(symbol)
                if asset_info:
                    csv_data.append({
                        'Symbol': symbol,
                        'Name': asset_info['name'],
                        'Shares': data['shares'],
                        'Asset_Type': data['asset_type'],
                        'Purchase_Price': data.get('purchase_price', asset_info['current_price']),
                        'Current_Price': asset_info['current_price'],
                        'Current_Value': data['shares'] * asset_info['current_price'],
                        'Added_Date': data['added_date']
                    })
            
            if csv_data:
                df = pd.DataFrame(csv_data)
                csv_string = df.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download Portfolio (CSV)",
                    data=csv_string,
                    file_name=f"portfolio_{st.session_state.username}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
                
                with st.expander("👀 Preview CSV Data"):
                    st.dataframe(df)
    
    with import_tab:
        st.markdown("### Import Portfolio Data")
        
        uploaded_file = st.file_uploader(
            "Choose a portfolio file",
            type=['json', 'csv'],
            help="Upload a previously exported portfolio file"
        )
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.json'):
                    content = uploaded_file.read()
                    data = json.loads(content)
                    
                    if 'portfolio' in data:
                        st.markdown("""
                        <div class="success-card">
                            <strong>✅ Valid portfolio file detected!</strong><br>
                            Ready to import your portfolio data.
                        </div>
                        """, unsafe_allow_html=True)
                        
                        with st.expander("👀 Preview Import Data"):
                            st.json(data['portfolio'])
                        
                        import_option = st.selectbox(
                            "Import Option",
                            ["Replace Current Portfolio", "Merge with Current Portfolio"]
                        )
                        
                        if st.button("🚀 Import Portfolio", type="primary"):
                            if import_option == "Replace Current Portfolio":
                                st.session_state.portfolio = data['portfolio']
                            else:
                                for symbol, asset_data in data['portfolio'].items():
                                    st.session_state.portfolio[symbol] = asset_data
                            
                            portfolios = load_portfolios()
                            portfolios[st.session_state.username] = st.session_state.portfolio
                            save_portfolios(portfolios)
                            
                            st.markdown("""
                            <div class="success-card">
                                <strong>🎉 Portfolio imported successfully!</strong><br>
                                Your portfolio has been updated with the imported data.
                            </div>
                            """, unsafe_allow_html=True)
                            st.rerun()
                    else:
                        st.error("❌ Invalid portfolio file format.")
                
                elif uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                    
                    required_columns = ['Symbol', 'Shares', 'Asset_Type']
                    if all(col in df.columns for col in required_columns):
                        st.markdown("""
                        <div class="success-card">
                            <strong>✅ Valid CSV file detected!</strong><br>
                            Ready to import your portfolio data.
                        </div>
                        """, unsafe_allow_html=True)
                        
                        with st.expander("👀 Preview Import Data"):
                            st.dataframe(df)
                        
                        csv_import_option = st.selectbox(
                            "Import Option",
                            ["Replace Current Portfolio", "Merge with Current Portfolio"],
                            key="csv_import_option"
                        )
                        
                        if st.button("🚀 Import CSV Portfolio", type="primary"):
                            new_portfolio = {}
                            
                            for _, row in df.iterrows():
                                symbol = row['Symbol']
                                new_portfolio[symbol] = {
                                    'shares': float(row['Shares']),
                                    'asset_type': row['Asset_Type'],
                                    'purchase_price': float(row.get('Purchase_Price', 100.0)),
                                    'added_date': datetime.now().isoformat()
                                }
                            
                            if csv_import_option == "Replace Current Portfolio":
                                st.session_state.portfolio = new_portfolio
                            else:
                                for symbol, asset_data in new_portfolio.items():
                                    st.session_state.portfolio[symbol] = asset_data
                            
                            portfolios = load_portfolios()
                            portfolios[st.session_state.username] = st.session_state.portfolio
                            save_portfolios(portfolios)
                            
                            st.markdown("""
                            <div class="success-card">
                                <strong>🎉 CSV portfolio imported successfully!</strong><br>
                                Your portfolio has been updated with the imported data.
                            </div>
                            """, unsafe_allow_html=True)
                            st.rerun()
                    else:
                        st.error(f"❌ CSV file must contain columns: {', '.join(required_columns)}")
                        
            except Exception as e:
                st.error(f"❌ Error importing file: {str(e)}")

if __name__ == "__main__":
    main(), 'S"{return_pct:+.1f}%",
                'P&L': f"${current_value - invested_value:+,.2f}",
                'Asset Type': data['asset_type'],
                'Weight %': f"{(current_value / metrics['total_current_value']) * 100:.1f}%" if metrics['total_current_value'] > 0 else "0.0%"
            })
    
    if portfolio_data:
        st.markdown("### 📊 Holdings Breakdown")
        df = pd.DataFrame(portfolio_data)
        st.dataframe(df, use_container_width=True, height=400)
        
        if PLOTLY_AVAILABLE:
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                st.markdown("#### 🥧 Portfolio Allocation")
                # Extract numeric values safely
                current_values = []
                symbols = []
                
                for item in portfolio_data:
                    try:
                        value_str = item['Current Value'].replace('$', '').replace(',', '')
                        current_values.append(float(value_str))
                        symbols.append(item['Symbol'])
                    except:
                        continue
                
                if current_values:
                    fig_pie = px.pie(
                        values=current_values,
                        names=symbols,
                        title="Current Holdings Distribution",
                        hole=0.4
                    )
                    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                    fig_pie.update_layout(height=400)
                    st.plotly_chart(fig_pie, use_container_width=True)
            
            with col_chart2:
                st.markdown("#### 📈 Performance by Asset")
                return_values = []
                
                for item in portfolio_data:
                    try:
                        return_str = item['Return %'].replace('%', '').replace('+', '')
                        return_values.append(float(return_str))
                    except:
                        return_values.append(0.0)
                
                if return_values and symbols:
                    fig_returns = px.bar(
                        x=symbols,
                        y=return_values,
                        title="Return % by Holding",
                        color=return_values,
                        color_continuous_scale=['red', 'yellow', 'green']
                    )
                    fig_returns.update_layout(height=400, xaxis_title="Assets", yaxis_title="Return %")
                    st.plotly_chart(fig_returns, use_container_width=True)
        else:
            st.markdown("#### 📊 Simple Visualization")
            current_values = []
            symbols = []
            
            for item in portfolio_data:
                try:
                    value_str = item['Current Value'].replace('$', '').replace(',', '')
                    current_values.append(float(value_str))
                    symbols.append(item['Symbol'])
                except:
                    continue
            
            if current_values and symbols:
                chart_data = pd.DataFrame({
                    'Symbol': symbols,
                    'Current Value': current_values
                })
                st.bar_chart(chart_data.set_index('Symbol'))

def show_asset_management():
    """Display asset management page"""
    
    st.markdown("### 🎯 Manage Portfolio Assets")
    
    if st.session_state.learning_mode:
        st.markdown("""
        <div class="info-card">
            <strong>🎯 Asset Management Guide:</strong><br>
            Use this section to build and maintain your investment portfolio. Add assets, track purchase prices, and manage your holdings.
        </div>
        """, unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["➕ Add Assets", "➖ Remove Assets"])
    
    with tab1:
        st.markdown("### Add New Investment")
        
        col1, col2 = st.columns(2)
        
        with col1:
            popular_assets = get_popular_assets()
            
            # Create categories for better organization
            asset_categories = {
                "🏢 Large Cap Stocks": [k for k in popular_assets.keys() if k in ["AAPL", "GOOGL", "GOOG", "MSFT", "AMZN", "TSLA", "NVDA", "META", "NFLX", "JPM", "JNJ", "V", "WMT", "PG", "UNH", "HD", "MA", "BAC", "ADBE", "CRM", "XOM", "CVX", "KO", "PFE", "INTC", "CSCO", "VZ", "MRK", "ABT", "TMO"]],
                "🚀 Growth & Tech": [k for k in popular_assets.keys() if k in ["AMD", "PYPL", "SHOP", "SQ", "ROKU", "ZM", "SNOW", "PLTR", "CRWD", "NET"]],
                "🏛️ Vanguard ETFs": [k for k in popular_assets.keys() if k.startswith("V") and k in ["VTI", "VEA", "VWO", "VTV", "VUG", "VIG", "VYM", "VXUS", "VOO", "VT", "VB", "VO", "VGT", "VHT", "VFH", "VDE", "VAW", "VIS", "VCR", "VDC", "VPU", "VNQ"]],
                "⚫ BlackRock iShares": [k for k in popular_assets.keys() if k in ["QQQ", "IWM", "EFA", "EEM", "IVV", "IEFA", "IEMG", "IJH", "IJR", "IVW", "IVE", "IWF", "IWD", "ITOT", "IXUS", "IYY", "IWB", "IWV", "ACWI", "ACWX"]],
                "🎯 Sector ETFs": [k for k in popular_assets.keys() if k.startswith("XL") or k in ["IBB", "SOXX", "SKYY", "HACK", "ROBO", "ARKK", "ARKQ", "ARKW", "ARKG", "ARKF"]],
                "🏦 Bonds & Fixed Income": [k for k in popular_assets.keys() if k in ["AGG", "BND", "TLT", "IEF", "SHY", "LQD", "HYG", "JNK", "TIP", "VTEB", "MUB", "EMB", "BNDX", "VGIT", "VGSH", "VGLT", "VCIT", "VCSH", "BSV", "BIV", "BLV"]],
                "💎 Fidelity Funds": [k for k in popular_assets.keys() if k.startswith("F") and len(k) > 2],
                "🥇 Commodities": [k for k in popular_assets.keys() if k in ["GLD", "SLV", "USO", "UNG", "DBA", "DBC", "PDBC", "GSG", "COMT", "IAU", "SGOL", "PPLT", "PALL", "CORN", "WEAT", "SOYB", "CANE", "COW"]],
                "🌍 International": [k for k in popular_assets.keys() if k.startswith("EW") or k in ["FXI", "INDA", "RSX", "ASHR", "MCHI", "KWEB", "YINN"]],
                "₿ Major Cryptocurrencies": [k for k in popular_assets.keys() if k.endswith("-USD") and k in ["BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "ADA-USD", "SOL-USD", "DOGE-USD", "DOT-USD", "MATIC-USD", "SHIB-USD", "LTC-USD", "TRX-USD", "AVAX-USD", "LINK-USD", "ATOM-USD"]],
                "🔗 DeFi & Web3": [k for k in popular_assets.keys() if k.endswith("-USD") and k in ["XLM-USD", "ALGO-USD", "VET-USD", "ICP-USD", "FIL-USD", "SAND-USD", "MANA-USD", "AXS-USD", "ENJ-USD", "CHZ-USD", "FLOW-USD", "THETA-USD", "XTZ-USD", "EGLD-USD", "NEAR-USD"]],
                "📈 Crypto ETFs": [k for k in popular_assets.keys() if k in ["BITO", "BITI", "GBTC", "ETHE", "GDLC", "BITQ", "BLOK", "LEGR"]],
                "💱 Currency ETFs": [k for k in popular_assets.keys() if k.startswith("FX") or k in ["UUP", "UDN"]],
                "💰 Dividend ETFs": [k for k in popular_assets.keys() if k in ["SCHD", "DVY", "VYM", "NOBL", "DGRO", "HDV", "SPHD", "PEY", "RDVY", "FDL"]],
                "🌱 ESG & Clean Energy": [k for k in popular_assets.keys() if k in ["ESG", "ESGU", "ESGD", "ESGE", "VSGX", "ESGV", "SUSA", "DSI", "ICLN", "PBW", "QCLN"]],
                "📊 Volatility & Leverage": [k for k in popular_assets.keys() if k in ["VIX", "UVXY", "SVXY", "VXX", "VIXY", "TQQQ", "SQQQ", "UPRO", "SPXU", "TNA", "TZA", "TECL", "TECS", "FAS", "FAZ"]]
            }
            
            selected_category = st.selectbox(
                "Choose Asset Category",
                [""] + list(asset_categories.keys()),
                help="Browse assets by category for easier selection"
            )
            
            if selected_category:
                available_assets = asset_categories[selected_category]
                selected_popular = st.selectbox(
                    f"Assets in {selected_category}",
                    [""] + available_assets,
                    format_func=lambda x: f"{x} - {popular_assets[x]}" if x else "",
                    help="Select from assets in this category"
                )
            else:
                selected_popular = st.selectbox(
                    "Or search all assets",
                    [""] + list(popular_assets.keys()),
                    format_func=lambda x: f"{x} - {popular_assets[x]}" if x else "",
                    help="Search through all available assets"
                )
            
            custom_symbol = st.text_input(
                "Or Enter Custom Symbol",
                placeholder="e.g., AAPL, BTC-USD, GLD.L (London)",
                help="Enter any valid symbol including international exchanges"
            )
            
            symbol_to_use = ""
            if selected_popular:
                symbol_to_use = selected_popular
            elif custom_symbol:
                symbol_to_use = custom_symbol.upper()
        
        with col2:
            shares = st.number_input(
                "Number of Shares/Units",
                min_value=0.001,
                value=1.0,
                step=0.1,
                help="Enter the quantity you own"
            )
            
            purchase_price = st.number_input(
                "Purchase Price per Share ($)",
                min_value=0.01,
                value=100.0,
                step=0.01,
                help="Enter the average price you paid per share"
            )
            
            # NEW: Currency selection
            currencies = get_currency_list()
            selected_currency = st.selectbox(
                "Currency",
                list(currencies.keys()),
                index=0,  # Default to USD
                format_func=lambda x: f"{x} - {currencies[x]['name']}",
                help="Select the currency for this asset"
            )
            
            asset_type = st.selectbox(
                "Asset Type",
                ["Stock", "ETF", "Cryptocurrency", "Bond", "Commodity", "Index Fund", "Mutual Fund", "REIT", "Other"]
            )
            
            if st.session_state.learning_mode:
                st.markdown("""
                <div class="info-card">
                    <strong>💡 Pro Tip:</strong><br>
                    • Select the correct currency for international assets<br>
                    • Purchase price should be in the selected currency<br>
                    • The system will convert to your base currency for analysis
                </div>
                """, unsafe_allow_html=True)
        
        if symbol_to_use:
            asset_info = fetch_asset_data(symbol_to_use)
            if asset_info:
                current_price = asset_info['current_price']
                
                # Convert prices for display if different currency
                if selected_currency != "USD":
                    exchange_rate = get_exchange_rate("USD", selected_currency)
                    current_price_display = current_price * exchange_rate
                    purchase_price_display = purchase_price
                else:
                    current_price_display = current_price
                    purchase_price_display = purchase_price
                
                potential_return = ((current_price_display - purchase_price_display) / purchase_price_display) * 100 if purchase_price_display > 0 else 0
                
                # Create three columns for metrics display
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                
                with metric_col1:
                    currency_symbol = currencies[selected_currency]["symbol"]
                    st.metric("Current Price", f"{currency_symbol}{current_price_display:.2f}")
                with metric_col2:
                    st.metric("Your Purchase Price", f"{currency_symbol}{purchase_price_display:.2f}")
                with metric_col3:
                    st.metric("Potential Return", f"{potential_return:+.1f}%", 
                             delta=f"{currency_symbol}{(current_price_display - purchase_price_display) * shares:+.2f}")
                
                if st.button("✅ Add to Portfolio", type="primary", use_container_width=True):
                    st.session_state.portfolio[symbol_to_use] = {
                        'shares': shares,
                        'asset_type': asset_type,
                        'purchase_price': purchase_price,
                        'currency': selected_currency,  # NEW: Store currency
                        'added_date': datetime.now().isoformat()
                    }
                    
                    portfolios = load_portfolios()
                    portfolios[st.session_state.username] = st.session_state.portfolio
                    save_portfolios(portfolios)
                    
                    st.markdown("""
                    <div class="success-card">
                        <strong>🎉 Asset Added Successfully!</strong><br>
                        Your portfolio metrics are being recalculated with the new asset allocation and currency conversion.
                    </div>
                    """, unsafe_allow_html=True)
                    st.rerun()
            else:
                st.error(f"❌ Could not find asset data for '{symbol_to_use}'. Please check the symbol.")
    
    with tab2:
        st.markdown("### Remove Assets from Portfolio")
        
        if st.session_state.portfolio:
            current_values = []
            for symbol, data in st.session_state.portfolio.items():
                asset_info = fetch_asset_data(symbol)
                if asset_info:
                    current_value = data['shares'] * asset_info['current_price']
                    current_values.append({
                        'Symbol': symbol,
                        'Shares': f"{data['shares']:.3f}",
                        'Current Value': f"${current_value:,.2f}",
                        'Asset Type': data['asset_type']
                    })
            
            if current_values:
                df_current = pd.DataFrame(current_values)
                st.dataframe(df_current, hide_index=True)
            
            assets_to_remove = st.multiselect(
                "Select assets to remove:",
                list(st.session_state.portfolio.keys()),
                help="Choose one or more assets to remove from your portfolio"
            )
            
            if assets_to_remove:
                st.warning(f"⚠️ You are about to remove {len(assets_to_remove)} asset(s) from your portfolio.")
                
                if st.button("🗑️ Remove Selected Assets", type="secondary"):
                    for asset in assets_to_remove:
                        del st.session_state.portfolio[asset]
                    
                    portfolios = load_portfolios()
                    portfolios[st.session_state.username] = st.session_state.portfolio
                    save_portfolios(portfolios)
                    
                    st.markdown("""
                    <div class="success-card">
                        <strong>✅ Assets Removed Successfully!</strong><br>
                        Your portfolio has been updated and metrics recalculated.
                    </div>
                    """, unsafe_allow_html=True)
                    st.rerun()
        else:
            st.markdown("""
            <div class="info-card">
                <strong>📝 No assets to remove</strong><br>
                Your portfolio is currently empty. Add some investments first!
            </div>
            """, unsafe_allow_html=True)

def show_analytics_dashboard():
    """Display analytics dashboard"""
    
    st.markdown("### 📊 Advanced Analytics Dashboard")
    
    if st.session_state.learning_mode:
        st.markdown("""
        <div class="info-card">
            <strong>📈 Analytics Dashboard Guide:</strong><br>
            This dashboard provides professional-grade analysis tools including performance metrics, technical analysis, and AI-powered insights.
        </div>
        """, unsafe_allow_html=True)
    
    if not st.session_state.portfolio:
        st.warning("⚠️ Add assets to your portfolio to see advanced analytics.")
        return
    
    metrics = calculate_portfolio_metrics_advanced(st.session_state.portfolio)
    
    st.markdown("### 📈 Advanced Performance Metrics")
    
    perf_col1, perf_col2, perf_col3, perf_col4, perf_col5 = st.columns(5)
    
    with perf_col1:
        st.metric(
            "Annualized Return", 
            f"{metrics['annualized_return']:+.1f}%",
            help="Expected annual return based on current performance"
        )
    
    with perf_col2:
        st.metric(
            "Portfolio Beta", 
            f"{metrics['beta']:.2f}",
            delta="Market Risk" if metrics['beta'] > 1 else "Lower Risk",
            help="Portfolio volatility relative to market"
        )
    
    with perf_col3:
        st.metric(
            "Sharpe Ratio", 
            f"{metrics['sharpe_ratio']:.2f}",
            delta="Excellent" if metrics['sharpe_ratio'] > 1 else "Good" if metrics['sharpe_ratio'] > 0.5 else "Needs Improvement",
            help="Risk-adjusted return measure"
        )
    
    with perf_col4:
        st.metric(
            "Max Drawdown", 
            f"{metrics['max_drawdown']:.1f}%",
            delta="Risk Level",
            delta_color="inverse",
            help="Largest peak-to-trough decline"
        )
    
    with perf_col5:
        st.metric(
            "Portfolio Volatility", 
            f"{metrics['volatility']:.1f}%",
            help="Measure of price fluctuation"
        )
    
    # Technical analysis for top holdings
    st.markdown("### 🔍 Technical Analysis - Top Holdings")
    
    portfolio_values = []
    for symbol, data in st.session_state.portfolio.items():
        asset_info = fetch_asset_data(symbol)
        if asset_info:
            value = data['shares'] * asset_info['current_price']
            portfolio_values.append((symbol, value))
    
    portfolio_values.sort(key=lambda x: x[1], reverse=True)
    top_holdings = [item[0] for item in portfolio_values[:3]]
    
    for i, symbol in enumerate(top_holdings):
        with st.expander(f"📊 {symbol} - Technical Analysis", expanded=(i == 0)):
            indicators = calculate_technical_indicators(symbol)
            
            if indicators is not None:
                tech_col1, tech_col2, tech_col3, tech_col4 = st.columns(4)
                
                current_price = indicators['Close'].iloc[-1]
                ma20 = indicators['MA_20'].iloc[-1]
                current_rsi = indicators['RSI'].iloc[-1]
                current_macd = indicators['MACD'].iloc[-1]
                current_signal = indicators['MACD_Signal'].iloc[-1]
                
                with tech_col1:
                    price_trend = "Bullish" if current_price > ma20 else "Bearish"
                    st.metric("Price Trend", price_trend, f"${current_price:.2f}")
                
                with tech_col2:
                    rsi_status = "Overbought" if current_rsi > 70 else "Oversold" if current_rsi < 30 else "Neutral"
                    rsi_color = "🔴" if current_rsi > 70 else "🟢" if current_rsi < 30 else "🟡"
                    st.metric("RSI Signal", f"{rsi_color} {rsi_status}", f"{current_rsi:.1f}")
                
                with tech_col3:
                    macd_trend = "Bullish" if current_macd > current_signal else "Bearish"
                    macd_color = "🟢" if current_macd > current_signal else "🔴"
                    st.metric("MACD Signal", f"{macd_color} {macd_trend}")
                
                with tech_col4:
                    support_level = current_price * 0.95
                    resistance_level = current_price * 1.05
                    st.metric("Support/Resistance", f"${support_level:.2f} / ${resistance_level:.2f}")
                
                if PLOTLY_AVAILABLE:
                    fig = make_subplots(
                        rows=3, cols=1,
                        subplot_titles=(
                            f'{symbol} - Price & Moving Averages',
                            'Relative Strength Index (RSI)',
                            'MACD & Signal Line'
                        ),
                        vertical_spacing=0.08,
                        row_heights=[0.5, 0.25, 0.25]
                    )
                    
                    fig.add_trace(
                        go.Scatter(x=indicators.index, y=indicators['Close'], name='Price', line=dict(color='blue', width=2)),
                        row=1, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=indicators.index, y=indicators['MA_20'], name='MA 20', line=dict(color='orange')),
                        row=1, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=indicators.index, y=indicators['MA_50'], name='MA 50', line=dict(color='red')),
                        row=1, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(x=indicators.index, y=indicators['RSI'], name='RSI', line=dict(color='purple')),
                        row=2, col=1
                    )
                    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
                    
                    fig.add_trace(
                        go.Scatter(x=indicators.index, y=indicators['MACD'], name='MACD', line=dict(color='blue')),
                        row=3, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=indicators.index, y=indicators['MACD_Signal'], name='Signal', line=dict(color='red')),
                        row=3, col=1
                    )
                    
                    fig.update_layout(height=700, showlegend=True, title_text=f"{symbol} Technical Analysis")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.markdown("#### 📈 Price Chart")
                    st.line_chart(indicators[['Close', 'MA_20', 'MA_50']])
                    
                    simple_col1, simple_col2 = st.columns(2)
                    with simple_col1:
                        st.markdown("#### RSI")
                        st.line_chart(indicators['RSI'])
                    with simple_col2:
                        st.markdown("#### MACD")
                        st.line_chart(indicators[['MACD', 'MACD_Signal']])
    
    # Investment suggestions
    st.markdown("### 🤖 AI-Powered Investment Insights")
    suggestions = generate_investment_suggestions(st.session_state.portfolio)
    
    enhanced_suggestions = []
    
    if metrics['total_return'] > 10:
        enhanced_suggestions.append({
            'type': 'success',
            'message': f'🎉 Excellent performance! Your portfolio is up {metrics["total_return"]:.1f}%. Consider taking some profits and rebalancing.'
        })
    elif metrics['total_return'] < -5:
        enhanced_suggestions.append({
            'type': 'warning',
            'message': f'📉 Portfolio is down {abs(metrics["total_return"]):.1f}%. Review underperforming assets and consider strategic rebalancing.'
        })
    
    if metrics['volatility'] > 25:
        enhanced_suggestions.append({
            'type': 'warning',
            'message': f'⚡ High volatility detected ({metrics["volatility"]:.1f}%). Consider adding stable assets like bonds or dividend stocks.'
        })
    
    all_suggestions = enhanced_suggestions + suggestions
    
    for suggestion in all_suggestions[:6]:
        if suggestion['type'] == 'diversification':
            st.markdown(f"""
            <div class="info-card">
                <strong>🎯 Diversification:</strong> {suggestion['message']}
            </div>
            """, unsafe_allow_html=True)
        elif suggestion['type'] == 'rebalancing':
            st.markdown(f"""
            <div class="warning-card">
                <strong>⚖️ Rebalancing:</strong> {suggestion['message']}
            </div>
            """, unsafe_allow_html=True)
        elif suggestion['type'] == 'opportunity':
            st.markdown(f"""
            <div class="success-card">
                <strong>🚀 Opportunity:</strong> {suggestion['message']}
            </div>
            """, unsafe_allow_html=True)
        elif suggestion['type'] == 'success':
            st.markdown(f"""
            <div class="success-card">
                <strong>✅ Success:</strong> {suggestion['message']}
            </div>
            """, unsafe_allow_html=True)
        elif suggestion['type'] == 'warning':
            st.markdown(f"""
            <div class="warning-card">
                <strong>⚠️ Warning:</strong> {suggestion['message']}
            </div>
            """, unsafe_allow_html=True)

def show_export_import():
    """Display export/import functionality"""
    
    st.markdown("### 📁 Export & Import Portfolio")
    
    if st.session_state.learning_mode:
        st.markdown("""
        <div class="info-card">
            <strong>📁 Export/Import Guide:</strong><br>
            Backup and restore your portfolio data. Export as JSON or CSV for safekeeping, or import previously saved portfolios.
        </div>
        """, unsafe_allow_html=True)
    
    export_tab, import_tab = st.tabs(["📤 Export Portfolio", "📥 Import Portfolio"])
    
    with export_tab:
        st.markdown("### Export Your Portfolio Data")
        
        if not st.session_state.portfolio:
            st.markdown("""
            <div class="warning-card">
                <strong>📝 No portfolio data to export</strong><br>
                Add some investments to your portfolio first, then return here to export your data.
            </div>
            """, unsafe_allow_html=True)
            return
        
        export_format = st.selectbox("Choose Export Format", ["JSON", "CSV"])
        
        if export_format == "JSON":
            export_data = {
                'username': st.session_state.username,
                'export_date': datetime.now().isoformat(),
                'portfolio': st.session_state.portfolio,
                'version': '2.0'
            }
            
            json_string = json.dumps(export_data, indent=2)
            
            st.download_button(
                label="📥 Download Portfolio (JSON)",
                data=json_string,
                file_name=f"portfolio_{st.session_state.username}_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )
            
            with st.expander("👀 Preview JSON Data"):
                st.code(json_string, language="json")
        
        else:  # CSV format
            csv_data = []
            for symbol, data in st.session_state.portfolio.items():
                asset_info = fetch_asset_data(symbol)
                if asset_info:
                    csv_data.append({
                        'Symbol': symbol,
                        'Name': asset_info['name'],
                        'Shares': data['shares'],
                        'Asset_Type': data['asset_type'],
                        'Purchase_Price': data.get('purchase_price', asset_info['current_price']),
                        'Current_Price': asset_info['current_price'],
                        'Current_Value': data['shares'] * asset_info['current_price'],
                        'Added_Date': data['added_date']
                    })
            
            if csv_data:
                df = pd.DataFrame(csv_data)
                csv_string = df.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download Portfolio (CSV)",
                    data=csv_string,
                    file_name=f"portfolio_{st.session_state.username}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
                
                with st.expander("👀 Preview CSV Data"):
                    st.dataframe(df)
    
    with import_tab:
        st.markdown("### Import Portfolio Data")
        
        uploaded_file = st.file_uploader(
            "Choose a portfolio file",
            type=['json', 'csv'],
            help="Upload a previously exported portfolio file"
        )
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.json'):
                    content = uploaded_file.read()
                    data = json.loads(content)
                    
                    if 'portfolio' in data:
                        st.markdown("""
                        <div class="success-card">
                            <strong>✅ Valid portfolio file detected!</strong><br>
                            Ready to import your portfolio data.
                        </div>
                        """, unsafe_allow_html=True)
                        
                        with st.expander("👀 Preview Import Data"):
                            st.json(data['portfolio'])
                        
                        import_option = st.selectbox(
                            "Import Option",
                            ["Replace Current Portfolio", "Merge with Current Portfolio"]
                        )
                        
                        if st.button("🚀 Import Portfolio", type="primary"):
                            if import_option == "Replace Current Portfolio":
                                st.session_state.portfolio = data['portfolio']
                            else:
                                for symbol, asset_data in data['portfolio'].items():
                                    st.session_state.portfolio[symbol] = asset_data
                            
                            portfolios = load_portfolios()
                            portfolios[st.session_state.username] = st.session_state.portfolio
                            save_portfolios(portfolios)
                            
                            st.markdown("""
                            <div class="success-card">
                                <strong>🎉 Portfolio imported successfully!</strong><br>
                                Your portfolio has been updated with the imported data.
                            </div>
                            """, unsafe_allow_html=True)
                            st.rerun()
                    else:
                        st.error("❌ Invalid portfolio file format.")
                
                elif uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                    
                    required_columns = ['Symbol', 'Shares', 'Asset_Type']
                    if all(col in df.columns for col in required_columns):
                        st.markdown("""
                        <div class="success-card">
                            <strong>✅ Valid CSV file detected!</strong><br>
                            Ready to import your portfolio data.
                        </div>
                        """, unsafe_allow_html=True)
                        
                        with st.expander("👀 Preview Import Data"):
                            st.dataframe(df)
                        
                        csv_import_option = st.selectbox(
                            "Import Option",
                            ["Replace Current Portfolio", "Merge with Current Portfolio"],
                            key="csv_import_option"
                        )
                        
                        if st.button("🚀 Import CSV Portfolio", type="primary"):
                            new_portfolio = {}
                            
                            for _, row in df.iterrows():
                                symbol = row['Symbol']
                                new_portfolio[symbol] = {
                                    'shares': float(row['Shares']),
                                    'asset_type': row['Asset_Type'],
                                    'purchase_price': float(row.get('Purchase_Price', 100.0)),
                                    'added_date': datetime.now().isoformat()
                                }
                            
                            if csv_import_option == "Replace Current Portfolio":
                                st.session_state.portfolio = new_portfolio
                            else:
                                for symbol, asset_data in new_portfolio.items():
                                    st.session_state.portfolio[symbol] = asset_data
                            
                            portfolios = load_portfolios()
                            portfolios[st.session_state.username] = st.session_state.portfolio
                            save_portfolios(portfolios)
                            
                            st.markdown("""
                            <div class="success-card">
                                <strong>🎉 CSV portfolio imported successfully!</strong><br>
                                Your portfolio has been updated with the imported data.
                            </div>
                            """, unsafe_allow_html=True)
                            st.rerun()
                    else:
                        st.error(f"❌ CSV file must contain columns: {', '.join(required_columns)}")
                        
            except Exception as e:
                st.error(f"❌ Error importing file: {str(e)}")

if __name__ == "__main__":
    main(), 'HK"{return_pct:+.1f}%",
                'P&L': f"${current_value - invested_value:+,.2f}",
                'Asset Type': data['asset_type'],
                'Weight %': f"{(current_value / metrics['total_current_value']) * 100:.1f}%" if metrics['total_current_value'] > 0 else "0.0%"
            })
    
    if portfolio_data:
        st.markdown("### 📊 Holdings Breakdown")
        df = pd.DataFrame(portfolio_data)
        st.dataframe(df, use_container_width=True, height=400)
        
        if PLOTLY_AVAILABLE:
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                st.markdown("#### 🥧 Portfolio Allocation")
                # Extract numeric values safely
                current_values = []
                symbols = []
                
                for item in portfolio_data:
                    try:
                        value_str = item['Current Value'].replace('$', '').replace(',', '')
                        current_values.append(float(value_str))
                        symbols.append(item['Symbol'])
                    except:
                        continue
                
                if current_values:
                    fig_pie = px.pie(
                        values=current_values,
                        names=symbols,
                        title="Current Holdings Distribution",
                        hole=0.4
                    )
                    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                    fig_pie.update_layout(height=400)
                    st.plotly_chart(fig_pie, use_container_width=True)
            
            with col_chart2:
                st.markdown("#### 📈 Performance by Asset")
                return_values = []
                
                for item in portfolio_data:
                    try:
                        return_str = item['Return %'].replace('%', '').replace('+', '')
                        return_values.append(float(return_str))
                    except:
                        return_values.append(0.0)
                
                if return_values and symbols:
                    fig_returns = px.bar(
                        x=symbols,
                        y=return_values,
                        title="Return % by Holding",
                        color=return_values,
                        color_continuous_scale=['red', 'yellow', 'green']
                    )
                    fig_returns.update_layout(height=400, xaxis_title="Assets", yaxis_title="Return %")
                    st.plotly_chart(fig_returns, use_container_width=True)
        else:
            st.markdown("#### 📊 Simple Visualization")
            current_values = []
            symbols = []
            
            for item in portfolio_data:
                try:
                    value_str = item['Current Value'].replace('$', '').replace(',', '')
                    current_values.append(float(value_str))
                    symbols.append(item['Symbol'])
                except:
                    continue
            
            if current_values and symbols:
                chart_data = pd.DataFrame({
                    'Symbol': symbols,
                    'Current Value': current_values
                })
                st.bar_chart(chart_data.set_index('Symbol'))

def show_asset_management():
    """Display asset management page"""
    
    st.markdown("### 🎯 Manage Portfolio Assets")
    
    if st.session_state.learning_mode:
        st.markdown("""
        <div class="info-card">
            <strong>🎯 Asset Management Guide:</strong><br>
            Use this section to build and maintain your investment portfolio. Add assets, track purchase prices, and manage your holdings.
        </div>
        """, unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["➕ Add Assets", "➖ Remove Assets"])
    
    with tab1:
        st.markdown("### Add New Investment")
        
        col1, col2 = st.columns(2)
        
        with col1:
            popular_assets = get_popular_assets()
            
            # Create categories for better organization
            asset_categories = {
                "🏢 Large Cap Stocks": [k for k in popular_assets.keys() if k in ["AAPL", "GOOGL", "GOOG", "MSFT", "AMZN", "TSLA", "NVDA", "META", "NFLX", "JPM", "JNJ", "V", "WMT", "PG", "UNH", "HD", "MA", "BAC", "ADBE", "CRM", "XOM", "CVX", "KO", "PFE", "INTC", "CSCO", "VZ", "MRK", "ABT", "TMO"]],
                "🚀 Growth & Tech": [k for k in popular_assets.keys() if k in ["AMD", "PYPL", "SHOP", "SQ", "ROKU", "ZM", "SNOW", "PLTR", "CRWD", "NET"]],
                "🏛️ Vanguard ETFs": [k for k in popular_assets.keys() if k.startswith("V") and k in ["VTI", "VEA", "VWO", "VTV", "VUG", "VIG", "VYM", "VXUS", "VOO", "VT", "VB", "VO", "VGT", "VHT", "VFH", "VDE", "VAW", "VIS", "VCR", "VDC", "VPU", "VNQ"]],
                "⚫ BlackRock iShares": [k for k in popular_assets.keys() if k in ["QQQ", "IWM", "EFA", "EEM", "IVV", "IEFA", "IEMG", "IJH", "IJR", "IVW", "IVE", "IWF", "IWD", "ITOT", "IXUS", "IYY", "IWB", "IWV", "ACWI", "ACWX"]],
                "🎯 Sector ETFs": [k for k in popular_assets.keys() if k.startswith("XL") or k in ["IBB", "SOXX", "SKYY", "HACK", "ROBO", "ARKK", "ARKQ", "ARKW", "ARKG", "ARKF"]],
                "🏦 Bonds & Fixed Income": [k for k in popular_assets.keys() if k in ["AGG", "BND", "TLT", "IEF", "SHY", "LQD", "HYG", "JNK", "TIP", "VTEB", "MUB", "EMB", "BNDX", "VGIT", "VGSH", "VGLT", "VCIT", "VCSH", "BSV", "BIV", "BLV"]],
                "💎 Fidelity Funds": [k for k in popular_assets.keys() if k.startswith("F") and len(k) > 2],
                "🥇 Commodities": [k for k in popular_assets.keys() if k in ["GLD", "SLV", "USO", "UNG", "DBA", "DBC", "PDBC", "GSG", "COMT", "IAU", "SGOL", "PPLT", "PALL", "CORN", "WEAT", "SOYB", "CANE", "COW"]],
                "🌍 International": [k for k in popular_assets.keys() if k.startswith("EW") or k in ["FXI", "INDA", "RSX", "ASHR", "MCHI", "KWEB", "YINN"]],
                "₿ Major Cryptocurrencies": [k for k in popular_assets.keys() if k.endswith("-USD") and k in ["BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "ADA-USD", "SOL-USD", "DOGE-USD", "DOT-USD", "MATIC-USD", "SHIB-USD", "LTC-USD", "TRX-USD", "AVAX-USD", "LINK-USD", "ATOM-USD"]],
                "🔗 DeFi & Web3": [k for k in popular_assets.keys() if k.endswith("-USD") and k in ["XLM-USD", "ALGO-USD", "VET-USD", "ICP-USD", "FIL-USD", "SAND-USD", "MANA-USD", "AXS-USD", "ENJ-USD", "CHZ-USD", "FLOW-USD", "THETA-USD", "XTZ-USD", "EGLD-USD", "NEAR-USD"]],
                "📈 Crypto ETFs": [k for k in popular_assets.keys() if k in ["BITO", "BITI", "GBTC", "ETHE", "GDLC", "BITQ", "BLOK", "LEGR"]],
                "💱 Currency ETFs": [k for k in popular_assets.keys() if k.startswith("FX") or k in ["UUP", "UDN"]],
                "💰 Dividend ETFs": [k for k in popular_assets.keys() if k in ["SCHD", "DVY", "VYM", "NOBL", "DGRO", "HDV", "SPHD", "PEY", "RDVY", "FDL"]],
                "🌱 ESG & Clean Energy": [k for k in popular_assets.keys() if k in ["ESG", "ESGU", "ESGD", "ESGE", "VSGX", "ESGV", "SUSA", "DSI", "ICLN", "PBW", "QCLN"]],
                "📊 Volatility & Leverage": [k for k in popular_assets.keys() if k in ["VIX", "UVXY", "SVXY", "VXX", "VIXY", "TQQQ", "SQQQ", "UPRO", "SPXU", "TNA", "TZA", "TECL", "TECS", "FAS", "FAZ"]]
            }
            
            selected_category = st.selectbox(
                "Choose Asset Category",
                [""] + list(asset_categories.keys()),
                help="Browse assets by category for easier selection"
            )
            
            if selected_category:
                available_assets = asset_categories[selected_category]
                selected_popular = st.selectbox(
                    f"Assets in {selected_category}",
                    [""] + available_assets,
                    format_func=lambda x: f"{x} - {popular_assets[x]}" if x else "",
                    help="Select from assets in this category"
                )
            else:
                selected_popular = st.selectbox(
                    "Or search all assets",
                    [""] + list(popular_assets.keys()),
                    format_func=lambda x: f"{x} - {popular_assets[x]}" if x else "",
                    help="Search through all available assets"
                )
            
            custom_symbol = st.text_input(
                "Or Enter Custom Symbol",
                placeholder="e.g., AAPL, BTC-USD, GLD.L (London)",
                help="Enter any valid symbol including international exchanges"
            )
            
            symbol_to_use = ""
            if selected_popular:
                symbol_to_use = selected_popular
            elif custom_symbol:
                symbol_to_use = custom_symbol.upper()
        
        with col2:
            shares = st.number_input(
                "Number of Shares/Units",
                min_value=0.001,
                value=1.0,
                step=0.1,
                help="Enter the quantity you own"
            )
            
            purchase_price = st.number_input(
                "Purchase Price per Share ($)",
                min_value=0.01,
                value=100.0,
                step=0.01,
                help="Enter the average price you paid per share"
            )
            
            # NEW: Currency selection
            currencies = get_currency_list()
            selected_currency = st.selectbox(
                "Currency",
                list(currencies.keys()),
                index=0,  # Default to USD
                format_func=lambda x: f"{x} - {currencies[x]['name']}",
                help="Select the currency for this asset"
            )
            
            asset_type = st.selectbox(
                "Asset Type",
                ["Stock", "ETF", "Cryptocurrency", "Bond", "Commodity", "Index Fund", "Mutual Fund", "REIT", "Other"]
            )
            
            if st.session_state.learning_mode:
                st.markdown("""
                <div class="info-card">
                    <strong>💡 Pro Tip:</strong><br>
                    • Select the correct currency for international assets<br>
                    • Purchase price should be in the selected currency<br>
                    • The system will convert to your base currency for analysis
                </div>
                """, unsafe_allow_html=True)
        
        if symbol_to_use:
            asset_info = fetch_asset_data(symbol_to_use)
            if asset_info:
                current_price = asset_info['current_price']
                
                # Convert prices for display if different currency
                if selected_currency != "USD":
                    exchange_rate = get_exchange_rate("USD", selected_currency)
                    current_price_display = current_price * exchange_rate
                    purchase_price_display = purchase_price
                else:
                    current_price_display = current_price
                    purchase_price_display = purchase_price
                
                potential_return = ((current_price_display - purchase_price_display) / purchase_price_display) * 100 if purchase_price_display > 0 else 0
                
                # Create three columns for metrics display
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                
                with metric_col1:
                    currency_symbol = currencies[selected_currency]["symbol"]
                    st.metric("Current Price", f"{currency_symbol}{current_price_display:.2f}")
                with metric_col2:
                    st.metric("Your Purchase Price", f"{currency_symbol}{purchase_price_display:.2f}")
                with metric_col3:
                    st.metric("Potential Return", f"{potential_return:+.1f}%", 
                             delta=f"{currency_symbol}{(current_price_display - purchase_price_display) * shares:+.2f}")
                
                if st.button("✅ Add to Portfolio", type="primary", use_container_width=True):
                    st.session_state.portfolio[symbol_to_use] = {
                        'shares': shares,
                        'asset_type': asset_type,
                        'purchase_price': purchase_price,
                        'currency': selected_currency,  # NEW: Store currency
                        'added_date': datetime.now().isoformat()
                    }
                    
                    portfolios = load_portfolios()
                    portfolios[st.session_state.username] = st.session_state.portfolio
                    save_portfolios(portfolios)
                    
                    st.markdown("""
                    <div class="success-card">
                        <strong>🎉 Asset Added Successfully!</strong><br>
                        Your portfolio metrics are being recalculated with the new asset allocation and currency conversion.
                    </div>
                    """, unsafe_allow_html=True)
                    st.rerun()
            else:
                st.error(f"❌ Could not find asset data for '{symbol_to_use}'. Please check the symbol.")
    
    with tab2:
        st.markdown("### Remove Assets from Portfolio")
        
        if st.session_state.portfolio:
            current_values = []
            for symbol, data in st.session_state.portfolio.items():
                asset_info = fetch_asset_data(symbol)
                if asset_info:
                    current_value = data['shares'] * asset_info['current_price']
                    current_values.append({
                        'Symbol': symbol,
                        'Shares': f"{data['shares']:.3f}",
                        'Current Value': f"${current_value:,.2f}",
                        'Asset Type': data['asset_type']
                    })
            
            if current_values:
                df_current = pd.DataFrame(current_values)
                st.dataframe(df_current, hide_index=True)
            
            assets_to_remove = st.multiselect(
                "Select assets to remove:",
                list(st.session_state.portfolio.keys()),
                help="Choose one or more assets to remove from your portfolio"
            )
            
            if assets_to_remove:
                st.warning(f"⚠️ You are about to remove {len(assets_to_remove)} asset(s) from your portfolio.")
                
                if st.button("🗑️ Remove Selected Assets", type="secondary"):
                    for asset in assets_to_remove:
                        del st.session_state.portfolio[asset]
                    
                    portfolios = load_portfolios()
                    portfolios[st.session_state.username] = st.session_state.portfolio
                    save_portfolios(portfolios)
                    
                    st.markdown("""
                    <div class="success-card">
                        <strong>✅ Assets Removed Successfully!</strong><br>
                        Your portfolio has been updated and metrics recalculated.
                    </div>
                    """, unsafe_allow_html=True)
                    st.rerun()
        else:
            st.markdown("""
            <div class="info-card">
                <strong>📝 No assets to remove</strong><br>
                Your portfolio is currently empty. Add some investments first!
            </div>
            """, unsafe_allow_html=True)

def show_analytics_dashboard():
    """Display analytics dashboard"""
    
    st.markdown("### 📊 Advanced Analytics Dashboard")
    
    if st.session_state.learning_mode:
        st.markdown("""
        <div class="info-card">
            <strong>📈 Analytics Dashboard Guide:</strong><br>
            This dashboard provides professional-grade analysis tools including performance metrics, technical analysis, and AI-powered insights.
        </div>
        """, unsafe_allow_html=True)
    
    if not st.session_state.portfolio:
        st.warning("⚠️ Add assets to your portfolio to see advanced analytics.")
        return
    
    metrics = calculate_portfolio_metrics_advanced(st.session_state.portfolio)
    
    st.markdown("### 📈 Advanced Performance Metrics")
    
    perf_col1, perf_col2, perf_col3, perf_col4, perf_col5 = st.columns(5)
    
    with perf_col1:
        st.metric(
            "Annualized Return", 
            f"{metrics['annualized_return']:+.1f}%",
            help="Expected annual return based on current performance"
        )
    
    with perf_col2:
        st.metric(
            "Portfolio Beta", 
            f"{metrics['beta']:.2f}",
            delta="Market Risk" if metrics['beta'] > 1 else "Lower Risk",
            help="Portfolio volatility relative to market"
        )
    
    with perf_col3:
        st.metric(
            "Sharpe Ratio", 
            f"{metrics['sharpe_ratio']:.2f}",
            delta="Excellent" if metrics['sharpe_ratio'] > 1 else "Good" if metrics['sharpe_ratio'] > 0.5 else "Needs Improvement",
            help="Risk-adjusted return measure"
        )
    
    with perf_col4:
        st.metric(
            "Max Drawdown", 
            f"{metrics['max_drawdown']:.1f}%",
            delta="Risk Level",
            delta_color="inverse",
            help="Largest peak-to-trough decline"
        )
    
    with perf_col5:
        st.metric(
            "Portfolio Volatility", 
            f"{metrics['volatility']:.1f}%",
            help="Measure of price fluctuation"
        )
    
    # Technical analysis for top holdings
    st.markdown("### 🔍 Technical Analysis - Top Holdings")
    
    portfolio_values = []
    for symbol, data in st.session_state.portfolio.items():
        asset_info = fetch_asset_data(symbol)
        if asset_info:
            value = data['shares'] * asset_info['current_price']
            portfolio_values.append((symbol, value))
    
    portfolio_values.sort(key=lambda x: x[1], reverse=True)
    top_holdings = [item[0] for item in portfolio_values[:3]]
    
    for i, symbol in enumerate(top_holdings):
        with st.expander(f"📊 {symbol} - Technical Analysis", expanded=(i == 0)):
            indicators = calculate_technical_indicators(symbol)
            
            if indicators is not None:
                tech_col1, tech_col2, tech_col3, tech_col4 = st.columns(4)
                
                current_price = indicators['Close'].iloc[-1]
                ma20 = indicators['MA_20'].iloc[-1]
                current_rsi = indicators['RSI'].iloc[-1]
                current_macd = indicators['MACD'].iloc[-1]
                current_signal = indicators['MACD_Signal'].iloc[-1]
                
                with tech_col1:
                    price_trend = "Bullish" if current_price > ma20 else "Bearish"
                    st.metric("Price Trend", price_trend, f"${current_price:.2f}")
                
                with tech_col2:
                    rsi_status = "Overbought" if current_rsi > 70 else "Oversold" if current_rsi < 30 else "Neutral"
                    rsi_color = "🔴" if current_rsi > 70 else "🟢" if current_rsi < 30 else "🟡"
                    st.metric("RSI Signal", f"{rsi_color} {rsi_status}", f"{current_rsi:.1f}")
                
                with tech_col3:
                    macd_trend = "Bullish" if current_macd > current_signal else "Bearish"
                    macd_color = "🟢" if current_macd > current_signal else "🔴"
                    st.metric("MACD Signal", f"{macd_color} {macd_trend}")
                
                with tech_col4:
                    support_level = current_price * 0.95
                    resistance_level = current_price * 1.05
                    st.metric("Support/Resistance", f"${support_level:.2f} / ${resistance_level:.2f}")
                
                if PLOTLY_AVAILABLE:
                    fig = make_subplots(
                        rows=3, cols=1,
                        subplot_titles=(
                            f'{symbol} - Price & Moving Averages',
                            'Relative Strength Index (RSI)',
                            'MACD & Signal Line'
                        ),
                        vertical_spacing=0.08,
                        row_heights=[0.5, 0.25, 0.25]
                    )
                    
                    fig.add_trace(
                        go.Scatter(x=indicators.index, y=indicators['Close'], name='Price', line=dict(color='blue', width=2)),
                        row=1, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=indicators.index, y=indicators['MA_20'], name='MA 20', line=dict(color='orange')),
                        row=1, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=indicators.index, y=indicators['MA_50'], name='MA 50', line=dict(color='red')),
                        row=1, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(x=indicators.index, y=indicators['RSI'], name='RSI', line=dict(color='purple')),
                        row=2, col=1
                    )
                    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
                    
                    fig.add_trace(
                        go.Scatter(x=indicators.index, y=indicators['MACD'], name='MACD', line=dict(color='blue')),
                        row=3, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=indicators.index, y=indicators['MACD_Signal'], name='Signal', line=dict(color='red')),
                        row=3, col=1
                    )
                    
                    fig.update_layout(height=700, showlegend=True, title_text=f"{symbol} Technical Analysis")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.markdown("#### 📈 Price Chart")
                    st.line_chart(indicators[['Close', 'MA_20', 'MA_50']])
                    
                    simple_col1, simple_col2 = st.columns(2)
                    with simple_col1:
                        st.markdown("#### RSI")
                        st.line_chart(indicators['RSI'])
                    with simple_col2:
                        st.markdown("#### MACD")
                        st.line_chart(indicators[['MACD', 'MACD_Signal']])
    
    # Investment suggestions
    st.markdown("### 🤖 AI-Powered Investment Insights")
    suggestions = generate_investment_suggestions(st.session_state.portfolio)
    
    enhanced_suggestions = []
    
    if metrics['total_return'] > 10:
        enhanced_suggestions.append({
            'type': 'success',
            'message': f'🎉 Excellent performance! Your portfolio is up {metrics["total_return"]:.1f}%. Consider taking some profits and rebalancing.'
        })
    elif metrics['total_return'] < -5:
        enhanced_suggestions.append({
            'type': 'warning',
            'message': f'📉 Portfolio is down {abs(metrics["total_return"]):.1f}%. Review underperforming assets and consider strategic rebalancing.'
        })
    
    if metrics['volatility'] > 25:
        enhanced_suggestions.append({
            'type': 'warning',
            'message': f'⚡ High volatility detected ({metrics["volatility"]:.1f}%). Consider adding stable assets like bonds or dividend stocks.'
        })
    
    all_suggestions = enhanced_suggestions + suggestions
    
    for suggestion in all_suggestions[:6]:
        if suggestion['type'] == 'diversification':
            st.markdown(f"""
            <div class="info-card">
                <strong>🎯 Diversification:</strong> {suggestion['message']}
            </div>
            """, unsafe_allow_html=True)
        elif suggestion['type'] == 'rebalancing':
            st.markdown(f"""
            <div class="warning-card">
                <strong>⚖️ Rebalancing:</strong> {suggestion['message']}
            </div>
            """, unsafe_allow_html=True)
        elif suggestion['type'] == 'opportunity':
            st.markdown(f"""
            <div class="success-card">
                <strong>🚀 Opportunity:</strong> {suggestion['message']}
            </div>
            """, unsafe_allow_html=True)
        elif suggestion['type'] == 'success':
            st.markdown(f"""
            <div class="success-card">
                <strong>✅ Success:</strong> {suggestion['message']}
            </div>
            """, unsafe_allow_html=True)
        elif suggestion['type'] == 'warning':
            st.markdown(f"""
            <div class="warning-card">
                <strong>⚠️ Warning:</strong> {suggestion['message']}
            </div>
            """, unsafe_allow_html=True)

def show_export_import():
    """Display export/import functionality"""
    
    st.markdown("### 📁 Export & Import Portfolio")
    
    if st.session_state.learning_mode:
        st.markdown("""
        <div class="info-card">
            <strong>📁 Export/Import Guide:</strong><br>
            Backup and restore your portfolio data. Export as JSON or CSV for safekeeping, or import previously saved portfolios.
        </div>
        """, unsafe_allow_html=True)
    
    export_tab, import_tab = st.tabs(["📤 Export Portfolio", "📥 Import Portfolio"])
    
    with export_tab:
        st.markdown("### Export Your Portfolio Data")
        
        if not st.session_state.portfolio:
            st.markdown("""
            <div class="warning-card">
                <strong>📝 No portfolio data to export</strong><br>
                Add some investments to your portfolio first, then return here to export your data.
            </div>
            """, unsafe_allow_html=True)
            return
        
        export_format = st.selectbox("Choose Export Format", ["JSON", "CSV"])
        
        if export_format == "JSON":
            export_data = {
                'username': st.session_state.username,
                'export_date': datetime.now().isoformat(),
                'portfolio': st.session_state.portfolio,
                'version': '2.0'
            }
            
            json_string = json.dumps(export_data, indent=2)
            
            st.download_button(
                label="📥 Download Portfolio (JSON)",
                data=json_string,
                file_name=f"portfolio_{st.session_state.username}_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )
            
            with st.expander("👀 Preview JSON Data"):
                st.code(json_string, language="json")
        
        else:  # CSV format
            csv_data = []
            for symbol, data in st.session_state.portfolio.items():
                asset_info = fetch_asset_data(symbol)
                if asset_info:
                    csv_data.append({
                        'Symbol': symbol,
                        'Name': asset_info['name'],
                        'Shares': data['shares'],
                        'Asset_Type': data['asset_type'],
                        'Purchase_Price': data.get('purchase_price', asset_info['current_price']),
                        'Current_Price': asset_info['current_price'],
                        'Current_Value': data['shares'] * asset_info['current_price'],
                        'Added_Date': data['added_date']
                    })
            
            if csv_data:
                df = pd.DataFrame(csv_data)
                csv_string = df.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download Portfolio (CSV)",
                    data=csv_string,
                    file_name=f"portfolio_{st.session_state.username}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
                
                with st.expander("👀 Preview CSV Data"):
                    st.dataframe(df)
    
    with import_tab:
        st.markdown("### Import Portfolio Data")
        
        uploaded_file = st.file_uploader(
            "Choose a portfolio file",
            type=['json', 'csv'],
            help="Upload a previously exported portfolio file"
        )
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.json'):
                    content = uploaded_file.read()
                    data = json.loads(content)
                    
                    if 'portfolio' in data:
                        st.markdown("""
                        <div class="success-card">
                            <strong>✅ Valid portfolio file detected!</strong><br>
                            Ready to import your portfolio data.
                        </div>
                        """, unsafe_allow_html=True)
                        
                        with st.expander("👀 Preview Import Data"):
                            st.json(data['portfolio'])
                        
                        import_option = st.selectbox(
                            "Import Option",
                            ["Replace Current Portfolio", "Merge with Current Portfolio"]
                        )
                        
                        if st.button("🚀 Import Portfolio", type="primary"):
                            if import_option == "Replace Current Portfolio":
                                st.session_state.portfolio = data['portfolio']
                            else:
                                for symbol, asset_data in data['portfolio'].items():
                                    st.session_state.portfolio[symbol] = asset_data
                            
                            portfolios = load_portfolios()
                            portfolios[st.session_state.username] = st.session_state.portfolio
                            save_portfolios(portfolios)
                            
                            st.markdown("""
                            <div class="success-card">
                                <strong>🎉 Portfolio imported successfully!</strong><br>
                                Your portfolio has been updated with the imported data.
                            </div>
                            """, unsafe_allow_html=True)
                            st.rerun()
                    else:
                        st.error("❌ Invalid portfolio file format.")
                
                elif uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                    
                    required_columns = ['Symbol', 'Shares', 'Asset_Type']
                    if all(col in df.columns for col in required_columns):
                        st.markdown("""
                        <div class="success-card">
                            <strong>✅ Valid CSV file detected!</strong><br>
                            Ready to import your portfolio data.
                        </div>
                        """, unsafe_allow_html=True)
                        
                        with st.expander("👀 Preview Import Data"):
                            st.dataframe(df)
                        
                        csv_import_option = st.selectbox(
                            "Import Option",
                            ["Replace Current Portfolio", "Merge with Current Portfolio"],
                            key="csv_import_option"
                        )
                        
                        if st.button("🚀 Import CSV Portfolio", type="primary"):
                            new_portfolio = {}
                            
                            for _, row in df.iterrows():
                                symbol = row['Symbol']
                                new_portfolio[symbol] = {
                                    'shares': float(row['Shares']),
                                    'asset_type': row['Asset_Type'],
                                    'purchase_price': float(row.get('Purchase_Price', 100.0)),
                                    'added_date': datetime.now().isoformat()
                                }
                            
                            if csv_import_option == "Replace Current Portfolio":
                                st.session_state.portfolio = new_portfolio
                            else:
                                for symbol, asset_data in new_portfolio.items():
                                    st.session_state.portfolio[symbol] = asset_data
                            
                            portfolios = load_portfolios()
                            portfolios[st.session_state.username] = st.session_state.portfolio
                            save_portfolios(portfolios)
                            
                            st.markdown("""
                            <div class="success-card">
                                <strong>🎉 CSV portfolio imported successfully!</strong><br>
                                Your portfolio has been updated with the imported data.
                            </div>
                            """, unsafe_allow_html=True)
                            st.rerun()
                    else:
                        st.error(f"❌ CSV file must contain columns: {', '.join(required_columns)}")
                        
            except Exception as e:
                st.error(f"❌ Error importing file: {str(e)}")

if __name__ == "__main__":
    main(), 'kr', 'zł', 'Kč', 'Ft']:
                            value_str = value_str.replace(symbol_char, '')
                        value_str = value_str.replace(',', '')
                        current_values.append(float(value_str))
                        symbols.append(item['Symbol'])
                    except:
                        continue
                
                if current_values:
                    fig_pie = px.pie(
                        values=current_values,
                        names=symbols,
                        title=f"Holdings Distribution ({base_currency})",
                        hole=0.4
                    )
                    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                    fig_pie.update_layout(height=400)
                    st.plotly_chart(fig_pie, use_container_width=True)
            
            with col_chart2:
                st.markdown("#### 📈 Performance by Asset")
                return_values = []
                
                for item in portfolio_data:
                    try:
                        return_str = item['Return %'].replace('%', '').replace('+', '')
                        return_values.append(float(return_str))
                    except:
                        return_values.append(0.0)
                
                if return_values and symbols:
                    fig_returns = px.bar(
                        x=symbols,
                        y=return_values,
                        title="Return % by Holding",
                        color=return_values,
                        color_continuous_scale=['red', 'yellow', 'green']
                    )
                    fig_returns.update_layout(height=400, xaxis_title="Assets", yaxis_title="Return %")
                    st.plotly_chart(fig_returns, use_container_width=True)
        else:
            st.markdown("#### 📊 Simple Visualization")
            current_values = []
            symbols = []
            
            for item in portfolio_data:
                try:
                    value_str = item['Current Value']
                    # Remove currency symbols and commas more robustly
                    for symbol_char in ['"{return_pct:+.1f}%",
                'P&L': f"${current_value - invested_value:+,.2f}",
                'Asset Type': data['asset_type'],
                'Weight %': f"{(current_value / metrics['total_current_value']) * 100:.1f}%" if metrics['total_current_value'] > 0 else "0.0%"
            })
    
    if portfolio_data:
        st.markdown("### 📊 Holdings Breakdown")
        df = pd.DataFrame(portfolio_data)
        st.dataframe(df, use_container_width=True, height=400)
        
        if PLOTLY_AVAILABLE:
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                st.markdown("#### 🥧 Portfolio Allocation")
                # Extract numeric values safely
                current_values = []
                symbols = []
                
                for item in portfolio_data:
                    try:
                        value_str = item['Current Value'].replace('$', '').replace(',', '')
                        current_values.append(float(value_str))
                        symbols.append(item['Symbol'])
                    except:
                        continue
                
                if current_values:
                    fig_pie = px.pie(
                        values=current_values,
                        names=symbols,
                        title="Current Holdings Distribution",
                        hole=0.4
                    )
                    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                    fig_pie.update_layout(height=400)
                    st.plotly_chart(fig_pie, use_container_width=True)
            
            with col_chart2:
                st.markdown("#### 📈 Performance by Asset")
                return_values = []
                
                for item in portfolio_data:
                    try:
                        return_str = item['Return %'].replace('%', '').replace('+', '')
                        return_values.append(float(return_str))
                    except:
                        return_values.append(0.0)
                
                if return_values and symbols:
                    fig_returns = px.bar(
                        x=symbols,
                        y=return_values,
                        title="Return % by Holding",
                        color=return_values,
                        color_continuous_scale=['red', 'yellow', 'green']
                    )
                    fig_returns.update_layout(height=400, xaxis_title="Assets", yaxis_title="Return %")
                    st.plotly_chart(fig_returns, use_container_width=True)
        else:
            st.markdown("#### 📊 Simple Visualization")
            current_values = []
            symbols = []
            
            for item in portfolio_data:
                try:
                    value_str = item['Current Value'].replace('$', '').replace(',', '')
                    current_values.append(float(value_str))
                    symbols.append(item['Symbol'])
                except:
                    continue
            
            if current_values and symbols:
                chart_data = pd.DataFrame({
                    'Symbol': symbols,
                    'Current Value': current_values
                })
                st.bar_chart(chart_data.set_index('Symbol'))

def show_asset_management():
    """Display asset management page"""
    
    st.markdown("### 🎯 Manage Portfolio Assets")
    
    if st.session_state.learning_mode:
        st.markdown("""
        <div class="info-card">
            <strong>🎯 Asset Management Guide:</strong><br>
            Use this section to build and maintain your investment portfolio. Add assets, track purchase prices, and manage your holdings.
        </div>
        """, unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["➕ Add Assets", "➖ Remove Assets"])
    
    with tab1:
        st.markdown("### Add New Investment")
        
        col1, col2 = st.columns(2)
        
        with col1:
            popular_assets = get_popular_assets()
            
            # Create categories for better organization
            asset_categories = {
                "🏢 Large Cap Stocks": [k for k in popular_assets.keys() if k in ["AAPL", "GOOGL", "GOOG", "MSFT", "AMZN", "TSLA", "NVDA", "META", "NFLX", "JPM", "JNJ", "V", "WMT", "PG", "UNH", "HD", "MA", "BAC", "ADBE", "CRM", "XOM", "CVX", "KO", "PFE", "INTC", "CSCO", "VZ", "MRK", "ABT", "TMO"]],
                "🚀 Growth & Tech": [k for k in popular_assets.keys() if k in ["AMD", "PYPL", "SHOP", "SQ", "ROKU", "ZM", "SNOW", "PLTR", "CRWD", "NET"]],
                "🏛️ Vanguard ETFs": [k for k in popular_assets.keys() if k.startswith("V") and k in ["VTI", "VEA", "VWO", "VTV", "VUG", "VIG", "VYM", "VXUS", "VOO", "VT", "VB", "VO", "VGT", "VHT", "VFH", "VDE", "VAW", "VIS", "VCR", "VDC", "VPU", "VNQ"]],
                "⚫ BlackRock iShares": [k for k in popular_assets.keys() if k in ["QQQ", "IWM", "EFA", "EEM", "IVV", "IEFA", "IEMG", "IJH", "IJR", "IVW", "IVE", "IWF", "IWD", "ITOT", "IXUS", "IYY", "IWB", "IWV", "ACWI", "ACWX"]],
                "🎯 Sector ETFs": [k for k in popular_assets.keys() if k.startswith("XL") or k in ["IBB", "SOXX", "SKYY", "HACK", "ROBO", "ARKK", "ARKQ", "ARKW", "ARKG", "ARKF"]],
                "🏦 Bonds & Fixed Income": [k for k in popular_assets.keys() if k in ["AGG", "BND", "TLT", "IEF", "SHY", "LQD", "HYG", "JNK", "TIP", "VTEB", "MUB", "EMB", "BNDX", "VGIT", "VGSH", "VGLT", "VCIT", "VCSH", "BSV", "BIV", "BLV"]],
                "💎 Fidelity Funds": [k for k in popular_assets.keys() if k.startswith("F") and len(k) > 2],
                "🥇 Commodities": [k for k in popular_assets.keys() if k in ["GLD", "SLV", "USO", "UNG", "DBA", "DBC", "PDBC", "GSG", "COMT", "IAU", "SGOL", "PPLT", "PALL", "CORN", "WEAT", "SOYB", "CANE", "COW"]],
                "🌍 International": [k for k in popular_assets.keys() if k.startswith("EW") or k in ["FXI", "INDA", "RSX", "ASHR", "MCHI", "KWEB", "YINN"]],
                "₿ Major Cryptocurrencies": [k for k in popular_assets.keys() if k.endswith("-USD") and k in ["BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "ADA-USD", "SOL-USD", "DOGE-USD", "DOT-USD", "MATIC-USD", "SHIB-USD", "LTC-USD", "TRX-USD", "AVAX-USD", "LINK-USD", "ATOM-USD"]],
                "🔗 DeFi & Web3": [k for k in popular_assets.keys() if k.endswith("-USD") and k in ["XLM-USD", "ALGO-USD", "VET-USD", "ICP-USD", "FIL-USD", "SAND-USD", "MANA-USD", "AXS-USD", "ENJ-USD", "CHZ-USD", "FLOW-USD", "THETA-USD", "XTZ-USD", "EGLD-USD", "NEAR-USD"]],
                "📈 Crypto ETFs": [k for k in popular_assets.keys() if k in ["BITO", "BITI", "GBTC", "ETHE", "GDLC", "BITQ", "BLOK", "LEGR"]],
                "💱 Currency ETFs": [k for k in popular_assets.keys() if k.startswith("FX") or k in ["UUP", "UDN"]],
                "💰 Dividend ETFs": [k for k in popular_assets.keys() if k in ["SCHD", "DVY", "VYM", "NOBL", "DGRO", "HDV", "SPHD", "PEY", "RDVY", "FDL"]],
                "🌱 ESG & Clean Energy": [k for k in popular_assets.keys() if k in ["ESG", "ESGU", "ESGD", "ESGE", "VSGX", "ESGV", "SUSA", "DSI", "ICLN", "PBW", "QCLN"]],
                "📊 Volatility & Leverage": [k for k in popular_assets.keys() if k in ["VIX", "UVXY", "SVXY", "VXX", "VIXY", "TQQQ", "SQQQ", "UPRO", "SPXU", "TNA", "TZA", "TECL", "TECS", "FAS", "FAZ"]]
            }
            
            selected_category = st.selectbox(
                "Choose Asset Category",
                [""] + list(asset_categories.keys()),
                help="Browse assets by category for easier selection"
            )
            
            if selected_category:
                available_assets = asset_categories[selected_category]
                selected_popular = st.selectbox(
                    f"Assets in {selected_category}",
                    [""] + available_assets,
                    format_func=lambda x: f"{x} - {popular_assets[x]}" if x else "",
                    help="Select from assets in this category"
                )
            else:
                selected_popular = st.selectbox(
                    "Or search all assets",
                    [""] + list(popular_assets.keys()),
                    format_func=lambda x: f"{x} - {popular_assets[x]}" if x else "",
                    help="Search through all available assets"
                )
            
            custom_symbol = st.text_input(
                "Or Enter Custom Symbol",
                placeholder="e.g., AAPL, BTC-USD, GLD.L (London)",
                help="Enter any valid symbol including international exchanges"
            )
            
            symbol_to_use = ""
            if selected_popular:
                symbol_to_use = selected_popular
            elif custom_symbol:
                symbol_to_use = custom_symbol.upper()
        
        with col2:
            shares = st.number_input(
                "Number of Shares/Units",
                min_value=0.001,
                value=1.0,
                step=0.1,
                help="Enter the quantity you own"
            )
            
            purchase_price = st.number_input(
                "Purchase Price per Share ($)",
                min_value=0.01,
                value=100.0,
                step=0.01,
                help="Enter the average price you paid per share"
            )
            
            # NEW: Currency selection
            currencies = get_currency_list()
            selected_currency = st.selectbox(
                "Currency",
                list(currencies.keys()),
                index=0,  # Default to USD
                format_func=lambda x: f"{x} - {currencies[x]['name']}",
                help="Select the currency for this asset"
            )
            
            asset_type = st.selectbox(
                "Asset Type",
                ["Stock", "ETF", "Cryptocurrency", "Bond", "Commodity", "Index Fund", "Mutual Fund", "REIT", "Other"]
            )
            
            if st.session_state.learning_mode:
                st.markdown("""
                <div class="info-card">
                    <strong>💡 Pro Tip:</strong><br>
                    • Select the correct currency for international assets<br>
                    • Purchase price should be in the selected currency<br>
                    • The system will convert to your base currency for analysis
                </div>
                """, unsafe_allow_html=True)
        
        if symbol_to_use:
            asset_info = fetch_asset_data(symbol_to_use)
            if asset_info:
                current_price = asset_info['current_price']
                
                # Convert prices for display if different currency
                if selected_currency != "USD":
                    exchange_rate = get_exchange_rate("USD", selected_currency)
                    current_price_display = current_price * exchange_rate
                    purchase_price_display = purchase_price
                else:
                    current_price_display = current_price
                    purchase_price_display = purchase_price
                
                potential_return = ((current_price_display - purchase_price_display) / purchase_price_display) * 100 if purchase_price_display > 0 else 0
                
                # Create three columns for metrics display
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                
                with metric_col1:
                    currency_symbol = currencies[selected_currency]["symbol"]
                    st.metric("Current Price", f"{currency_symbol}{current_price_display:.2f}")
                with metric_col2:
                    st.metric("Your Purchase Price", f"{currency_symbol}{purchase_price_display:.2f}")
                with metric_col3:
                    st.metric("Potential Return", f"{potential_return:+.1f}%", 
                             delta=f"{currency_symbol}{(current_price_display - purchase_price_display) * shares:+.2f}")
                
                if st.button("✅ Add to Portfolio", type="primary", use_container_width=True):
                    st.session_state.portfolio[symbol_to_use] = {
                        'shares': shares,
                        'asset_type': asset_type,
                        'purchase_price': purchase_price,
                        'currency': selected_currency,  # NEW: Store currency
                        'added_date': datetime.now().isoformat()
                    }
                    
                    portfolios = load_portfolios()
                    portfolios[st.session_state.username] = st.session_state.portfolio
                    save_portfolios(portfolios)
                    
                    st.markdown("""
                    <div class="success-card">
                        <strong>🎉 Asset Added Successfully!</strong><br>
                        Your portfolio metrics are being recalculated with the new asset allocation and currency conversion.
                    </div>
                    """, unsafe_allow_html=True)
                    st.rerun()
            else:
                st.error(f"❌ Could not find asset data for '{symbol_to_use}'. Please check the symbol.")
    
    with tab2:
        st.markdown("### Remove Assets from Portfolio")
        
        if st.session_state.portfolio:
            current_values = []
            for symbol, data in st.session_state.portfolio.items():
                asset_info = fetch_asset_data(symbol)
                if asset_info:
                    current_value = data['shares'] * asset_info['current_price']
                    current_values.append({
                        'Symbol': symbol,
                        'Shares': f"{data['shares']:.3f}",
                        'Current Value': f"${current_value:,.2f}",
                        'Asset Type': data['asset_type']
                    })
            
            if current_values:
                df_current = pd.DataFrame(current_values)
                st.dataframe(df_current, hide_index=True)
            
            assets_to_remove = st.multiselect(
                "Select assets to remove:",
                list(st.session_state.portfolio.keys()),
                help="Choose one or more assets to remove from your portfolio"
            )
            
            if assets_to_remove:
                st.warning(f"⚠️ You are about to remove {len(assets_to_remove)} asset(s) from your portfolio.")
                
                if st.button("🗑️ Remove Selected Assets", type="secondary"):
                    for asset in assets_to_remove:
                        del st.session_state.portfolio[asset]
                    
                    portfolios = load_portfolios()
                    portfolios[st.session_state.username] = st.session_state.portfolio
                    save_portfolios(portfolios)
                    
                    st.markdown("""
                    <div class="success-card">
                        <strong>✅ Assets Removed Successfully!</strong><br>
                        Your portfolio has been updated and metrics recalculated.
                    </div>
                    """, unsafe_allow_html=True)
                    st.rerun()
        else:
            st.markdown("""
            <div class="info-card">
                <strong>📝 No assets to remove</strong><br>
                Your portfolio is currently empty. Add some investments first!
            </div>
            """, unsafe_allow_html=True)

def show_analytics_dashboard():
    """Display analytics dashboard"""
    
    st.markdown("### 📊 Advanced Analytics Dashboard")
    
    if st.session_state.learning_mode:
        st.markdown("""
        <div class="info-card">
            <strong>📈 Analytics Dashboard Guide:</strong><br>
            This dashboard provides professional-grade analysis tools including performance metrics, technical analysis, and AI-powered insights.
        </div>
        """, unsafe_allow_html=True)
    
    if not st.session_state.portfolio:
        st.warning("⚠️ Add assets to your portfolio to see advanced analytics.")
        return
    
    metrics = calculate_portfolio_metrics_advanced(st.session_state.portfolio)
    
    st.markdown("### 📈 Advanced Performance Metrics")
    
    perf_col1, perf_col2, perf_col3, perf_col4, perf_col5 = st.columns(5)
    
    with perf_col1:
        st.metric(
            "Annualized Return", 
            f"{metrics['annualized_return']:+.1f}%",
            help="Expected annual return based on current performance"
        )
    
    with perf_col2:
        st.metric(
            "Portfolio Beta", 
            f"{metrics['beta']:.2f}",
            delta="Market Risk" if metrics['beta'] > 1 else "Lower Risk",
            help="Portfolio volatility relative to market"
        )
    
    with perf_col3:
        st.metric(
            "Sharpe Ratio", 
            f"{metrics['sharpe_ratio']:.2f}",
            delta="Excellent" if metrics['sharpe_ratio'] > 1 else "Good" if metrics['sharpe_ratio'] > 0.5 else "Needs Improvement",
            help="Risk-adjusted return measure"
        )
    
    with perf_col4:
        st.metric(
            "Max Drawdown", 
            f"{metrics['max_drawdown']:.1f}%",
            delta="Risk Level",
            delta_color="inverse",
            help="Largest peak-to-trough decline"
        )
    
    with perf_col5:
        st.metric(
            "Portfolio Volatility", 
            f"{metrics['volatility']:.1f}%",
            help="Measure of price fluctuation"
        )
    
    # Technical analysis for top holdings
    st.markdown("### 🔍 Technical Analysis - Top Holdings")
    
    portfolio_values = []
    for symbol, data in st.session_state.portfolio.items():
        asset_info = fetch_asset_data(symbol)
        if asset_info:
            value = data['shares'] * asset_info['current_price']
            portfolio_values.append((symbol, value))
    
    portfolio_values.sort(key=lambda x: x[1], reverse=True)
    top_holdings = [item[0] for item in portfolio_values[:3]]
    
    for i, symbol in enumerate(top_holdings):
        with st.expander(f"📊 {symbol} - Technical Analysis", expanded=(i == 0)):
            indicators = calculate_technical_indicators(symbol)
            
            if indicators is not None:
                tech_col1, tech_col2, tech_col3, tech_col4 = st.columns(4)
                
                current_price = indicators['Close'].iloc[-1]
                ma20 = indicators['MA_20'].iloc[-1]
                current_rsi = indicators['RSI'].iloc[-1]
                current_macd = indicators['MACD'].iloc[-1]
                current_signal = indicators['MACD_Signal'].iloc[-1]
                
                with tech_col1:
                    price_trend = "Bullish" if current_price > ma20 else "Bearish"
                    st.metric("Price Trend", price_trend, f"${current_price:.2f}")
                
                with tech_col2:
                    rsi_status = "Overbought" if current_rsi > 70 else "Oversold" if current_rsi < 30 else "Neutral"
                    rsi_color = "🔴" if current_rsi > 70 else "🟢" if current_rsi < 30 else "🟡"
                    st.metric("RSI Signal", f"{rsi_color} {rsi_status}", f"{current_rsi:.1f}")
                
                with tech_col3:
                    macd_trend = "Bullish" if current_macd > current_signal else "Bearish"
                    macd_color = "🟢" if current_macd > current_signal else "🔴"
                    st.metric("MACD Signal", f"{macd_color} {macd_trend}")
                
                with tech_col4:
                    support_level = current_price * 0.95
                    resistance_level = current_price * 1.05
                    st.metric("Support/Resistance", f"${support_level:.2f} / ${resistance_level:.2f}")
                
                if PLOTLY_AVAILABLE:
                    fig = make_subplots(
                        rows=3, cols=1,
                        subplot_titles=(
                            f'{symbol} - Price & Moving Averages',
                            'Relative Strength Index (RSI)',
                            'MACD & Signal Line'
                        ),
                        vertical_spacing=0.08,
                        row_heights=[0.5, 0.25, 0.25]
                    )
                    
                    fig.add_trace(
                        go.Scatter(x=indicators.index, y=indicators['Close'], name='Price', line=dict(color='blue', width=2)),
                        row=1, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=indicators.index, y=indicators['MA_20'], name='MA 20', line=dict(color='orange')),
                        row=1, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=indicators.index, y=indicators['MA_50'], name='MA 50', line=dict(color='red')),
                        row=1, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(x=indicators.index, y=indicators['RSI'], name='RSI', line=dict(color='purple')),
                        row=2, col=1
                    )
                    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
                    
                    fig.add_trace(
                        go.Scatter(x=indicators.index, y=indicators['MACD'], name='MACD', line=dict(color='blue')),
                        row=3, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=indicators.index, y=indicators['MACD_Signal'], name='Signal', line=dict(color='red')),
                        row=3, col=1
                    )
                    
                    fig.update_layout(height=700, showlegend=True, title_text=f"{symbol} Technical Analysis")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.markdown("#### 📈 Price Chart")
                    st.line_chart(indicators[['Close', 'MA_20', 'MA_50']])
                    
                    simple_col1, simple_col2 = st.columns(2)
                    with simple_col1:
                        st.markdown("#### RSI")
                        st.line_chart(indicators['RSI'])
                    with simple_col2:
                        st.markdown("#### MACD")
                        st.line_chart(indicators[['MACD', 'MACD_Signal']])
    
    # Investment suggestions
    st.markdown("### 🤖 AI-Powered Investment Insights")
    suggestions = generate_investment_suggestions(st.session_state.portfolio)
    
    enhanced_suggestions = []
    
    if metrics['total_return'] > 10:
        enhanced_suggestions.append({
            'type': 'success',
            'message': f'🎉 Excellent performance! Your portfolio is up {metrics["total_return"]:.1f}%. Consider taking some profits and rebalancing.'
        })
    elif metrics['total_return'] < -5:
        enhanced_suggestions.append({
            'type': 'warning',
            'message': f'📉 Portfolio is down {abs(metrics["total_return"]):.1f}%. Review underperforming assets and consider strategic rebalancing.'
        })
    
    if metrics['volatility'] > 25:
        enhanced_suggestions.append({
            'type': 'warning',
            'message': f'⚡ High volatility detected ({metrics["volatility"]:.1f}%). Consider adding stable assets like bonds or dividend stocks.'
        })
    
    all_suggestions = enhanced_suggestions + suggestions
    
    for suggestion in all_suggestions[:6]:
        if suggestion['type'] == 'diversification':
            st.markdown(f"""
            <div class="info-card">
                <strong>🎯 Diversification:</strong> {suggestion['message']}
            </div>
            """, unsafe_allow_html=True)
        elif suggestion['type'] == 'rebalancing':
            st.markdown(f"""
            <div class="warning-card">
                <strong>⚖️ Rebalancing:</strong> {suggestion['message']}
            </div>
            """, unsafe_allow_html=True)
        elif suggestion['type'] == 'opportunity':
            st.markdown(f"""
            <div class="success-card">
                <strong>🚀 Opportunity:</strong> {suggestion['message']}
            </div>
            """, unsafe_allow_html=True)
        elif suggestion['type'] == 'success':
            st.markdown(f"""
            <div class="success-card">
                <strong>✅ Success:</strong> {suggestion['message']}
            </div>
            """, unsafe_allow_html=True)
        elif suggestion['type'] == 'warning':
            st.markdown(f"""
            <div class="warning-card">
                <strong>⚠️ Warning:</strong> {suggestion['message']}
            </div>
            """, unsafe_allow_html=True)

def show_export_import():
    """Display export/import functionality"""
    
    st.markdown("### 📁 Export & Import Portfolio")
    
    if st.session_state.learning_mode:
        st.markdown("""
        <div class="info-card">
            <strong>📁 Export/Import Guide:</strong><br>
            Backup and restore your portfolio data. Export as JSON or CSV for safekeeping, or import previously saved portfolios.
        </div>
        """, unsafe_allow_html=True)
    
    export_tab, import_tab = st.tabs(["📤 Export Portfolio", "📥 Import Portfolio"])
    
    with export_tab:
        st.markdown("### Export Your Portfolio Data")
        
        if not st.session_state.portfolio:
            st.markdown("""
            <div class="warning-card">
                <strong>📝 No portfolio data to export</strong><br>
                Add some investments to your portfolio first, then return here to export your data.
            </div>
            """, unsafe_allow_html=True)
            return
        
        export_format = st.selectbox("Choose Export Format", ["JSON", "CSV"])
        
        if export_format == "JSON":
            export_data = {
                'username': st.session_state.username,
                'export_date': datetime.now().isoformat(),
                'portfolio': st.session_state.portfolio,
                'version': '2.0'
            }
            
            json_string = json.dumps(export_data, indent=2)
            
            st.download_button(
                label="📥 Download Portfolio (JSON)",
                data=json_string,
                file_name=f"portfolio_{st.session_state.username}_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )
            
            with st.expander("👀 Preview JSON Data"):
                st.code(json_string, language="json")
        
        else:  # CSV format
            csv_data = []
            for symbol, data in st.session_state.portfolio.items():
                asset_info = fetch_asset_data(symbol)
                if asset_info:
                    csv_data.append({
                        'Symbol': symbol,
                        'Name': asset_info['name'],
                        'Shares': data['shares'],
                        'Asset_Type': data['asset_type'],
                        'Purchase_Price': data.get('purchase_price', asset_info['current_price']),
                        'Current_Price': asset_info['current_price'],
                        'Current_Value': data['shares'] * asset_info['current_price'],
                        'Added_Date': data['added_date']
                    })
            
            if csv_data:
                df = pd.DataFrame(csv_data)
                csv_string = df.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download Portfolio (CSV)",
                    data=csv_string,
                    file_name=f"portfolio_{st.session_state.username}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
                
                with st.expander("👀 Preview CSV Data"):
                    st.dataframe(df)
    
    with import_tab:
        st.markdown("### Import Portfolio Data")
        
        uploaded_file = st.file_uploader(
            "Choose a portfolio file",
            type=['json', 'csv'],
            help="Upload a previously exported portfolio file"
        )
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.json'):
                    content = uploaded_file.read()
                    data = json.loads(content)
                    
                    if 'portfolio' in data:
                        st.markdown("""
                        <div class="success-card">
                            <strong>✅ Valid portfolio file detected!</strong><br>
                            Ready to import your portfolio data.
                        </div>
                        """, unsafe_allow_html=True)
                        
                        with st.expander("👀 Preview Import Data"):
                            st.json(data['portfolio'])
                        
                        import_option = st.selectbox(
                            "Import Option",
                            ["Replace Current Portfolio", "Merge with Current Portfolio"]
                        )
                        
                        if st.button("🚀 Import Portfolio", type="primary"):
                            if import_option == "Replace Current Portfolio":
                                st.session_state.portfolio = data['portfolio']
                            else:
                                for symbol, asset_data in data['portfolio'].items():
                                    st.session_state.portfolio[symbol] = asset_data
                            
                            portfolios = load_portfolios()
                            portfolios[st.session_state.username] = st.session_state.portfolio
                            save_portfolios(portfolios)
                            
                            st.markdown("""
                            <div class="success-card">
                                <strong>🎉 Portfolio imported successfully!</strong><br>
                                Your portfolio has been updated with the imported data.
                            </div>
                            """, unsafe_allow_html=True)
                            st.rerun()
                    else:
                        st.error("❌ Invalid portfolio file format.")
                
                elif uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                    
                    required_columns = ['Symbol', 'Shares', 'Asset_Type']
                    if all(col in df.columns for col in required_columns):
                        st.markdown("""
                        <div class="success-card">
                            <strong>✅ Valid CSV file detected!</strong><br>
                            Ready to import your portfolio data.
                        </div>
                        """, unsafe_allow_html=True)
                        
                        with st.expander("👀 Preview Import Data"):
                            st.dataframe(df)
                        
                        csv_import_option = st.selectbox(
                            "Import Option",
                            ["Replace Current Portfolio", "Merge with Current Portfolio"],
                            key="csv_import_option"
                        )
                        
                        if st.button("🚀 Import CSV Portfolio", type="primary"):
                            new_portfolio = {}
                            
                            for _, row in df.iterrows():
                                symbol = row['Symbol']
                                new_portfolio[symbol] = {
                                    'shares': float(row['Shares']),
                                    'asset_type': row['Asset_Type'],
                                    'purchase_price': float(row.get('Purchase_Price', 100.0)),
                                    'added_date': datetime.now().isoformat()
                                }
                            
                            if csv_import_option == "Replace Current Portfolio":
                                st.session_state.portfolio = new_portfolio
                            else:
                                for symbol, asset_data in new_portfolio.items():
                                    st.session_state.portfolio[symbol] = asset_data
                            
                            portfolios = load_portfolios()
                            portfolios[st.session_state.username] = st.session_state.portfolio
                            save_portfolios(portfolios)
                            
                            st.markdown("""
                            <div class="success-card">
                                <strong>🎉 CSV portfolio imported successfully!</strong><br>
                                Your portfolio has been updated with the imported data.
                            </div>
                            """, unsafe_allow_html=True)
                            st.rerun()
                    else:
                        st.error(f"❌ CSV file must contain columns: {', '.join(required_columns)}")
                        
            except Exception as e:
                st.error(f"❌ Error importing file: {str(e)}")

if __name__ == "__main__":
    main(), '€', '£', '¥', '₹', '₩', 'C"{return_pct:+.1f}%",
                'P&L': f"${current_value - invested_value:+,.2f}",
                'Asset Type': data['asset_type'],
                'Weight %': f"{(current_value / metrics['total_current_value']) * 100:.1f}%" if metrics['total_current_value'] > 0 else "0.0%"
            })
    
    if portfolio_data:
        st.markdown("### 📊 Holdings Breakdown")
        df = pd.DataFrame(portfolio_data)
        st.dataframe(df, use_container_width=True, height=400)
        
        if PLOTLY_AVAILABLE:
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                st.markdown("#### 🥧 Portfolio Allocation")
                # Extract numeric values safely
                current_values = []
                symbols = []
                
                for item in portfolio_data:
                    try:
                        value_str = item['Current Value'].replace('$', '').replace(',', '')
                        current_values.append(float(value_str))
                        symbols.append(item['Symbol'])
                    except:
                        continue
                
                if current_values:
                    fig_pie = px.pie(
                        values=current_values,
                        names=symbols,
                        title="Current Holdings Distribution",
                        hole=0.4
                    )
                    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                    fig_pie.update_layout(height=400)
                    st.plotly_chart(fig_pie, use_container_width=True)
            
            with col_chart2:
                st.markdown("#### 📈 Performance by Asset")
                return_values = []
                
                for item in portfolio_data:
                    try:
                        return_str = item['Return %'].replace('%', '').replace('+', '')
                        return_values.append(float(return_str))
                    except:
                        return_values.append(0.0)
                
                if return_values and symbols:
                    fig_returns = px.bar(
                        x=symbols,
                        y=return_values,
                        title="Return % by Holding",
                        color=return_values,
                        color_continuous_scale=['red', 'yellow', 'green']
                    )
                    fig_returns.update_layout(height=400, xaxis_title="Assets", yaxis_title="Return %")
                    st.plotly_chart(fig_returns, use_container_width=True)
        else:
            st.markdown("#### 📊 Simple Visualization")
            current_values = []
            symbols = []
            
            for item in portfolio_data:
                try:
                    value_str = item['Current Value'].replace('$', '').replace(',', '')
                    current_values.append(float(value_str))
                    symbols.append(item['Symbol'])
                except:
                    continue
            
            if current_values and symbols:
                chart_data = pd.DataFrame({
                    'Symbol': symbols,
                    'Current Value': current_values
                })
                st.bar_chart(chart_data.set_index('Symbol'))

def show_asset_management():
    """Display asset management page"""
    
    st.markdown("### 🎯 Manage Portfolio Assets")
    
    if st.session_state.learning_mode:
        st.markdown("""
        <div class="info-card">
            <strong>🎯 Asset Management Guide:</strong><br>
            Use this section to build and maintain your investment portfolio. Add assets, track purchase prices, and manage your holdings.
        </div>
        """, unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["➕ Add Assets", "➖ Remove Assets"])
    
    with tab1:
        st.markdown("### Add New Investment")
        
        col1, col2 = st.columns(2)
        
        with col1:
            popular_assets = get_popular_assets()
            
            # Create categories for better organization
            asset_categories = {
                "🏢 Large Cap Stocks": [k for k in popular_assets.keys() if k in ["AAPL", "GOOGL", "GOOG", "MSFT", "AMZN", "TSLA", "NVDA", "META", "NFLX", "JPM", "JNJ", "V", "WMT", "PG", "UNH", "HD", "MA", "BAC", "ADBE", "CRM", "XOM", "CVX", "KO", "PFE", "INTC", "CSCO", "VZ", "MRK", "ABT", "TMO"]],
                "🚀 Growth & Tech": [k for k in popular_assets.keys() if k in ["AMD", "PYPL", "SHOP", "SQ", "ROKU", "ZM", "SNOW", "PLTR", "CRWD", "NET"]],
                "🏛️ Vanguard ETFs": [k for k in popular_assets.keys() if k.startswith("V") and k in ["VTI", "VEA", "VWO", "VTV", "VUG", "VIG", "VYM", "VXUS", "VOO", "VT", "VB", "VO", "VGT", "VHT", "VFH", "VDE", "VAW", "VIS", "VCR", "VDC", "VPU", "VNQ"]],
                "⚫ BlackRock iShares": [k for k in popular_assets.keys() if k in ["QQQ", "IWM", "EFA", "EEM", "IVV", "IEFA", "IEMG", "IJH", "IJR", "IVW", "IVE", "IWF", "IWD", "ITOT", "IXUS", "IYY", "IWB", "IWV", "ACWI", "ACWX"]],
                "🎯 Sector ETFs": [k for k in popular_assets.keys() if k.startswith("XL") or k in ["IBB", "SOXX", "SKYY", "HACK", "ROBO", "ARKK", "ARKQ", "ARKW", "ARKG", "ARKF"]],
                "🏦 Bonds & Fixed Income": [k for k in popular_assets.keys() if k in ["AGG", "BND", "TLT", "IEF", "SHY", "LQD", "HYG", "JNK", "TIP", "VTEB", "MUB", "EMB", "BNDX", "VGIT", "VGSH", "VGLT", "VCIT", "VCSH", "BSV", "BIV", "BLV"]],
                "💎 Fidelity Funds": [k for k in popular_assets.keys() if k.startswith("F") and len(k) > 2],
                "🥇 Commodities": [k for k in popular_assets.keys() if k in ["GLD", "SLV", "USO", "UNG", "DBA", "DBC", "PDBC", "GSG", "COMT", "IAU", "SGOL", "PPLT", "PALL", "CORN", "WEAT", "SOYB", "CANE", "COW"]],
                "🌍 International": [k for k in popular_assets.keys() if k.startswith("EW") or k in ["FXI", "INDA", "RSX", "ASHR", "MCHI", "KWEB", "YINN"]],
                "₿ Major Cryptocurrencies": [k for k in popular_assets.keys() if k.endswith("-USD") and k in ["BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "ADA-USD", "SOL-USD", "DOGE-USD", "DOT-USD", "MATIC-USD", "SHIB-USD", "LTC-USD", "TRX-USD", "AVAX-USD", "LINK-USD", "ATOM-USD"]],
                "🔗 DeFi & Web3": [k for k in popular_assets.keys() if k.endswith("-USD") and k in ["XLM-USD", "ALGO-USD", "VET-USD", "ICP-USD", "FIL-USD", "SAND-USD", "MANA-USD", "AXS-USD", "ENJ-USD", "CHZ-USD", "FLOW-USD", "THETA-USD", "XTZ-USD", "EGLD-USD", "NEAR-USD"]],
                "📈 Crypto ETFs": [k for k in popular_assets.keys() if k in ["BITO", "BITI", "GBTC", "ETHE", "GDLC", "BITQ", "BLOK", "LEGR"]],
                "💱 Currency ETFs": [k for k in popular_assets.keys() if k.startswith("FX") or k in ["UUP", "UDN"]],
                "💰 Dividend ETFs": [k for k in popular_assets.keys() if k in ["SCHD", "DVY", "VYM", "NOBL", "DGRO", "HDV", "SPHD", "PEY", "RDVY", "FDL"]],
                "🌱 ESG & Clean Energy": [k for k in popular_assets.keys() if k in ["ESG", "ESGU", "ESGD", "ESGE", "VSGX", "ESGV", "SUSA", "DSI", "ICLN", "PBW", "QCLN"]],
                "📊 Volatility & Leverage": [k for k in popular_assets.keys() if k in ["VIX", "UVXY", "SVXY", "VXX", "VIXY", "TQQQ", "SQQQ", "UPRO", "SPXU", "TNA", "TZA", "TECL", "TECS", "FAS", "FAZ"]]
            }
            
            selected_category = st.selectbox(
                "Choose Asset Category",
                [""] + list(asset_categories.keys()),
                help="Browse assets by category for easier selection"
            )
            
            if selected_category:
                available_assets = asset_categories[selected_category]
                selected_popular = st.selectbox(
                    f"Assets in {selected_category}",
                    [""] + available_assets,
                    format_func=lambda x: f"{x} - {popular_assets[x]}" if x else "",
                    help="Select from assets in this category"
                )
            else:
                selected_popular = st.selectbox(
                    "Or search all assets",
                    [""] + list(popular_assets.keys()),
                    format_func=lambda x: f"{x} - {popular_assets[x]}" if x else "",
                    help="Search through all available assets"
                )
            
            custom_symbol = st.text_input(
                "Or Enter Custom Symbol",
                placeholder="e.g., AAPL, BTC-USD, GLD.L (London)",
                help="Enter any valid symbol including international exchanges"
            )
            
            symbol_to_use = ""
            if selected_popular:
                symbol_to_use = selected_popular
            elif custom_symbol:
                symbol_to_use = custom_symbol.upper()
        
        with col2:
            shares = st.number_input(
                "Number of Shares/Units",
                min_value=0.001,
                value=1.0,
                step=0.1,
                help="Enter the quantity you own"
            )
            
            purchase_price = st.number_input(
                "Purchase Price per Share ($)",
                min_value=0.01,
                value=100.0,
                step=0.01,
                help="Enter the average price you paid per share"
            )
            
            # NEW: Currency selection
            currencies = get_currency_list()
            selected_currency = st.selectbox(
                "Currency",
                list(currencies.keys()),
                index=0,  # Default to USD
                format_func=lambda x: f"{x} - {currencies[x]['name']}",
                help="Select the currency for this asset"
            )
            
            asset_type = st.selectbox(
                "Asset Type",
                ["Stock", "ETF", "Cryptocurrency", "Bond", "Commodity", "Index Fund", "Mutual Fund", "REIT", "Other"]
            )
            
            if st.session_state.learning_mode:
                st.markdown("""
                <div class="info-card">
                    <strong>💡 Pro Tip:</strong><br>
                    • Select the correct currency for international assets<br>
                    • Purchase price should be in the selected currency<br>
                    • The system will convert to your base currency for analysis
                </div>
                """, unsafe_allow_html=True)
        
        if symbol_to_use:
            asset_info = fetch_asset_data(symbol_to_use)
            if asset_info:
                current_price = asset_info['current_price']
                
                # Convert prices for display if different currency
                if selected_currency != "USD":
                    exchange_rate = get_exchange_rate("USD", selected_currency)
                    current_price_display = current_price * exchange_rate
                    purchase_price_display = purchase_price
                else:
                    current_price_display = current_price
                    purchase_price_display = purchase_price
                
                potential_return = ((current_price_display - purchase_price_display) / purchase_price_display) * 100 if purchase_price_display > 0 else 0
                
                # Create three columns for metrics display
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                
                with metric_col1:
                    currency_symbol = currencies[selected_currency]["symbol"]
                    st.metric("Current Price", f"{currency_symbol}{current_price_display:.2f}")
                with metric_col2:
                    st.metric("Your Purchase Price", f"{currency_symbol}{purchase_price_display:.2f}")
                with metric_col3:
                    st.metric("Potential Return", f"{potential_return:+.1f}%", 
                             delta=f"{currency_symbol}{(current_price_display - purchase_price_display) * shares:+.2f}")
                
                if st.button("✅ Add to Portfolio", type="primary", use_container_width=True):
                    st.session_state.portfolio[symbol_to_use] = {
                        'shares': shares,
                        'asset_type': asset_type,
                        'purchase_price': purchase_price,
                        'currency': selected_currency,  # NEW: Store currency
                        'added_date': datetime.now().isoformat()
                    }
                    
                    portfolios = load_portfolios()
                    portfolios[st.session_state.username] = st.session_state.portfolio
                    save_portfolios(portfolios)
                    
                    st.markdown("""
                    <div class="success-card">
                        <strong>🎉 Asset Added Successfully!</strong><br>
                        Your portfolio metrics are being recalculated with the new asset allocation and currency conversion.
                    </div>
                    """, unsafe_allow_html=True)
                    st.rerun()
            else:
                st.error(f"❌ Could not find asset data for '{symbol_to_use}'. Please check the symbol.")
    
    with tab2:
        st.markdown("### Remove Assets from Portfolio")
        
        if st.session_state.portfolio:
            current_values = []
            for symbol, data in st.session_state.portfolio.items():
                asset_info = fetch_asset_data(symbol)
                if asset_info:
                    current_value = data['shares'] * asset_info['current_price']
                    current_values.append({
                        'Symbol': symbol,
                        'Shares': f"{data['shares']:.3f}",
                        'Current Value': f"${current_value:,.2f}",
                        'Asset Type': data['asset_type']
                    })
            
            if current_values:
                df_current = pd.DataFrame(current_values)
                st.dataframe(df_current, hide_index=True)
            
            assets_to_remove = st.multiselect(
                "Select assets to remove:",
                list(st.session_state.portfolio.keys()),
                help="Choose one or more assets to remove from your portfolio"
            )
            
            if assets_to_remove:
                st.warning(f"⚠️ You are about to remove {len(assets_to_remove)} asset(s) from your portfolio.")
                
                if st.button("🗑️ Remove Selected Assets", type="secondary"):
                    for asset in assets_to_remove:
                        del st.session_state.portfolio[asset]
                    
                    portfolios = load_portfolios()
                    portfolios[st.session_state.username] = st.session_state.portfolio
                    save_portfolios(portfolios)
                    
                    st.markdown("""
                    <div class="success-card">
                        <strong>✅ Assets Removed Successfully!</strong><br>
                        Your portfolio has been updated and metrics recalculated.
                    </div>
                    """, unsafe_allow_html=True)
                    st.rerun()
        else:
            st.markdown("""
            <div class="info-card">
                <strong>📝 No assets to remove</strong><br>
                Your portfolio is currently empty. Add some investments first!
            </div>
            """, unsafe_allow_html=True)

def show_analytics_dashboard():
    """Display analytics dashboard"""
    
    st.markdown("### 📊 Advanced Analytics Dashboard")
    
    if st.session_state.learning_mode:
        st.markdown("""
        <div class="info-card">
            <strong>📈 Analytics Dashboard Guide:</strong><br>
            This dashboard provides professional-grade analysis tools including performance metrics, technical analysis, and AI-powered insights.
        </div>
        """, unsafe_allow_html=True)
    
    if not st.session_state.portfolio:
        st.warning("⚠️ Add assets to your portfolio to see advanced analytics.")
        return
    
    metrics = calculate_portfolio_metrics_advanced(st.session_state.portfolio)
    
    st.markdown("### 📈 Advanced Performance Metrics")
    
    perf_col1, perf_col2, perf_col3, perf_col4, perf_col5 = st.columns(5)
    
    with perf_col1:
        st.metric(
            "Annualized Return", 
            f"{metrics['annualized_return']:+.1f}%",
            help="Expected annual return based on current performance"
        )
    
    with perf_col2:
        st.metric(
            "Portfolio Beta", 
            f"{metrics['beta']:.2f}",
            delta="Market Risk" if metrics['beta'] > 1 else "Lower Risk",
            help="Portfolio volatility relative to market"
        )
    
    with perf_col3:
        st.metric(
            "Sharpe Ratio", 
            f"{metrics['sharpe_ratio']:.2f}",
            delta="Excellent" if metrics['sharpe_ratio'] > 1 else "Good" if metrics['sharpe_ratio'] > 0.5 else "Needs Improvement",
            help="Risk-adjusted return measure"
        )
    
    with perf_col4:
        st.metric(
            "Max Drawdown", 
            f"{metrics['max_drawdown']:.1f}%",
            delta="Risk Level",
            delta_color="inverse",
            help="Largest peak-to-trough decline"
        )
    
    with perf_col5:
        st.metric(
            "Portfolio Volatility", 
            f"{metrics['volatility']:.1f}%",
            help="Measure of price fluctuation"
        )
    
    # Technical analysis for top holdings
    st.markdown("### 🔍 Technical Analysis - Top Holdings")
    
    portfolio_values = []
    for symbol, data in st.session_state.portfolio.items():
        asset_info = fetch_asset_data(symbol)
        if asset_info:
            value = data['shares'] * asset_info['current_price']
            portfolio_values.append((symbol, value))
    
    portfolio_values.sort(key=lambda x: x[1], reverse=True)
    top_holdings = [item[0] for item in portfolio_values[:3]]
    
    for i, symbol in enumerate(top_holdings):
        with st.expander(f"📊 {symbol} - Technical Analysis", expanded=(i == 0)):
            indicators = calculate_technical_indicators(symbol)
            
            if indicators is not None:
                tech_col1, tech_col2, tech_col3, tech_col4 = st.columns(4)
                
                current_price = indicators['Close'].iloc[-1]
                ma20 = indicators['MA_20'].iloc[-1]
                current_rsi = indicators['RSI'].iloc[-1]
                current_macd = indicators['MACD'].iloc[-1]
                current_signal = indicators['MACD_Signal'].iloc[-1]
                
                with tech_col1:
                    price_trend = "Bullish" if current_price > ma20 else "Bearish"
                    st.metric("Price Trend", price_trend, f"${current_price:.2f}")
                
                with tech_col2:
                    rsi_status = "Overbought" if current_rsi > 70 else "Oversold" if current_rsi < 30 else "Neutral"
                    rsi_color = "🔴" if current_rsi > 70 else "🟢" if current_rsi < 30 else "🟡"
                    st.metric("RSI Signal", f"{rsi_color} {rsi_status}", f"{current_rsi:.1f}")
                
                with tech_col3:
                    macd_trend = "Bullish" if current_macd > current_signal else "Bearish"
                    macd_color = "🟢" if current_macd > current_signal else "🔴"
                    st.metric("MACD Signal", f"{macd_color} {macd_trend}")
                
                with tech_col4:
                    support_level = current_price * 0.95
                    resistance_level = current_price * 1.05
                    st.metric("Support/Resistance", f"${support_level:.2f} / ${resistance_level:.2f}")
                
                if PLOTLY_AVAILABLE:
                    fig = make_subplots(
                        rows=3, cols=1,
                        subplot_titles=(
                            f'{symbol} - Price & Moving Averages',
                            'Relative Strength Index (RSI)',
                            'MACD & Signal Line'
                        ),
                        vertical_spacing=0.08,
                        row_heights=[0.5, 0.25, 0.25]
                    )
                    
                    fig.add_trace(
                        go.Scatter(x=indicators.index, y=indicators['Close'], name='Price', line=dict(color='blue', width=2)),
                        row=1, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=indicators.index, y=indicators['MA_20'], name='MA 20', line=dict(color='orange')),
                        row=1, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=indicators.index, y=indicators['MA_50'], name='MA 50', line=dict(color='red')),
                        row=1, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(x=indicators.index, y=indicators['RSI'], name='RSI', line=dict(color='purple')),
                        row=2, col=1
                    )
                    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
                    
                    fig.add_trace(
                        go.Scatter(x=indicators.index, y=indicators['MACD'], name='MACD', line=dict(color='blue')),
                        row=3, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=indicators.index, y=indicators['MACD_Signal'], name='Signal', line=dict(color='red')),
                        row=3, col=1
                    )
                    
                    fig.update_layout(height=700, showlegend=True, title_text=f"{symbol} Technical Analysis")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.markdown("#### 📈 Price Chart")
                    st.line_chart(indicators[['Close', 'MA_20', 'MA_50']])
                    
                    simple_col1, simple_col2 = st.columns(2)
                    with simple_col1:
                        st.markdown("#### RSI")
                        st.line_chart(indicators['RSI'])
                    with simple_col2:
                        st.markdown("#### MACD")
                        st.line_chart(indicators[['MACD', 'MACD_Signal']])
    
    # Investment suggestions
    st.markdown("### 🤖 AI-Powered Investment Insights")
    suggestions = generate_investment_suggestions(st.session_state.portfolio)
    
    enhanced_suggestions = []
    
    if metrics['total_return'] > 10:
        enhanced_suggestions.append({
            'type': 'success',
            'message': f'🎉 Excellent performance! Your portfolio is up {metrics["total_return"]:.1f}%. Consider taking some profits and rebalancing.'
        })
    elif metrics['total_return'] < -5:
        enhanced_suggestions.append({
            'type': 'warning',
            'message': f'📉 Portfolio is down {abs(metrics["total_return"]):.1f}%. Review underperforming assets and consider strategic rebalancing.'
        })
    
    if metrics['volatility'] > 25:
        enhanced_suggestions.append({
            'type': 'warning',
            'message': f'⚡ High volatility detected ({metrics["volatility"]:.1f}%). Consider adding stable assets like bonds or dividend stocks.'
        })
    
    all_suggestions = enhanced_suggestions + suggestions
    
    for suggestion in all_suggestions[:6]:
        if suggestion['type'] == 'diversification':
            st.markdown(f"""
            <div class="info-card">
                <strong>🎯 Diversification:</strong> {suggestion['message']}
            </div>
            """, unsafe_allow_html=True)
        elif suggestion['type'] == 'rebalancing':
            st.markdown(f"""
            <div class="warning-card">
                <strong>⚖️ Rebalancing:</strong> {suggestion['message']}
            </div>
            """, unsafe_allow_html=True)
        elif suggestion['type'] == 'opportunity':
            st.markdown(f"""
            <div class="success-card">
                <strong>🚀 Opportunity:</strong> {suggestion['message']}
            </div>
            """, unsafe_allow_html=True)
        elif suggestion['type'] == 'success':
            st.markdown(f"""
            <div class="success-card">
                <strong>✅ Success:</strong> {suggestion['message']}
            </div>
            """, unsafe_allow_html=True)
        elif suggestion['type'] == 'warning':
            st.markdown(f"""
            <div class="warning-card">
                <strong>⚠️ Warning:</strong> {suggestion['message']}
            </div>
            """, unsafe_allow_html=True)

def show_export_import():
    """Display export/import functionality"""
    
    st.markdown("### 📁 Export & Import Portfolio")
    
    if st.session_state.learning_mode:
        st.markdown("""
        <div class="info-card">
            <strong>📁 Export/Import Guide:</strong><br>
            Backup and restore your portfolio data. Export as JSON or CSV for safekeeping, or import previously saved portfolios.
        </div>
        """, unsafe_allow_html=True)
    
    export_tab, import_tab = st.tabs(["📤 Export Portfolio", "📥 Import Portfolio"])
    
    with export_tab:
        st.markdown("### Export Your Portfolio Data")
        
        if not st.session_state.portfolio:
            st.markdown("""
            <div class="warning-card">
                <strong>📝 No portfolio data to export</strong><br>
                Add some investments to your portfolio first, then return here to export your data.
            </div>
            """, unsafe_allow_html=True)
            return
        
        export_format = st.selectbox("Choose Export Format", ["JSON", "CSV"])
        
        if export_format == "JSON":
            export_data = {
                'username': st.session_state.username,
                'export_date': datetime.now().isoformat(),
                'portfolio': st.session_state.portfolio,
                'version': '2.0'
            }
            
            json_string = json.dumps(export_data, indent=2)
            
            st.download_button(
                label="📥 Download Portfolio (JSON)",
                data=json_string,
                file_name=f"portfolio_{st.session_state.username}_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )
            
            with st.expander("👀 Preview JSON Data"):
                st.code(json_string, language="json")
        
        else:  # CSV format
            csv_data = []
            for symbol, data in st.session_state.portfolio.items():
                asset_info = fetch_asset_data(symbol)
                if asset_info:
                    csv_data.append({
                        'Symbol': symbol,
                        'Name': asset_info['name'],
                        'Shares': data['shares'],
                        'Asset_Type': data['asset_type'],
                        'Purchase_Price': data.get('purchase_price', asset_info['current_price']),
                        'Current_Price': asset_info['current_price'],
                        'Current_Value': data['shares'] * asset_info['current_price'],
                        'Added_Date': data['added_date']
                    })
            
            if csv_data:
                df = pd.DataFrame(csv_data)
                csv_string = df.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download Portfolio (CSV)",
                    data=csv_string,
                    file_name=f"portfolio_{st.session_state.username}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
                
                with st.expander("👀 Preview CSV Data"):
                    st.dataframe(df)
    
    with import_tab:
        st.markdown("### Import Portfolio Data")
        
        uploaded_file = st.file_uploader(
            "Choose a portfolio file",
            type=['json', 'csv'],
            help="Upload a previously exported portfolio file"
        )
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.json'):
                    content = uploaded_file.read()
                    data = json.loads(content)
                    
                    if 'portfolio' in data:
                        st.markdown("""
                        <div class="success-card">
                            <strong>✅ Valid portfolio file detected!</strong><br>
                            Ready to import your portfolio data.
                        </div>
                        """, unsafe_allow_html=True)
                        
                        with st.expander("👀 Preview Import Data"):
                            st.json(data['portfolio'])
                        
                        import_option = st.selectbox(
                            "Import Option",
                            ["Replace Current Portfolio", "Merge with Current Portfolio"]
                        )
                        
                        if st.button("🚀 Import Portfolio", type="primary"):
                            if import_option == "Replace Current Portfolio":
                                st.session_state.portfolio = data['portfolio']
                            else:
                                for symbol, asset_data in data['portfolio'].items():
                                    st.session_state.portfolio[symbol] = asset_data
                            
                            portfolios = load_portfolios()
                            portfolios[st.session_state.username] = st.session_state.portfolio
                            save_portfolios(portfolios)
                            
                            st.markdown("""
                            <div class="success-card">
                                <strong>🎉 Portfolio imported successfully!</strong><br>
                                Your portfolio has been updated with the imported data.
                            </div>
                            """, unsafe_allow_html=True)
                            st.rerun()
                    else:
                        st.error("❌ Invalid portfolio file format.")
                
                elif uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                    
                    required_columns = ['Symbol', 'Shares', 'Asset_Type']
                    if all(col in df.columns for col in required_columns):
                        st.markdown("""
                        <div class="success-card">
                            <strong>✅ Valid CSV file detected!</strong><br>
                            Ready to import your portfolio data.
                        </div>
                        """, unsafe_allow_html=True)
                        
                        with st.expander("👀 Preview Import Data"):
                            st.dataframe(df)
                        
                        csv_import_option = st.selectbox(
                            "Import Option",
                            ["Replace Current Portfolio", "Merge with Current Portfolio"],
                            key="csv_import_option"
                        )
                        
                        if st.button("🚀 Import CSV Portfolio", type="primary"):
                            new_portfolio = {}
                            
                            for _, row in df.iterrows():
                                symbol = row['Symbol']
                                new_portfolio[symbol] = {
                                    'shares': float(row['Shares']),
                                    'asset_type': row['Asset_Type'],
                                    'purchase_price': float(row.get('Purchase_Price', 100.0)),
                                    'added_date': datetime.now().isoformat()
                                }
                            
                            if csv_import_option == "Replace Current Portfolio":
                                st.session_state.portfolio = new_portfolio
                            else:
                                for symbol, asset_data in new_portfolio.items():
                                    st.session_state.portfolio[symbol] = asset_data
                            
                            portfolios = load_portfolios()
                            portfolios[st.session_state.username] = st.session_state.portfolio
                            save_portfolios(portfolios)
                            
                            st.markdown("""
                            <div class="success-card">
                                <strong>🎉 CSV portfolio imported successfully!</strong><br>
                                Your portfolio has been updated with the imported data.
                            </div>
                            """, unsafe_allow_html=True)
                            st.rerun()
                    else:
                        st.error(f"❌ CSV file must contain columns: {', '.join(required_columns)}")
                        
            except Exception as e:
                st.error(f"❌ Error importing file: {str(e)}")

if __name__ == "__main__":
    main(), 'A"{return_pct:+.1f}%",
                'P&L': f"${current_value - invested_value:+,.2f}",
                'Asset Type': data['asset_type'],
                'Weight %': f"{(current_value / metrics['total_current_value']) * 100:.1f}%" if metrics['total_current_value'] > 0 else "0.0%"
            })
    
    if portfolio_data:
        st.markdown("### 📊 Holdings Breakdown")
        df = pd.DataFrame(portfolio_data)
        st.dataframe(df, use_container_width=True, height=400)
        
        if PLOTLY_AVAILABLE:
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                st.markdown("#### 🥧 Portfolio Allocation")
                # Extract numeric values safely
                current_values = []
                symbols = []
                
                for item in portfolio_data:
                    try:
                        value_str = item['Current Value'].replace('$', '').replace(',', '')
                        current_values.append(float(value_str))
                        symbols.append(item['Symbol'])
                    except:
                        continue
                
                if current_values:
                    fig_pie = px.pie(
                        values=current_values,
                        names=symbols,
                        title="Current Holdings Distribution",
                        hole=0.4
                    )
                    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                    fig_pie.update_layout(height=400)
                    st.plotly_chart(fig_pie, use_container_width=True)
            
            with col_chart2:
                st.markdown("#### 📈 Performance by Asset")
                return_values = []
                
                for item in portfolio_data:
                    try:
                        return_str = item['Return %'].replace('%', '').replace('+', '')
                        return_values.append(float(return_str))
                    except:
                        return_values.append(0.0)
                
                if return_values and symbols:
                    fig_returns = px.bar(
                        x=symbols,
                        y=return_values,
                        title="Return % by Holding",
                        color=return_values,
                        color_continuous_scale=['red', 'yellow', 'green']
                    )
                    fig_returns.update_layout(height=400, xaxis_title="Assets", yaxis_title="Return %")
                    st.plotly_chart(fig_returns, use_container_width=True)
        else:
            st.markdown("#### 📊 Simple Visualization")
            current_values = []
            symbols = []
            
            for item in portfolio_data:
                try:
                    value_str = item['Current Value'].replace('$', '').replace(',', '')
                    current_values.append(float(value_str))
                    symbols.append(item['Symbol'])
                except:
                    continue
            
            if current_values and symbols:
                chart_data = pd.DataFrame({
                    'Symbol': symbols,
                    'Current Value': current_values
                })
                st.bar_chart(chart_data.set_index('Symbol'))

def show_asset_management():
    """Display asset management page"""
    
    st.markdown("### 🎯 Manage Portfolio Assets")
    
    if st.session_state.learning_mode:
        st.markdown("""
        <div class="info-card">
            <strong>🎯 Asset Management Guide:</strong><br>
            Use this section to build and maintain your investment portfolio. Add assets, track purchase prices, and manage your holdings.
        </div>
        """, unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["➕ Add Assets", "➖ Remove Assets"])
    
    with tab1:
        st.markdown("### Add New Investment")
        
        col1, col2 = st.columns(2)
        
        with col1:
            popular_assets = get_popular_assets()
            
            # Create categories for better organization
            asset_categories = {
                "🏢 Large Cap Stocks": [k for k in popular_assets.keys() if k in ["AAPL", "GOOGL", "GOOG", "MSFT", "AMZN", "TSLA", "NVDA", "META", "NFLX", "JPM", "JNJ", "V", "WMT", "PG", "UNH", "HD", "MA", "BAC", "ADBE", "CRM", "XOM", "CVX", "KO", "PFE", "INTC", "CSCO", "VZ", "MRK", "ABT", "TMO"]],
                "🚀 Growth & Tech": [k for k in popular_assets.keys() if k in ["AMD", "PYPL", "SHOP", "SQ", "ROKU", "ZM", "SNOW", "PLTR", "CRWD", "NET"]],
                "🏛️ Vanguard ETFs": [k for k in popular_assets.keys() if k.startswith("V") and k in ["VTI", "VEA", "VWO", "VTV", "VUG", "VIG", "VYM", "VXUS", "VOO", "VT", "VB", "VO", "VGT", "VHT", "VFH", "VDE", "VAW", "VIS", "VCR", "VDC", "VPU", "VNQ"]],
                "⚫ BlackRock iShares": [k for k in popular_assets.keys() if k in ["QQQ", "IWM", "EFA", "EEM", "IVV", "IEFA", "IEMG", "IJH", "IJR", "IVW", "IVE", "IWF", "IWD", "ITOT", "IXUS", "IYY", "IWB", "IWV", "ACWI", "ACWX"]],
                "🎯 Sector ETFs": [k for k in popular_assets.keys() if k.startswith("XL") or k in ["IBB", "SOXX", "SKYY", "HACK", "ROBO", "ARKK", "ARKQ", "ARKW", "ARKG", "ARKF"]],
                "🏦 Bonds & Fixed Income": [k for k in popular_assets.keys() if k in ["AGG", "BND", "TLT", "IEF", "SHY", "LQD", "HYG", "JNK", "TIP", "VTEB", "MUB", "EMB", "BNDX", "VGIT", "VGSH", "VGLT", "VCIT", "VCSH", "BSV", "BIV", "BLV"]],
                "💎 Fidelity Funds": [k for k in popular_assets.keys() if k.startswith("F") and len(k) > 2],
                "🥇 Commodities": [k for k in popular_assets.keys() if k in ["GLD", "SLV", "USO", "UNG", "DBA", "DBC", "PDBC", "GSG", "COMT", "IAU", "SGOL", "PPLT", "PALL", "CORN", "WEAT", "SOYB", "CANE", "COW"]],
                "🌍 International": [k for k in popular_assets.keys() if k.startswith("EW") or k in ["FXI", "INDA", "RSX", "ASHR", "MCHI", "KWEB", "YINN"]],
                "₿ Major Cryptocurrencies": [k for k in popular_assets.keys() if k.endswith("-USD") and k in ["BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "ADA-USD", "SOL-USD", "DOGE-USD", "DOT-USD", "MATIC-USD", "SHIB-USD", "LTC-USD", "TRX-USD", "AVAX-USD", "LINK-USD", "ATOM-USD"]],
                "🔗 DeFi & Web3": [k for k in popular_assets.keys() if k.endswith("-USD") and k in ["XLM-USD", "ALGO-USD", "VET-USD", "ICP-USD", "FIL-USD", "SAND-USD", "MANA-USD", "AXS-USD", "ENJ-USD", "CHZ-USD", "FLOW-USD", "THETA-USD", "XTZ-USD", "EGLD-USD", "NEAR-USD"]],
                "📈 Crypto ETFs": [k for k in popular_assets.keys() if k in ["BITO", "BITI", "GBTC", "ETHE", "GDLC", "BITQ", "BLOK", "LEGR"]],
                "💱 Currency ETFs": [k for k in popular_assets.keys() if k.startswith("FX") or k in ["UUP", "UDN"]],
                "💰 Dividend ETFs": [k for k in popular_assets.keys() if k in ["SCHD", "DVY", "VYM", "NOBL", "DGRO", "HDV", "SPHD", "PEY", "RDVY", "FDL"]],
                "🌱 ESG & Clean Energy": [k for k in popular_assets.keys() if k in ["ESG", "ESGU", "ESGD", "ESGE", "VSGX", "ESGV", "SUSA", "DSI", "ICLN", "PBW", "QCLN"]],
                "📊 Volatility & Leverage": [k for k in popular_assets.keys() if k in ["VIX", "UVXY", "SVXY", "VXX", "VIXY", "TQQQ", "SQQQ", "UPRO", "SPXU", "TNA", "TZA", "TECL", "TECS", "FAS", "FAZ"]]
            }
            
            selected_category = st.selectbox(
                "Choose Asset Category",
                [""] + list(asset_categories.keys()),
                help="Browse assets by category for easier selection"
            )
            
            if selected_category:
                available_assets = asset_categories[selected_category]
                selected_popular = st.selectbox(
                    f"Assets in {selected_category}",
                    [""] + available_assets,
                    format_func=lambda x: f"{x} - {popular_assets[x]}" if x else "",
                    help="Select from assets in this category"
                )
            else:
                selected_popular = st.selectbox(
                    "Or search all assets",
                    [""] + list(popular_assets.keys()),
                    format_func=lambda x: f"{x} - {popular_assets[x]}" if x else "",
                    help="Search through all available assets"
                )
            
            custom_symbol = st.text_input(
                "Or Enter Custom Symbol",
                placeholder="e.g., AAPL, BTC-USD, GLD.L (London)",
                help="Enter any valid symbol including international exchanges"
            )
            
            symbol_to_use = ""
            if selected_popular:
                symbol_to_use = selected_popular
            elif custom_symbol:
                symbol_to_use = custom_symbol.upper()
        
        with col2:
            shares = st.number_input(
                "Number of Shares/Units",
                min_value=0.001,
                value=1.0,
                step=0.1,
                help="Enter the quantity you own"
            )
            
            purchase_price = st.number_input(
                "Purchase Price per Share ($)",
                min_value=0.01,
                value=100.0,
                step=0.01,
                help="Enter the average price you paid per share"
            )
            
            # NEW: Currency selection
            currencies = get_currency_list()
            selected_currency = st.selectbox(
                "Currency",
                list(currencies.keys()),
                index=0,  # Default to USD
                format_func=lambda x: f"{x} - {currencies[x]['name']}",
                help="Select the currency for this asset"
            )
            
            asset_type = st.selectbox(
                "Asset Type",
                ["Stock", "ETF", "Cryptocurrency", "Bond", "Commodity", "Index Fund", "Mutual Fund", "REIT", "Other"]
            )
            
            if st.session_state.learning_mode:
                st.markdown("""
                <div class="info-card">
                    <strong>💡 Pro Tip:</strong><br>
                    • Select the correct currency for international assets<br>
                    • Purchase price should be in the selected currency<br>
                    • The system will convert to your base currency for analysis
                </div>
                """, unsafe_allow_html=True)
        
        if symbol_to_use:
            asset_info = fetch_asset_data(symbol_to_use)
            if asset_info:
                current_price = asset_info['current_price']
                
                # Convert prices for display if different currency
                if selected_currency != "USD":
                    exchange_rate = get_exchange_rate("USD", selected_currency)
                    current_price_display = current_price * exchange_rate
                    purchase_price_display = purchase_price
                else:
                    current_price_display = current_price
                    purchase_price_display = purchase_price
                
                potential_return = ((current_price_display - purchase_price_display) / purchase_price_display) * 100 if purchase_price_display > 0 else 0
                
                # Create three columns for metrics display
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                
                with metric_col1:
                    currency_symbol = currencies[selected_currency]["symbol"]
                    st.metric("Current Price", f"{currency_symbol}{current_price_display:.2f}")
                with metric_col2:
                    st.metric("Your Purchase Price", f"{currency_symbol}{purchase_price_display:.2f}")
                with metric_col3:
                    st.metric("Potential Return", f"{potential_return:+.1f}%", 
                             delta=f"{currency_symbol}{(current_price_display - purchase_price_display) * shares:+.2f}")
                
                if st.button("✅ Add to Portfolio", type="primary", use_container_width=True):
                    st.session_state.portfolio[symbol_to_use] = {
                        'shares': shares,
                        'asset_type': asset_type,
                        'purchase_price': purchase_price,
                        'currency': selected_currency,  # NEW: Store currency
                        'added_date': datetime.now().isoformat()
                    }
                    
                    portfolios = load_portfolios()
                    portfolios[st.session_state.username] = st.session_state.portfolio
                    save_portfolios(portfolios)
                    
                    st.markdown("""
                    <div class="success-card">
                        <strong>🎉 Asset Added Successfully!</strong><br>
                        Your portfolio metrics are being recalculated with the new asset allocation and currency conversion.
                    </div>
                    """, unsafe_allow_html=True)
                    st.rerun()
            else:
                st.error(f"❌ Could not find asset data for '{symbol_to_use}'. Please check the symbol.")
    
    with tab2:
        st.markdown("### Remove Assets from Portfolio")
        
        if st.session_state.portfolio:
            current_values = []
            for symbol, data in st.session_state.portfolio.items():
                asset_info = fetch_asset_data(symbol)
                if asset_info:
                    current_value = data['shares'] * asset_info['current_price']
                    current_values.append({
                        'Symbol': symbol,
                        'Shares': f"{data['shares']:.3f}",
                        'Current Value': f"${current_value:,.2f}",
                        'Asset Type': data['asset_type']
                    })
            
            if current_values:
                df_current = pd.DataFrame(current_values)
                st.dataframe(df_current, hide_index=True)
            
            assets_to_remove = st.multiselect(
                "Select assets to remove:",
                list(st.session_state.portfolio.keys()),
                help="Choose one or more assets to remove from your portfolio"
            )
            
            if assets_to_remove:
                st.warning(f"⚠️ You are about to remove {len(assets_to_remove)} asset(s) from your portfolio.")
                
                if st.button("🗑️ Remove Selected Assets", type="secondary"):
                    for asset in assets_to_remove:
                        del st.session_state.portfolio[asset]
                    
                    portfolios = load_portfolios()
                    portfolios[st.session_state.username] = st.session_state.portfolio
                    save_portfolios(portfolios)
                    
                    st.markdown("""
                    <div class="success-card">
                        <strong>✅ Assets Removed Successfully!</strong><br>
                        Your portfolio has been updated and metrics recalculated.
                    </div>
                    """, unsafe_allow_html=True)
                    st.rerun()
        else:
            st.markdown("""
            <div class="info-card">
                <strong>📝 No assets to remove</strong><br>
                Your portfolio is currently empty. Add some investments first!
            </div>
            """, unsafe_allow_html=True)

def show_analytics_dashboard():
    """Display analytics dashboard"""
    
    st.markdown("### 📊 Advanced Analytics Dashboard")
    
    if st.session_state.learning_mode:
        st.markdown("""
        <div class="info-card">
            <strong>📈 Analytics Dashboard Guide:</strong><br>
            This dashboard provides professional-grade analysis tools including performance metrics, technical analysis, and AI-powered insights.
        </div>
        """, unsafe_allow_html=True)
    
    if not st.session_state.portfolio:
        st.warning("⚠️ Add assets to your portfolio to see advanced analytics.")
        return
    
    metrics = calculate_portfolio_metrics_advanced(st.session_state.portfolio)
    
    st.markdown("### 📈 Advanced Performance Metrics")
    
    perf_col1, perf_col2, perf_col3, perf_col4, perf_col5 = st.columns(5)
    
    with perf_col1:
        st.metric(
            "Annualized Return", 
            f"{metrics['annualized_return']:+.1f}%",
            help="Expected annual return based on current performance"
        )
    
    with perf_col2:
        st.metric(
            "Portfolio Beta", 
            f"{metrics['beta']:.2f}",
            delta="Market Risk" if metrics['beta'] > 1 else "Lower Risk",
            help="Portfolio volatility relative to market"
        )
    
    with perf_col3:
        st.metric(
            "Sharpe Ratio", 
            f"{metrics['sharpe_ratio']:.2f}",
            delta="Excellent" if metrics['sharpe_ratio'] > 1 else "Good" if metrics['sharpe_ratio'] > 0.5 else "Needs Improvement",
            help="Risk-adjusted return measure"
        )
    
    with perf_col4:
        st.metric(
            "Max Drawdown", 
            f"{metrics['max_drawdown']:.1f}%",
            delta="Risk Level",
            delta_color="inverse",
            help="Largest peak-to-trough decline"
        )
    
    with perf_col5:
        st.metric(
            "Portfolio Volatility", 
            f"{metrics['volatility']:.1f}%",
            help="Measure of price fluctuation"
        )
    
    # Technical analysis for top holdings
    st.markdown("### 🔍 Technical Analysis - Top Holdings")
    
    portfolio_values = []
    for symbol, data in st.session_state.portfolio.items():
        asset_info = fetch_asset_data(symbol)
        if asset_info:
            value = data['shares'] * asset_info['current_price']
            portfolio_values.append((symbol, value))
    
    portfolio_values.sort(key=lambda x: x[1], reverse=True)
    top_holdings = [item[0] for item in portfolio_values[:3]]
    
    for i, symbol in enumerate(top_holdings):
        with st.expander(f"📊 {symbol} - Technical Analysis", expanded=(i == 0)):
            indicators = calculate_technical_indicators(symbol)
            
            if indicators is not None:
                tech_col1, tech_col2, tech_col3, tech_col4 = st.columns(4)
                
                current_price = indicators['Close'].iloc[-1]
                ma20 = indicators['MA_20'].iloc[-1]
                current_rsi = indicators['RSI'].iloc[-1]
                current_macd = indicators['MACD'].iloc[-1]
                current_signal = indicators['MACD_Signal'].iloc[-1]
                
                with tech_col1:
                    price_trend = "Bullish" if current_price > ma20 else "Bearish"
                    st.metric("Price Trend", price_trend, f"${current_price:.2f}")
                
                with tech_col2:
                    rsi_status = "Overbought" if current_rsi > 70 else "Oversold" if current_rsi < 30 else "Neutral"
                    rsi_color = "🔴" if current_rsi > 70 else "🟢" if current_rsi < 30 else "🟡"
                    st.metric("RSI Signal", f"{rsi_color} {rsi_status}", f"{current_rsi:.1f}")
                
                with tech_col3:
                    macd_trend = "Bullish" if current_macd > current_signal else "Bearish"
                    macd_color = "🟢" if current_macd > current_signal else "🔴"
                    st.metric("MACD Signal", f"{macd_color} {macd_trend}")
                
                with tech_col4:
                    support_level = current_price * 0.95
                    resistance_level = current_price * 1.05
                    st.metric("Support/Resistance", f"${support_level:.2f} / ${resistance_level:.2f}")
                
                if PLOTLY_AVAILABLE:
                    fig = make_subplots(
                        rows=3, cols=1,
                        subplot_titles=(
                            f'{symbol} - Price & Moving Averages',
                            'Relative Strength Index (RSI)',
                            'MACD & Signal Line'
                        ),
                        vertical_spacing=0.08,
                        row_heights=[0.5, 0.25, 0.25]
                    )
                    
                    fig.add_trace(
                        go.Scatter(x=indicators.index, y=indicators['Close'], name='Price', line=dict(color='blue', width=2)),
                        row=1, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=indicators.index, y=indicators['MA_20'], name='MA 20', line=dict(color='orange')),
                        row=1, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=indicators.index, y=indicators['MA_50'], name='MA 50', line=dict(color='red')),
                        row=1, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(x=indicators.index, y=indicators['RSI'], name='RSI', line=dict(color='purple')),
                        row=2, col=1
                    )
                    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
                    
                    fig.add_trace(
                        go.Scatter(x=indicators.index, y=indicators['MACD'], name='MACD', line=dict(color='blue')),
                        row=3, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=indicators.index, y=indicators['MACD_Signal'], name='Signal', line=dict(color='red')),
                        row=3, col=1
                    )
                    
                    fig.update_layout(height=700, showlegend=True, title_text=f"{symbol} Technical Analysis")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.markdown("#### 📈 Price Chart")
                    st.line_chart(indicators[['Close', 'MA_20', 'MA_50']])
                    
                    simple_col1, simple_col2 = st.columns(2)
                    with simple_col1:
                        st.markdown("#### RSI")
                        st.line_chart(indicators['RSI'])
                    with simple_col2:
                        st.markdown("#### MACD")
                        st.line_chart(indicators[['MACD', 'MACD_Signal']])
    
    # Investment suggestions
    st.markdown("### 🤖 AI-Powered Investment Insights")
    suggestions = generate_investment_suggestions(st.session_state.portfolio)
    
    enhanced_suggestions = []
    
    if metrics['total_return'] > 10:
        enhanced_suggestions.append({
            'type': 'success',
            'message': f'🎉 Excellent performance! Your portfolio is up {metrics["total_return"]:.1f}%. Consider taking some profits and rebalancing.'
        })
    elif metrics['total_return'] < -5:
        enhanced_suggestions.append({
            'type': 'warning',
            'message': f'📉 Portfolio is down {abs(metrics["total_return"]):.1f}%. Review underperforming assets and consider strategic rebalancing.'
        })
    
    if metrics['volatility'] > 25:
        enhanced_suggestions.append({
            'type': 'warning',
            'message': f'⚡ High volatility detected ({metrics["volatility"]:.1f}%). Consider adding stable assets like bonds or dividend stocks.'
        })
    
    all_suggestions = enhanced_suggestions + suggestions
    
    for suggestion in all_suggestions[:6]:
        if suggestion['type'] == 'diversification':
            st.markdown(f"""
            <div class="info-card">
                <strong>🎯 Diversification:</strong> {suggestion['message']}
            </div>
            """, unsafe_allow_html=True)
        elif suggestion['type'] == 'rebalancing':
            st.markdown(f"""
            <div class="warning-card">
                <strong>⚖️ Rebalancing:</strong> {suggestion['message']}
            </div>
            """, unsafe_allow_html=True)
        elif suggestion['type'] == 'opportunity':
            st.markdown(f"""
            <div class="success-card">
                <strong>🚀 Opportunity:</strong> {suggestion['message']}
            </div>
            """, unsafe_allow_html=True)
        elif suggestion['type'] == 'success':
            st.markdown(f"""
            <div class="success-card">
                <strong>✅ Success:</strong> {suggestion['message']}
            </div>
            """, unsafe_allow_html=True)
        elif suggestion['type'] == 'warning':
            st.markdown(f"""
            <div class="warning-card">
                <strong>⚠️ Warning:</strong> {suggestion['message']}
            </div>
            """, unsafe_allow_html=True)

def show_export_import():
    """Display export/import functionality"""
    
    st.markdown("### 📁 Export & Import Portfolio")
    
    if st.session_state.learning_mode:
        st.markdown("""
        <div class="info-card">
            <strong>📁 Export/Import Guide:</strong><br>
            Backup and restore your portfolio data. Export as JSON or CSV for safekeeping, or import previously saved portfolios.
        </div>
        """, unsafe_allow_html=True)
    
    export_tab, import_tab = st.tabs(["📤 Export Portfolio", "📥 Import Portfolio"])
    
    with export_tab:
        st.markdown("### Export Your Portfolio Data")
        
        if not st.session_state.portfolio:
            st.markdown("""
            <div class="warning-card">
                <strong>📝 No portfolio data to export</strong><br>
                Add some investments to your portfolio first, then return here to export your data.
            </div>
            """, unsafe_allow_html=True)
            return
        
        export_format = st.selectbox("Choose Export Format", ["JSON", "CSV"])
        
        if export_format == "JSON":
            export_data = {
                'username': st.session_state.username,
                'export_date': datetime.now().isoformat(),
                'portfolio': st.session_state.portfolio,
                'version': '2.0'
            }
            
            json_string = json.dumps(export_data, indent=2)
            
            st.download_button(
                label="📥 Download Portfolio (JSON)",
                data=json_string,
                file_name=f"portfolio_{st.session_state.username}_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )
            
            with st.expander("👀 Preview JSON Data"):
                st.code(json_string, language="json")
        
        else:  # CSV format
            csv_data = []
            for symbol, data in st.session_state.portfolio.items():
                asset_info = fetch_asset_data(symbol)
                if asset_info:
                    csv_data.append({
                        'Symbol': symbol,
                        'Name': asset_info['name'],
                        'Shares': data['shares'],
                        'Asset_Type': data['asset_type'],
                        'Purchase_Price': data.get('purchase_price', asset_info['current_price']),
                        'Current_Price': asset_info['current_price'],
                        'Current_Value': data['shares'] * asset_info['current_price'],
                        'Added_Date': data['added_date']
                    })
            
            if csv_data:
                df = pd.DataFrame(csv_data)
                csv_string = df.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download Portfolio (CSV)",
                    data=csv_string,
                    file_name=f"portfolio_{st.session_state.username}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
                
                with st.expander("👀 Preview CSV Data"):
                    st.dataframe(df)
    
    with import_tab:
        st.markdown("### Import Portfolio Data")
        
        uploaded_file = st.file_uploader(
            "Choose a portfolio file",
            type=['json', 'csv'],
            help="Upload a previously exported portfolio file"
        )
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.json'):
                    content = uploaded_file.read()
                    data = json.loads(content)
                    
                    if 'portfolio' in data:
                        st.markdown("""
                        <div class="success-card">
                            <strong>✅ Valid portfolio file detected!</strong><br>
                            Ready to import your portfolio data.
                        </div>
                        """, unsafe_allow_html=True)
                        
                        with st.expander("👀 Preview Import Data"):
                            st.json(data['portfolio'])
                        
                        import_option = st.selectbox(
                            "Import Option",
                            ["Replace Current Portfolio", "Merge with Current Portfolio"]
                        )
                        
                        if st.button("🚀 Import Portfolio", type="primary"):
                            if import_option == "Replace Current Portfolio":
                                st.session_state.portfolio = data['portfolio']
                            else:
                                for symbol, asset_data in data['portfolio'].items():
                                    st.session_state.portfolio[symbol] = asset_data
                            
                            portfolios = load_portfolios()
                            portfolios[st.session_state.username] = st.session_state.portfolio
                            save_portfolios(portfolios)
                            
                            st.markdown("""
                            <div class="success-card">
                                <strong>🎉 Portfolio imported successfully!</strong><br>
                                Your portfolio has been updated with the imported data.
                            </div>
                            """, unsafe_allow_html=True)
                            st.rerun()
                    else:
                        st.error("❌ Invalid portfolio file format.")
                
                elif uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                    
                    required_columns = ['Symbol', 'Shares', 'Asset_Type']
                    if all(col in df.columns for col in required_columns):
                        st.markdown("""
                        <div class="success-card">
                            <strong>✅ Valid CSV file detected!</strong><br>
                            Ready to import your portfolio data.
                        </div>
                        """, unsafe_allow_html=True)
                        
                        with st.expander("👀 Preview Import Data"):
                            st.dataframe(df)
                        
                        csv_import_option = st.selectbox(
                            "Import Option",
                            ["Replace Current Portfolio", "Merge with Current Portfolio"],
                            key="csv_import_option"
                        )
                        
                        if st.button("🚀 Import CSV Portfolio", type="primary"):
                            new_portfolio = {}
                            
                            for _, row in df.iterrows():
                                symbol = row['Symbol']
                                new_portfolio[symbol] = {
                                    'shares': float(row['Shares']),
                                    'asset_type': row['Asset_Type'],
                                    'purchase_price': float(row.get('Purchase_Price', 100.0)),
                                    'added_date': datetime.now().isoformat()
                                }
                            
                            if csv_import_option == "Replace Current Portfolio":
                                st.session_state.portfolio = new_portfolio
                            else:
                                for symbol, asset_data in new_portfolio.items():
                                    st.session_state.portfolio[symbol] = asset_data
                            
                            portfolios = load_portfolios()
                            portfolios[st.session_state.username] = st.session_state.portfolio
                            save_portfolios(portfolios)
                            
                            st.markdown("""
                            <div class="success-card">
                                <strong>🎉 CSV portfolio imported successfully!</strong><br>
                                Your portfolio has been updated with the imported data.
                            </div>
                            """, unsafe_allow_html=True)
                            st.rerun()
                    else:
                        st.error(f"❌ CSV file must contain columns: {', '.join(required_columns)}")
                        
            except Exception as e:
                st.error(f"❌ Error importing file: {str(e)}")

if __name__ == "__main__":
    main(), 'CHF', 'R"{return_pct:+.1f}%",
                'P&L': f"${current_value - invested_value:+,.2f}",
                'Asset Type': data['asset_type'],
                'Weight %': f"{(current_value / metrics['total_current_value']) * 100:.1f}%" if metrics['total_current_value'] > 0 else "0.0%"
            })
    
    if portfolio_data:
        st.markdown("### 📊 Holdings Breakdown")
        df = pd.DataFrame(portfolio_data)
        st.dataframe(df, use_container_width=True, height=400)
        
        if PLOTLY_AVAILABLE:
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                st.markdown("#### 🥧 Portfolio Allocation")
                # Extract numeric values safely
                current_values = []
                symbols = []
                
                for item in portfolio_data:
                    try:
                        value_str = item['Current Value'].replace('$', '').replace(',', '')
                        current_values.append(float(value_str))
                        symbols.append(item['Symbol'])
                    except:
                        continue
                
                if current_values:
                    fig_pie = px.pie(
                        values=current_values,
                        names=symbols,
                        title="Current Holdings Distribution",
                        hole=0.4
                    )
                    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                    fig_pie.update_layout(height=400)
                    st.plotly_chart(fig_pie, use_container_width=True)
            
            with col_chart2:
                st.markdown("#### 📈 Performance by Asset")
                return_values = []
                
                for item in portfolio_data:
                    try:
                        return_str = item['Return %'].replace('%', '').replace('+', '')
                        return_values.append(float(return_str))
                    except:
                        return_values.append(0.0)
                
                if return_values and symbols:
                    fig_returns = px.bar(
                        x=symbols,
                        y=return_values,
                        title="Return % by Holding",
                        color=return_values,
                        color_continuous_scale=['red', 'yellow', 'green']
                    )
                    fig_returns.update_layout(height=400, xaxis_title="Assets", yaxis_title="Return %")
                    st.plotly_chart(fig_returns, use_container_width=True)
        else:
            st.markdown("#### 📊 Simple Visualization")
            current_values = []
            symbols = []
            
            for item in portfolio_data:
                try:
                    value_str = item['Current Value'].replace('$', '').replace(',', '')
                    current_values.append(float(value_str))
                    symbols.append(item['Symbol'])
                except:
                    continue
            
            if current_values and symbols:
                chart_data = pd.DataFrame({
                    'Symbol': symbols,
                    'Current Value': current_values
                })
                st.bar_chart(chart_data.set_index('Symbol'))

def show_asset_management():
    """Display asset management page"""
    
    st.markdown("### 🎯 Manage Portfolio Assets")
    
    if st.session_state.learning_mode:
        st.markdown("""
        <div class="info-card">
            <strong>🎯 Asset Management Guide:</strong><br>
            Use this section to build and maintain your investment portfolio. Add assets, track purchase prices, and manage your holdings.
        </div>
        """, unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["➕ Add Assets", "➖ Remove Assets"])
    
    with tab1:
        st.markdown("### Add New Investment")
        
        col1, col2 = st.columns(2)
        
        with col1:
            popular_assets = get_popular_assets()
            
            # Create categories for better organization
            asset_categories = {
                "🏢 Large Cap Stocks": [k for k in popular_assets.keys() if k in ["AAPL", "GOOGL", "GOOG", "MSFT", "AMZN", "TSLA", "NVDA", "META", "NFLX", "JPM", "JNJ", "V", "WMT", "PG", "UNH", "HD", "MA", "BAC", "ADBE", "CRM", "XOM", "CVX", "KO", "PFE", "INTC", "CSCO", "VZ", "MRK", "ABT", "TMO"]],
                "🚀 Growth & Tech": [k for k in popular_assets.keys() if k in ["AMD", "PYPL", "SHOP", "SQ", "ROKU", "ZM", "SNOW", "PLTR", "CRWD", "NET"]],
                "🏛️ Vanguard ETFs": [k for k in popular_assets.keys() if k.startswith("V") and k in ["VTI", "VEA", "VWO", "VTV", "VUG", "VIG", "VYM", "VXUS", "VOO", "VT", "VB", "VO", "VGT", "VHT", "VFH", "VDE", "VAW", "VIS", "VCR", "VDC", "VPU", "VNQ"]],
                "⚫ BlackRock iShares": [k for k in popular_assets.keys() if k in ["QQQ", "IWM", "EFA", "EEM", "IVV", "IEFA", "IEMG", "IJH", "IJR", "IVW", "IVE", "IWF", "IWD", "ITOT", "IXUS", "IYY", "IWB", "IWV", "ACWI", "ACWX"]],
                "🎯 Sector ETFs": [k for k in popular_assets.keys() if k.startswith("XL") or k in ["IBB", "SOXX", "SKYY", "HACK", "ROBO", "ARKK", "ARKQ", "ARKW", "ARKG", "ARKF"]],
                "🏦 Bonds & Fixed Income": [k for k in popular_assets.keys() if k in ["AGG", "BND", "TLT", "IEF", "SHY", "LQD", "HYG", "JNK", "TIP", "VTEB", "MUB", "EMB", "BNDX", "VGIT", "VGSH", "VGLT", "VCIT", "VCSH", "BSV", "BIV", "BLV"]],
                "💎 Fidelity Funds": [k for k in popular_assets.keys() if k.startswith("F") and len(k) > 2],
                "🥇 Commodities": [k for k in popular_assets.keys() if k in ["GLD", "SLV", "USO", "UNG", "DBA", "DBC", "PDBC", "GSG", "COMT", "IAU", "SGOL", "PPLT", "PALL", "CORN", "WEAT", "SOYB", "CANE", "COW"]],
                "🌍 International": [k for k in popular_assets.keys() if k.startswith("EW") or k in ["FXI", "INDA", "RSX", "ASHR", "MCHI", "KWEB", "YINN"]],
                "₿ Major Cryptocurrencies": [k for k in popular_assets.keys() if k.endswith("-USD") and k in ["BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "ADA-USD", "SOL-USD", "DOGE-USD", "DOT-USD", "MATIC-USD", "SHIB-USD", "LTC-USD", "TRX-USD", "AVAX-USD", "LINK-USD", "ATOM-USD"]],
                "🔗 DeFi & Web3": [k for k in popular_assets.keys() if k.endswith("-USD") and k in ["XLM-USD", "ALGO-USD", "VET-USD", "ICP-USD", "FIL-USD", "SAND-USD", "MANA-USD", "AXS-USD", "ENJ-USD", "CHZ-USD", "FLOW-USD", "THETA-USD", "XTZ-USD", "EGLD-USD", "NEAR-USD"]],
                "📈 Crypto ETFs": [k for k in popular_assets.keys() if k in ["BITO", "BITI", "GBTC", "ETHE", "GDLC", "BITQ", "BLOK", "LEGR"]],
                "💱 Currency ETFs": [k for k in popular_assets.keys() if k.startswith("FX") or k in ["UUP", "UDN"]],
                "💰 Dividend ETFs": [k for k in popular_assets.keys() if k in ["SCHD", "DVY", "VYM", "NOBL", "DGRO", "HDV", "SPHD", "PEY", "RDVY", "FDL"]],
                "🌱 ESG & Clean Energy": [k for k in popular_assets.keys() if k in ["ESG", "ESGU", "ESGD", "ESGE", "VSGX", "ESGV", "SUSA", "DSI", "ICLN", "PBW", "QCLN"]],
                "📊 Volatility & Leverage": [k for k in popular_assets.keys() if k in ["VIX", "UVXY", "SVXY", "VXX", "VIXY", "TQQQ", "SQQQ", "UPRO", "SPXU", "TNA", "TZA", "TECL", "TECS", "FAS", "FAZ"]]
            }
            
            selected_category = st.selectbox(
                "Choose Asset Category",
                [""] + list(asset_categories.keys()),
                help="Browse assets by category for easier selection"
            )
            
            if selected_category:
                available_assets = asset_categories[selected_category]
                selected_popular = st.selectbox(
                    f"Assets in {selected_category}",
                    [""] + available_assets,
                    format_func=lambda x: f"{x} - {popular_assets[x]}" if x else "",
                    help="Select from assets in this category"
                )
            else:
                selected_popular = st.selectbox(
                    "Or search all assets",
                    [""] + list(popular_assets.keys()),
                    format_func=lambda x: f"{x} - {popular_assets[x]}" if x else "",
                    help="Search through all available assets"
                )
            
            custom_symbol = st.text_input(
                "Or Enter Custom Symbol",
                placeholder="e.g., AAPL, BTC-USD, GLD.L (London)",
                help="Enter any valid symbol including international exchanges"
            )
            
            symbol_to_use = ""
            if selected_popular:
                symbol_to_use = selected_popular
            elif custom_symbol:
                symbol_to_use = custom_symbol.upper()
        
        with col2:
            shares = st.number_input(
                "Number of Shares/Units",
                min_value=0.001,
                value=1.0,
                step=0.1,
                help="Enter the quantity you own"
            )
            
            purchase_price = st.number_input(
                "Purchase Price per Share ($)",
                min_value=0.01,
                value=100.0,
                step=0.01,
                help="Enter the average price you paid per share"
            )
            
            # NEW: Currency selection
            currencies = get_currency_list()
            selected_currency = st.selectbox(
                "Currency",
                list(currencies.keys()),
                index=0,  # Default to USD
                format_func=lambda x: f"{x} - {currencies[x]['name']}",
                help="Select the currency for this asset"
            )
            
            asset_type = st.selectbox(
                "Asset Type",
                ["Stock", "ETF", "Cryptocurrency", "Bond", "Commodity", "Index Fund", "Mutual Fund", "REIT", "Other"]
            )
            
            if st.session_state.learning_mode:
                st.markdown("""
                <div class="info-card">
                    <strong>💡 Pro Tip:</strong><br>
                    • Select the correct currency for international assets<br>
                    • Purchase price should be in the selected currency<br>
                    • The system will convert to your base currency for analysis
                </div>
                """, unsafe_allow_html=True)
        
        if symbol_to_use:
            asset_info = fetch_asset_data(symbol_to_use)
            if asset_info:
                current_price = asset_info['current_price']
                
                # Convert prices for display if different currency
                if selected_currency != "USD":
                    exchange_rate = get_exchange_rate("USD", selected_currency)
                    current_price_display = current_price * exchange_rate
                    purchase_price_display = purchase_price
                else:
                    current_price_display = current_price
                    purchase_price_display = purchase_price
                
                potential_return = ((current_price_display - purchase_price_display) / purchase_price_display) * 100 if purchase_price_display > 0 else 0
                
                # Create three columns for metrics display
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                
                with metric_col1:
                    currency_symbol = currencies[selected_currency]["symbol"]
                    st.metric("Current Price", f"{currency_symbol}{current_price_display:.2f}")
                with metric_col2:
                    st.metric("Your Purchase Price", f"{currency_symbol}{purchase_price_display:.2f}")
                with metric_col3:
                    st.metric("Potential Return", f"{potential_return:+.1f}%", 
                             delta=f"{currency_symbol}{(current_price_display - purchase_price_display) * shares:+.2f}")
                
                if st.button("✅ Add to Portfolio", type="primary", use_container_width=True):
                    st.session_state.portfolio[symbol_to_use] = {
                        'shares': shares,
                        'asset_type': asset_type,
                        'purchase_price': purchase_price,
                        'currency': selected_currency,  # NEW: Store currency
                        'added_date': datetime.now().isoformat()
                    }
                    
                    portfolios = load_portfolios()
                    portfolios[st.session_state.username] = st.session_state.portfolio
                    save_portfolios(portfolios)
                    
                    st.markdown("""
                    <div class="success-card">
                        <strong>🎉 Asset Added Successfully!</strong><br>
                        Your portfolio metrics are being recalculated with the new asset allocation and currency conversion.
                    </div>
                    """, unsafe_allow_html=True)
                    st.rerun()
            else:
                st.error(f"❌ Could not find asset data for '{symbol_to_use}'. Please check the symbol.")
    
    with tab2:
        st.markdown("### Remove Assets from Portfolio")
        
        if st.session_state.portfolio:
            current_values = []
            for symbol, data in st.session_state.portfolio.items():
                asset_info = fetch_asset_data(symbol)
                if asset_info:
                    current_value = data['shares'] * asset_info['current_price']
                    current_values.append({
                        'Symbol': symbol,
                        'Shares': f"{data['shares']:.3f}",
                        'Current Value': f"${current_value:,.2f}",
                        'Asset Type': data['asset_type']
                    })
            
            if current_values:
                df_current = pd.DataFrame(current_values)
                st.dataframe(df_current, hide_index=True)
            
            assets_to_remove = st.multiselect(
                "Select assets to remove:",
                list(st.session_state.portfolio.keys()),
                help="Choose one or more assets to remove from your portfolio"
            )
            
            if assets_to_remove:
                st.warning(f"⚠️ You are about to remove {len(assets_to_remove)} asset(s) from your portfolio.")
                
                if st.button("🗑️ Remove Selected Assets", type="secondary"):
                    for asset in assets_to_remove:
                        del st.session_state.portfolio[asset]
                    
                    portfolios = load_portfolios()
                    portfolios[st.session_state.username] = st.session_state.portfolio
                    save_portfolios(portfolios)
                    
                    st.markdown("""
                    <div class="success-card">
                        <strong>✅ Assets Removed Successfully!</strong><br>
                        Your portfolio has been updated and metrics recalculated.
                    </div>
                    """, unsafe_allow_html=True)
                    st.rerun()
        else:
            st.markdown("""
            <div class="info-card">
                <strong>📝 No assets to remove</strong><br>
                Your portfolio is currently empty. Add some investments first!
            </div>
            """, unsafe_allow_html=True)

def show_analytics_dashboard():
    """Display analytics dashboard"""
    
    st.markdown("### 📊 Advanced Analytics Dashboard")
    
    if st.session_state.learning_mode:
        st.markdown("""
        <div class="info-card">
            <strong>📈 Analytics Dashboard Guide:</strong><br>
            This dashboard provides professional-grade analysis tools including performance metrics, technical analysis, and AI-powered insights.
        </div>
        """, unsafe_allow_html=True)
    
    if not st.session_state.portfolio:
        st.warning("⚠️ Add assets to your portfolio to see advanced analytics.")
        return
    
    metrics = calculate_portfolio_metrics_advanced(st.session_state.portfolio)
    
    st.markdown("### 📈 Advanced Performance Metrics")
    
    perf_col1, perf_col2, perf_col3, perf_col4, perf_col5 = st.columns(5)
    
    with perf_col1:
        st.metric(
            "Annualized Return", 
            f"{metrics['annualized_return']:+.1f}%",
            help="Expected annual return based on current performance"
        )
    
    with perf_col2:
        st.metric(
            "Portfolio Beta", 
            f"{metrics['beta']:.2f}",
            delta="Market Risk" if metrics['beta'] > 1 else "Lower Risk",
            help="Portfolio volatility relative to market"
        )
    
    with perf_col3:
        st.metric(
            "Sharpe Ratio", 
            f"{metrics['sharpe_ratio']:.2f}",
            delta="Excellent" if metrics['sharpe_ratio'] > 1 else "Good" if metrics['sharpe_ratio'] > 0.5 else "Needs Improvement",
            help="Risk-adjusted return measure"
        )
    
    with perf_col4:
        st.metric(
            "Max Drawdown", 
            f"{metrics['max_drawdown']:.1f}%",
            delta="Risk Level",
            delta_color="inverse",
            help="Largest peak-to-trough decline"
        )
    
    with perf_col5:
        st.metric(
            "Portfolio Volatility", 
            f"{metrics['volatility']:.1f}%",
            help="Measure of price fluctuation"
        )
    
    # Technical analysis for top holdings
    st.markdown("### 🔍 Technical Analysis - Top Holdings")
    
    portfolio_values = []
    for symbol, data in st.session_state.portfolio.items():
        asset_info = fetch_asset_data(symbol)
        if asset_info:
            value = data['shares'] * asset_info['current_price']
            portfolio_values.append((symbol, value))
    
    portfolio_values.sort(key=lambda x: x[1], reverse=True)
    top_holdings = [item[0] for item in portfolio_values[:3]]
    
    for i, symbol in enumerate(top_holdings):
        with st.expander(f"📊 {symbol} - Technical Analysis", expanded=(i == 0)):
            indicators = calculate_technical_indicators(symbol)
            
            if indicators is not None:
                tech_col1, tech_col2, tech_col3, tech_col4 = st.columns(4)
                
                current_price = indicators['Close'].iloc[-1]
                ma20 = indicators['MA_20'].iloc[-1]
                current_rsi = indicators['RSI'].iloc[-1]
                current_macd = indicators['MACD'].iloc[-1]
                current_signal = indicators['MACD_Signal'].iloc[-1]
                
                with tech_col1:
                    price_trend = "Bullish" if current_price > ma20 else "Bearish"
                    st.metric("Price Trend", price_trend, f"${current_price:.2f}")
                
                with tech_col2:
                    rsi_status = "Overbought" if current_rsi > 70 else "Oversold" if current_rsi < 30 else "Neutral"
                    rsi_color = "🔴" if current_rsi > 70 else "🟢" if current_rsi < 30 else "🟡"
                    st.metric("RSI Signal", f"{rsi_color} {rsi_status}", f"{current_rsi:.1f}")
                
                with tech_col3:
                    macd_trend = "Bullish" if current_macd > current_signal else "Bearish"
                    macd_color = "🟢" if current_macd > current_signal else "🔴"
                    st.metric("MACD Signal", f"{macd_color} {macd_trend}")
                
                with tech_col4:
                    support_level = current_price * 0.95
                    resistance_level = current_price * 1.05
                    st.metric("Support/Resistance", f"${support_level:.2f} / ${resistance_level:.2f}")
                
                if PLOTLY_AVAILABLE:
                    fig = make_subplots(
                        rows=3, cols=1,
                        subplot_titles=(
                            f'{symbol} - Price & Moving Averages',
                            'Relative Strength Index (RSI)',
                            'MACD & Signal Line'
                        ),
                        vertical_spacing=0.08,
                        row_heights=[0.5, 0.25, 0.25]
                    )
                    
                    fig.add_trace(
                        go.Scatter(x=indicators.index, y=indicators['Close'], name='Price', line=dict(color='blue', width=2)),
                        row=1, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=indicators.index, y=indicators['MA_20'], name='MA 20', line=dict(color='orange')),
                        row=1, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=indicators.index, y=indicators['MA_50'], name='MA 50', line=dict(color='red')),
                        row=1, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(x=indicators.index, y=indicators['RSI'], name='RSI', line=dict(color='purple')),
                        row=2, col=1
                    )
                    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
                    
                    fig.add_trace(
                        go.Scatter(x=indicators.index, y=indicators['MACD'], name='MACD', line=dict(color='blue')),
                        row=3, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=indicators.index, y=indicators['MACD_Signal'], name='Signal', line=dict(color='red')),
                        row=3, col=1
                    )
                    
                    fig.update_layout(height=700, showlegend=True, title_text=f"{symbol} Technical Analysis")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.markdown("#### 📈 Price Chart")
                    st.line_chart(indicators[['Close', 'MA_20', 'MA_50']])
                    
                    simple_col1, simple_col2 = st.columns(2)
                    with simple_col1:
                        st.markdown("#### RSI")
                        st.line_chart(indicators['RSI'])
                    with simple_col2:
                        st.markdown("#### MACD")
                        st.line_chart(indicators[['MACD', 'MACD_Signal']])
    
    # Investment suggestions
    st.markdown("### 🤖 AI-Powered Investment Insights")
    suggestions = generate_investment_suggestions(st.session_state.portfolio)
    
    enhanced_suggestions = []
    
    if metrics['total_return'] > 10:
        enhanced_suggestions.append({
            'type': 'success',
            'message': f'🎉 Excellent performance! Your portfolio is up {metrics["total_return"]:.1f}%. Consider taking some profits and rebalancing.'
        })
    elif metrics['total_return'] < -5:
        enhanced_suggestions.append({
            'type': 'warning',
            'message': f'📉 Portfolio is down {abs(metrics["total_return"]):.1f}%. Review underperforming assets and consider strategic rebalancing.'
        })
    
    if metrics['volatility'] > 25:
        enhanced_suggestions.append({
            'type': 'warning',
            'message': f'⚡ High volatility detected ({metrics["volatility"]:.1f}%). Consider adding stable assets like bonds or dividend stocks.'
        })
    
    all_suggestions = enhanced_suggestions + suggestions
    
    for suggestion in all_suggestions[:6]:
        if suggestion['type'] == 'diversification':
            st.markdown(f"""
            <div class="info-card">
                <strong>🎯 Diversification:</strong> {suggestion['message']}
            </div>
            """, unsafe_allow_html=True)
        elif suggestion['type'] == 'rebalancing':
            st.markdown(f"""
            <div class="warning-card">
                <strong>⚖️ Rebalancing:</strong> {suggestion['message']}
            </div>
            """, unsafe_allow_html=True)
        elif suggestion['type'] == 'opportunity':
            st.markdown(f"""
            <div class="success-card">
                <strong>🚀 Opportunity:</strong> {suggestion['message']}
            </div>
            """, unsafe_allow_html=True)
        elif suggestion['type'] == 'success':
            st.markdown(f"""
            <div class="success-card">
                <strong>✅ Success:</strong> {suggestion['message']}
            </div>
            """, unsafe_allow_html=True)
        elif suggestion['type'] == 'warning':
            st.markdown(f"""
            <div class="warning-card">
                <strong>⚠️ Warning:</strong> {suggestion['message']}
            </div>
            """, unsafe_allow_html=True)

def show_export_import():
    """Display export/import functionality"""
    
    st.markdown("### 📁 Export & Import Portfolio")
    
    if st.session_state.learning_mode:
        st.markdown("""
        <div class="info-card">
            <strong>📁 Export/Import Guide:</strong><br>
            Backup and restore your portfolio data. Export as JSON or CSV for safekeeping, or import previously saved portfolios.
        </div>
        """, unsafe_allow_html=True)
    
    export_tab, import_tab = st.tabs(["📤 Export Portfolio", "📥 Import Portfolio"])
    
    with export_tab:
        st.markdown("### Export Your Portfolio Data")
        
        if not st.session_state.portfolio:
            st.markdown("""
            <div class="warning-card">
                <strong>📝 No portfolio data to export</strong><br>
                Add some investments to your portfolio first, then return here to export your data.
            </div>
            """, unsafe_allow_html=True)
            return
        
        export_format = st.selectbox("Choose Export Format", ["JSON", "CSV"])
        
        if export_format == "JSON":
            export_data = {
                'username': st.session_state.username,
                'export_date': datetime.now().isoformat(),
                'portfolio': st.session_state.portfolio,
                'version': '2.0'
            }
            
            json_string = json.dumps(export_data, indent=2)
            
            st.download_button(
                label="📥 Download Portfolio (JSON)",
                data=json_string,
                file_name=f"portfolio_{st.session_state.username}_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )
            
            with st.expander("👀 Preview JSON Data"):
                st.code(json_string, language="json")
        
        else:  # CSV format
            csv_data = []
            for symbol, data in st.session_state.portfolio.items():
                asset_info = fetch_asset_data(symbol)
                if asset_info:
                    csv_data.append({
                        'Symbol': symbol,
                        'Name': asset_info['name'],
                        'Shares': data['shares'],
                        'Asset_Type': data['asset_type'],
                        'Purchase_Price': data.get('purchase_price', asset_info['current_price']),
                        'Current_Price': asset_info['current_price'],
                        'Current_Value': data['shares'] * asset_info['current_price'],
                        'Added_Date': data['added_date']
                    })
            
            if csv_data:
                df = pd.DataFrame(csv_data)
                csv_string = df.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download Portfolio (CSV)",
                    data=csv_string,
                    file_name=f"portfolio_{st.session_state.username}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
                
                with st.expander("👀 Preview CSV Data"):
                    st.dataframe(df)
    
    with import_tab:
        st.markdown("### Import Portfolio Data")
        
        uploaded_file = st.file_uploader(
            "Choose a portfolio file",
            type=['json', 'csv'],
            help="Upload a previously exported portfolio file"
        )
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.json'):
                    content = uploaded_file.read()
                    data = json.loads(content)
                    
                    if 'portfolio' in data:
                        st.markdown("""
                        <div class="success-card">
                            <strong>✅ Valid portfolio file detected!</strong><br>
                            Ready to import your portfolio data.
                        </div>
                        """, unsafe_allow_html=True)
                        
                        with st.expander("👀 Preview Import Data"):
                            st.json(data['portfolio'])
                        
                        import_option = st.selectbox(
                            "Import Option",
                            ["Replace Current Portfolio", "Merge with Current Portfolio"]
                        )
                        
                        if st.button("🚀 Import Portfolio", type="primary"):
                            if import_option == "Replace Current Portfolio":
                                st.session_state.portfolio = data['portfolio']
                            else:
                                for symbol, asset_data in data['portfolio'].items():
                                    st.session_state.portfolio[symbol] = asset_data
                            
                            portfolios = load_portfolios()
                            portfolios[st.session_state.username] = st.session_state.portfolio
                            save_portfolios(portfolios)
                            
                            st.markdown("""
                            <div class="success-card">
                                <strong>🎉 Portfolio imported successfully!</strong><br>
                                Your portfolio has been updated with the imported data.
                            </div>
                            """, unsafe_allow_html=True)
                            st.rerun()
                    else:
                        st.error("❌ Invalid portfolio file format.")
                
                elif uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                    
                    required_columns = ['Symbol', 'Shares', 'Asset_Type']
                    if all(col in df.columns for col in required_columns):
                        st.markdown("""
                        <div class="success-card">
                            <strong>✅ Valid CSV file detected!</strong><br>
                            Ready to import your portfolio data.
                        </div>
                        """, unsafe_allow_html=True)
                        
                        with st.expander("👀 Preview Import Data"):
                            st.dataframe(df)
                        
                        csv_import_option = st.selectbox(
                            "Import Option",
                            ["Replace Current Portfolio", "Merge with Current Portfolio"],
                            key="csv_import_option"
                        )
                        
                        if st.button("🚀 Import CSV Portfolio", type="primary"):
                            new_portfolio = {}
                            
                            for _, row in df.iterrows():
                                symbol = row['Symbol']
                                new_portfolio[symbol] = {
                                    'shares': float(row['Shares']),
                                    'asset_type': row['Asset_Type'],
                                    'purchase_price': float(row.get('Purchase_Price', 100.0)),
                                    'added_date': datetime.now().isoformat()
                                }
                            
                            if csv_import_option == "Replace Current Portfolio":
                                st.session_state.portfolio = new_portfolio
                            else:
                                for symbol, asset_data in new_portfolio.items():
                                    st.session_state.portfolio[symbol] = asset_data
                            
                            portfolios = load_portfolios()
                            portfolios[st.session_state.username] = st.session_state.portfolio
                            save_portfolios(portfolios)
                            
                            st.markdown("""
                            <div class="success-card">
                                <strong>🎉 CSV portfolio imported successfully!</strong><br>
                                Your portfolio has been updated with the imported data.
                            </div>
                            """, unsafe_allow_html=True)
                            st.rerun()
                    else:
                        st.error(f"❌ CSV file must contain columns: {', '.join(required_columns)}")
                        
            except Exception as e:
                st.error(f"❌ Error importing file: {str(e)}")

if __name__ == "__main__":
    main(), 'S"{return_pct:+.1f}%",
                'P&L': f"${current_value - invested_value:+,.2f}",
                'Asset Type': data['asset_type'],
                'Weight %': f"{(current_value / metrics['total_current_value']) * 100:.1f}%" if metrics['total_current_value'] > 0 else "0.0%"
            })
    
    if portfolio_data:
        st.markdown("### 📊 Holdings Breakdown")
        df = pd.DataFrame(portfolio_data)
        st.dataframe(df, use_container_width=True, height=400)
        
        if PLOTLY_AVAILABLE:
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                st.markdown("#### 🥧 Portfolio Allocation")
                # Extract numeric values safely
                current_values = []
                symbols = []
                
                for item in portfolio_data:
                    try:
                        value_str = item['Current Value'].replace('$', '').replace(',', '')
                        current_values.append(float(value_str))
                        symbols.append(item['Symbol'])
                    except:
                        continue
                
                if current_values:
                    fig_pie = px.pie(
                        values=current_values,
                        names=symbols,
                        title="Current Holdings Distribution",
                        hole=0.4
                    )
                    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                    fig_pie.update_layout(height=400)
                    st.plotly_chart(fig_pie, use_container_width=True)
            
            with col_chart2:
                st.markdown("#### 📈 Performance by Asset")
                return_values = []
                
                for item in portfolio_data:
                    try:
                        return_str = item['Return %'].replace('%', '').replace('+', '')
                        return_values.append(float(return_str))
                    except:
                        return_values.append(0.0)
                
                if return_values and symbols:
                    fig_returns = px.bar(
                        x=symbols,
                        y=return_values,
                        title="Return % by Holding",
                        color=return_values,
                        color_continuous_scale=['red', 'yellow', 'green']
                    )
                    fig_returns.update_layout(height=400, xaxis_title="Assets", yaxis_title="Return %")
                    st.plotly_chart(fig_returns, use_container_width=True)
        else:
            st.markdown("#### 📊 Simple Visualization")
            current_values = []
            symbols = []
            
            for item in portfolio_data:
                try:
                    value_str = item['Current Value'].replace('$', '').replace(',', '')
                    current_values.append(float(value_str))
                    symbols.append(item['Symbol'])
                except:
                    continue
            
            if current_values and symbols:
                chart_data = pd.DataFrame({
                    'Symbol': symbols,
                    'Current Value': current_values
                })
                st.bar_chart(chart_data.set_index('Symbol'))

def show_asset_management():
    """Display asset management page"""
    
    st.markdown("### 🎯 Manage Portfolio Assets")
    
    if st.session_state.learning_mode:
        st.markdown("""
        <div class="info-card">
            <strong>🎯 Asset Management Guide:</strong><br>
            Use this section to build and maintain your investment portfolio. Add assets, track purchase prices, and manage your holdings.
        </div>
        """, unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["➕ Add Assets", "➖ Remove Assets"])
    
    with tab1:
        st.markdown("### Add New Investment")
        
        col1, col2 = st.columns(2)
        
        with col1:
            popular_assets = get_popular_assets()
            
            # Create categories for better organization
            asset_categories = {
                "🏢 Large Cap Stocks": [k for k in popular_assets.keys() if k in ["AAPL", "GOOGL", "GOOG", "MSFT", "AMZN", "TSLA", "NVDA", "META", "NFLX", "JPM", "JNJ", "V", "WMT", "PG", "UNH", "HD", "MA", "BAC", "ADBE", "CRM", "XOM", "CVX", "KO", "PFE", "INTC", "CSCO", "VZ", "MRK", "ABT", "TMO"]],
                "🚀 Growth & Tech": [k for k in popular_assets.keys() if k in ["AMD", "PYPL", "SHOP", "SQ", "ROKU", "ZM", "SNOW", "PLTR", "CRWD", "NET"]],
                "🏛️ Vanguard ETFs": [k for k in popular_assets.keys() if k.startswith("V") and k in ["VTI", "VEA", "VWO", "VTV", "VUG", "VIG", "VYM", "VXUS", "VOO", "VT", "VB", "VO", "VGT", "VHT", "VFH", "VDE", "VAW", "VIS", "VCR", "VDC", "VPU", "VNQ"]],
                "⚫ BlackRock iShares": [k for k in popular_assets.keys() if k in ["QQQ", "IWM", "EFA", "EEM", "IVV", "IEFA", "IEMG", "IJH", "IJR", "IVW", "IVE", "IWF", "IWD", "ITOT", "IXUS", "IYY", "IWB", "IWV", "ACWI", "ACWX"]],
                "🎯 Sector ETFs": [k for k in popular_assets.keys() if k.startswith("XL") or k in ["IBB", "SOXX", "SKYY", "HACK", "ROBO", "ARKK", "ARKQ", "ARKW", "ARKG", "ARKF"]],
                "🏦 Bonds & Fixed Income": [k for k in popular_assets.keys() if k in ["AGG", "BND", "TLT", "IEF", "SHY", "LQD", "HYG", "JNK", "TIP", "VTEB", "MUB", "EMB", "BNDX", "VGIT", "VGSH", "VGLT", "VCIT", "VCSH", "BSV", "BIV", "BLV"]],
                "💎 Fidelity Funds": [k for k in popular_assets.keys() if k.startswith("F") and len(k) > 2],
                "🥇 Commodities": [k for k in popular_assets.keys() if k in ["GLD", "SLV", "USO", "UNG", "DBA", "DBC", "PDBC", "GSG", "COMT", "IAU", "SGOL", "PPLT", "PALL", "CORN", "WEAT", "SOYB", "CANE", "COW"]],
                "🌍 International": [k for k in popular_assets.keys() if k.startswith("EW") or k in ["FXI", "INDA", "RSX", "ASHR", "MCHI", "KWEB", "YINN"]],
                "₿ Major Cryptocurrencies": [k for k in popular_assets.keys() if k.endswith("-USD") and k in ["BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "ADA-USD", "SOL-USD", "DOGE-USD", "DOT-USD", "MATIC-USD", "SHIB-USD", "LTC-USD", "TRX-USD", "AVAX-USD", "LINK-USD", "ATOM-USD"]],
                "🔗 DeFi & Web3": [k for k in popular_assets.keys() if k.endswith("-USD") and k in ["XLM-USD", "ALGO-USD", "VET-USD", "ICP-USD", "FIL-USD", "SAND-USD", "MANA-USD", "AXS-USD", "ENJ-USD", "CHZ-USD", "FLOW-USD", "THETA-USD", "XTZ-USD", "EGLD-USD", "NEAR-USD"]],
                "📈 Crypto ETFs": [k for k in popular_assets.keys() if k in ["BITO", "BITI", "GBTC", "ETHE", "GDLC", "BITQ", "BLOK", "LEGR"]],
                "💱 Currency ETFs": [k for k in popular_assets.keys() if k.startswith("FX") or k in ["UUP", "UDN"]],
                "💰 Dividend ETFs": [k for k in popular_assets.keys() if k in ["SCHD", "DVY", "VYM", "NOBL", "DGRO", "HDV", "SPHD", "PEY", "RDVY", "FDL"]],
                "🌱 ESG & Clean Energy": [k for k in popular_assets.keys() if k in ["ESG", "ESGU", "ESGD", "ESGE", "VSGX", "ESGV", "SUSA", "DSI", "ICLN", "PBW", "QCLN"]],
                "📊 Volatility & Leverage": [k for k in popular_assets.keys() if k in ["VIX", "UVXY", "SVXY", "VXX", "VIXY", "TQQQ", "SQQQ", "UPRO", "SPXU", "TNA", "TZA", "TECL", "TECS", "FAS", "FAZ"]]
            }
            
            selected_category = st.selectbox(
                "Choose Asset Category",
                [""] + list(asset_categories.keys()),
                help="Browse assets by category for easier selection"
            )
            
            if selected_category:
                available_assets = asset_categories[selected_category]
                selected_popular = st.selectbox(
                    f"Assets in {selected_category}",
                    [""] + available_assets,
                    format_func=lambda x: f"{x} - {popular_assets[x]}" if x else "",
                    help="Select from assets in this category"
                )
            else:
                selected_popular = st.selectbox(
                    "Or search all assets",
                    [""] + list(popular_assets.keys()),
                    format_func=lambda x: f"{x} - {popular_assets[x]}" if x else "",
                    help="Search through all available assets"
                )
            
            custom_symbol = st.text_input(
                "Or Enter Custom Symbol",
                placeholder="e.g., AAPL, BTC-USD, GLD.L (London)",
                help="Enter any valid symbol including international exchanges"
            )
            
            symbol_to_use = ""
            if selected_popular:
                symbol_to_use = selected_popular
            elif custom_symbol:
                symbol_to_use = custom_symbol.upper()
        
        with col2:
            shares = st.number_input(
                "Number of Shares/Units",
                min_value=0.001,
                value=1.0,
                step=0.1,
                help="Enter the quantity you own"
            )
            
            purchase_price = st.number_input(
                "Purchase Price per Share ($)",
                min_value=0.01,
                value=100.0,
                step=0.01,
                help="Enter the average price you paid per share"
            )
            
            # NEW: Currency selection
            currencies = get_currency_list()
            selected_currency = st.selectbox(
                "Currency",
                list(currencies.keys()),
                index=0,  # Default to USD
                format_func=lambda x: f"{x} - {currencies[x]['name']}",
                help="Select the currency for this asset"
            )
            
            asset_type = st.selectbox(
                "Asset Type",
                ["Stock", "ETF", "Cryptocurrency", "Bond", "Commodity", "Index Fund", "Mutual Fund", "REIT", "Other"]
            )
            
            if st.session_state.learning_mode:
                st.markdown("""
                <div class="info-card">
                    <strong>💡 Pro Tip:</strong><br>
                    • Select the correct currency for international assets<br>
                    • Purchase price should be in the selected currency<br>
                    • The system will convert to your base currency for analysis
                </div>
                """, unsafe_allow_html=True)
        
        if symbol_to_use:
            asset_info = fetch_asset_data(symbol_to_use)
            if asset_info:
                current_price = asset_info['current_price']
                
                # Convert prices for display if different currency
                if selected_currency != "USD":
                    exchange_rate = get_exchange_rate("USD", selected_currency)
                    current_price_display = current_price * exchange_rate
                    purchase_price_display = purchase_price
                else:
                    current_price_display = current_price
                    purchase_price_display = purchase_price
                
                potential_return = ((current_price_display - purchase_price_display) / purchase_price_display) * 100 if purchase_price_display > 0 else 0
                
                # Create three columns for metrics display
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                
                with metric_col1:
                    currency_symbol = currencies[selected_currency]["symbol"]
                    st.metric("Current Price", f"{currency_symbol}{current_price_display:.2f}")
                with metric_col2:
                    st.metric("Your Purchase Price", f"{currency_symbol}{purchase_price_display:.2f}")
                with metric_col3:
                    st.metric("Potential Return", f"{potential_return:+.1f}%", 
                             delta=f"{currency_symbol}{(current_price_display - purchase_price_display) * shares:+.2f}")
                
                if st.button("✅ Add to Portfolio", type="primary", use_container_width=True):
                    st.session_state.portfolio[symbol_to_use] = {
                        'shares': shares,
                        'asset_type': asset_type,
                        'purchase_price': purchase_price,
                        'currency': selected_currency,  # NEW: Store currency
                        'added_date': datetime.now().isoformat()
                    }
                    
                    portfolios = load_portfolios()
                    portfolios[st.session_state.username] = st.session_state.portfolio
                    save_portfolios(portfolios)
                    
                    st.markdown("""
                    <div class="success-card">
                        <strong>🎉 Asset Added Successfully!</strong><br>
                        Your portfolio metrics are being recalculated with the new asset allocation and currency conversion.
                    </div>
                    """, unsafe_allow_html=True)
                    st.rerun()
            else:
                st.error(f"❌ Could not find asset data for '{symbol_to_use}'. Please check the symbol.")
    
    with tab2:
        st.markdown("### Remove Assets from Portfolio")
        
        if st.session_state.portfolio:
            current_values = []
            for symbol, data in st.session_state.portfolio.items():
                asset_info = fetch_asset_data(symbol)
                if asset_info:
                    current_value = data['shares'] * asset_info['current_price']
                    current_values.append({
                        'Symbol': symbol,
                        'Shares': f"{data['shares']:.3f}",
                        'Current Value': f"${current_value:,.2f}",
                        'Asset Type': data['asset_type']
                    })
            
            if current_values:
                df_current = pd.DataFrame(current_values)
                st.dataframe(df_current, hide_index=True)
            
            assets_to_remove = st.multiselect(
                "Select assets to remove:",
                list(st.session_state.portfolio.keys()),
                help="Choose one or more assets to remove from your portfolio"
            )
            
            if assets_to_remove:
                st.warning(f"⚠️ You are about to remove {len(assets_to_remove)} asset(s) from your portfolio.")
                
                if st.button("🗑️ Remove Selected Assets", type="secondary"):
                    for asset in assets_to_remove:
                        del st.session_state.portfolio[asset]
                    
                    portfolios = load_portfolios()
                    portfolios[st.session_state.username] = st.session_state.portfolio
                    save_portfolios(portfolios)
                    
                    st.markdown("""
                    <div class="success-card">
                        <strong>✅ Assets Removed Successfully!</strong><br>
                        Your portfolio has been updated and metrics recalculated.
                    </div>
                    """, unsafe_allow_html=True)
                    st.rerun()
        else:
            st.markdown("""
            <div class="info-card">
                <strong>📝 No assets to remove</strong><br>
                Your portfolio is currently empty. Add some investments first!
            </div>
            """, unsafe_allow_html=True)

def show_analytics_dashboard():
    """Display analytics dashboard"""
    
    st.markdown("### 📊 Advanced Analytics Dashboard")
    
    if st.session_state.learning_mode:
        st.markdown("""
        <div class="info-card">
            <strong>📈 Analytics Dashboard Guide:</strong><br>
            This dashboard provides professional-grade analysis tools including performance metrics, technical analysis, and AI-powered insights.
        </div>
        """, unsafe_allow_html=True)
    
    if not st.session_state.portfolio:
        st.warning("⚠️ Add assets to your portfolio to see advanced analytics.")
        return
    
    metrics = calculate_portfolio_metrics_advanced(st.session_state.portfolio)
    
    st.markdown("### 📈 Advanced Performance Metrics")
    
    perf_col1, perf_col2, perf_col3, perf_col4, perf_col5 = st.columns(5)
    
    with perf_col1:
        st.metric(
            "Annualized Return", 
            f"{metrics['annualized_return']:+.1f}%",
            help="Expected annual return based on current performance"
        )
    
    with perf_col2:
        st.metric(
            "Portfolio Beta", 
            f"{metrics['beta']:.2f}",
            delta="Market Risk" if metrics['beta'] > 1 else "Lower Risk",
            help="Portfolio volatility relative to market"
        )
    
    with perf_col3:
        st.metric(
            "Sharpe Ratio", 
            f"{metrics['sharpe_ratio']:.2f}",
            delta="Excellent" if metrics['sharpe_ratio'] > 1 else "Good" if metrics['sharpe_ratio'] > 0.5 else "Needs Improvement",
            help="Risk-adjusted return measure"
        )
    
    with perf_col4:
        st.metric(
            "Max Drawdown", 
            f"{metrics['max_drawdown']:.1f}%",
            delta="Risk Level",
            delta_color="inverse",
            help="Largest peak-to-trough decline"
        )
    
    with perf_col5:
        st.metric(
            "Portfolio Volatility", 
            f"{metrics['volatility']:.1f}%",
            help="Measure of price fluctuation"
        )
    
    # Technical analysis for top holdings
    st.markdown("### 🔍 Technical Analysis - Top Holdings")
    
    portfolio_values = []
    for symbol, data in st.session_state.portfolio.items():
        asset_info = fetch_asset_data(symbol)
        if asset_info:
            value = data['shares'] * asset_info['current_price']
            portfolio_values.append((symbol, value))
    
    portfolio_values.sort(key=lambda x: x[1], reverse=True)
    top_holdings = [item[0] for item in portfolio_values[:3]]
    
    for i, symbol in enumerate(top_holdings):
        with st.expander(f"📊 {symbol} - Technical Analysis", expanded=(i == 0)):
            indicators = calculate_technical_indicators(symbol)
            
            if indicators is not None:
                tech_col1, tech_col2, tech_col3, tech_col4 = st.columns(4)
                
                current_price = indicators['Close'].iloc[-1]
                ma20 = indicators['MA_20'].iloc[-1]
                current_rsi = indicators['RSI'].iloc[-1]
                current_macd = indicators['MACD'].iloc[-1]
                current_signal = indicators['MACD_Signal'].iloc[-1]
                
                with tech_col1:
                    price_trend = "Bullish" if current_price > ma20 else "Bearish"
                    st.metric("Price Trend", price_trend, f"${current_price:.2f}")
                
                with tech_col2:
                    rsi_status = "Overbought" if current_rsi > 70 else "Oversold" if current_rsi < 30 else "Neutral"
                    rsi_color = "🔴" if current_rsi > 70 else "🟢" if current_rsi < 30 else "🟡"
                    st.metric("RSI Signal", f"{rsi_color} {rsi_status}", f"{current_rsi:.1f}")
                
                with tech_col3:
                    macd_trend = "Bullish" if current_macd > current_signal else "Bearish"
                    macd_color = "🟢" if current_macd > current_signal else "🔴"
                    st.metric("MACD Signal", f"{macd_color} {macd_trend}")
                
                with tech_col4:
                    support_level = current_price * 0.95
                    resistance_level = current_price * 1.05
                    st.metric("Support/Resistance", f"${support_level:.2f} / ${resistance_level:.2f}")
                
                if PLOTLY_AVAILABLE:
                    fig = make_subplots(
                        rows=3, cols=1,
                        subplot_titles=(
                            f'{symbol} - Price & Moving Averages',
                            'Relative Strength Index (RSI)',
                            'MACD & Signal Line'
                        ),
                        vertical_spacing=0.08,
                        row_heights=[0.5, 0.25, 0.25]
                    )
                    
                    fig.add_trace(
                        go.Scatter(x=indicators.index, y=indicators['Close'], name='Price', line=dict(color='blue', width=2)),
                        row=1, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=indicators.index, y=indicators['MA_20'], name='MA 20', line=dict(color='orange')),
                        row=1, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=indicators.index, y=indicators['MA_50'], name='MA 50', line=dict(color='red')),
                        row=1, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(x=indicators.index, y=indicators['RSI'], name='RSI', line=dict(color='purple')),
                        row=2, col=1
                    )
                    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
                    
                    fig.add_trace(
                        go.Scatter(x=indicators.index, y=indicators['MACD'], name='MACD', line=dict(color='blue')),
                        row=3, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=indicators.index, y=indicators['MACD_Signal'], name='Signal', line=dict(color='red')),
                        row=3, col=1
                    )
                    
                    fig.update_layout(height=700, showlegend=True, title_text=f"{symbol} Technical Analysis")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.markdown("#### 📈 Price Chart")
                    st.line_chart(indicators[['Close', 'MA_20', 'MA_50']])
                    
                    simple_col1, simple_col2 = st.columns(2)
                    with simple_col1:
                        st.markdown("#### RSI")
                        st.line_chart(indicators['RSI'])
                    with simple_col2:
                        st.markdown("#### MACD")
                        st.line_chart(indicators[['MACD', 'MACD_Signal']])
    
    # Investment suggestions
    st.markdown("### 🤖 AI-Powered Investment Insights")
    suggestions = generate_investment_suggestions(st.session_state.portfolio)
    
    enhanced_suggestions = []
    
    if metrics['total_return'] > 10:
        enhanced_suggestions.append({
            'type': 'success',
            'message': f'🎉 Excellent performance! Your portfolio is up {metrics["total_return"]:.1f}%. Consider taking some profits and rebalancing.'
        })
    elif metrics['total_return'] < -5:
        enhanced_suggestions.append({
            'type': 'warning',
            'message': f'📉 Portfolio is down {abs(metrics["total_return"]):.1f}%. Review underperforming assets and consider strategic rebalancing.'
        })
    
    if metrics['volatility'] > 25:
        enhanced_suggestions.append({
            'type': 'warning',
            'message': f'⚡ High volatility detected ({metrics["volatility"]:.1f}%). Consider adding stable assets like bonds or dividend stocks.'
        })
    
    all_suggestions = enhanced_suggestions + suggestions
    
    for suggestion in all_suggestions[:6]:
        if suggestion['type'] == 'diversification':
            st.markdown(f"""
            <div class="info-card">
                <strong>🎯 Diversification:</strong> {suggestion['message']}
            </div>
            """, unsafe_allow_html=True)
        elif suggestion['type'] == 'rebalancing':
            st.markdown(f"""
            <div class="warning-card">
                <strong>⚖️ Rebalancing:</strong> {suggestion['message']}
            </div>
            """, unsafe_allow_html=True)
        elif suggestion['type'] == 'opportunity':
            st.markdown(f"""
            <div class="success-card">
                <strong>🚀 Opportunity:</strong> {suggestion['message']}
            </div>
            """, unsafe_allow_html=True)
        elif suggestion['type'] == 'success':
            st.markdown(f"""
            <div class="success-card">
                <strong>✅ Success:</strong> {suggestion['message']}
            </div>
            """, unsafe_allow_html=True)
        elif suggestion['type'] == 'warning':
            st.markdown(f"""
            <div class="warning-card">
                <strong>⚠️ Warning:</strong> {suggestion['message']}
            </div>
            """, unsafe_allow_html=True)

def show_export_import():
    """Display export/import functionality"""
    
    st.markdown("### 📁 Export & Import Portfolio")
    
    if st.session_state.learning_mode:
        st.markdown("""
        <div class="info-card">
            <strong>📁 Export/Import Guide:</strong><br>
            Backup and restore your portfolio data. Export as JSON or CSV for safekeeping, or import previously saved portfolios.
        </div>
        """, unsafe_allow_html=True)
    
    export_tab, import_tab = st.tabs(["📤 Export Portfolio", "📥 Import Portfolio"])
    
    with export_tab:
        st.markdown("### Export Your Portfolio Data")
        
        if not st.session_state.portfolio:
            st.markdown("""
            <div class="warning-card">
                <strong>📝 No portfolio data to export</strong><br>
                Add some investments to your portfolio first, then return here to export your data.
            </div>
            """, unsafe_allow_html=True)
            return
        
        export_format = st.selectbox("Choose Export Format", ["JSON", "CSV"])
        
        if export_format == "JSON":
            export_data = {
                'username': st.session_state.username,
                'export_date': datetime.now().isoformat(),
                'portfolio': st.session_state.portfolio,
                'version': '2.0'
            }
            
            json_string = json.dumps(export_data, indent=2)
            
            st.download_button(
                label="📥 Download Portfolio (JSON)",
                data=json_string,
                file_name=f"portfolio_{st.session_state.username}_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )
            
            with st.expander("👀 Preview JSON Data"):
                st.code(json_string, language="json")
        
        else:  # CSV format
            csv_data = []
            for symbol, data in st.session_state.portfolio.items():
                asset_info = fetch_asset_data(symbol)
                if asset_info:
                    csv_data.append({
                        'Symbol': symbol,
                        'Name': asset_info['name'],
                        'Shares': data['shares'],
                        'Asset_Type': data['asset_type'],
                        'Purchase_Price': data.get('purchase_price', asset_info['current_price']),
                        'Current_Price': asset_info['current_price'],
                        'Current_Value': data['shares'] * asset_info['current_price'],
                        'Added_Date': data['added_date']
                    })
            
            if csv_data:
                df = pd.DataFrame(csv_data)
                csv_string = df.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download Portfolio (CSV)",
                    data=csv_string,
                    file_name=f"portfolio_{st.session_state.username}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
                
                with st.expander("👀 Preview CSV Data"):
                    st.dataframe(df)
    
    with import_tab:
        st.markdown("### Import Portfolio Data")
        
        uploaded_file = st.file_uploader(
            "Choose a portfolio file",
            type=['json', 'csv'],
            help="Upload a previously exported portfolio file"
        )
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.json'):
                    content = uploaded_file.read()
                    data = json.loads(content)
                    
                    if 'portfolio' in data:
                        st.markdown("""
                        <div class="success-card">
                            <strong>✅ Valid portfolio file detected!</strong><br>
                            Ready to import your portfolio data.
                        </div>
                        """, unsafe_allow_html=True)
                        
                        with st.expander("👀 Preview Import Data"):
                            st.json(data['portfolio'])
                        
                        import_option = st.selectbox(
                            "Import Option",
                            ["Replace Current Portfolio", "Merge with Current Portfolio"]
                        )
                        
                        if st.button("🚀 Import Portfolio", type="primary"):
                            if import_option == "Replace Current Portfolio":
                                st.session_state.portfolio = data['portfolio']
                            else:
                                for symbol, asset_data in data['portfolio'].items():
                                    st.session_state.portfolio[symbol] = asset_data
                            
                            portfolios = load_portfolios()
                            portfolios[st.session_state.username] = st.session_state.portfolio
                            save_portfolios(portfolios)
                            
                            st.markdown("""
                            <div class="success-card">
                                <strong>🎉 Portfolio imported successfully!</strong><br>
                                Your portfolio has been updated with the imported data.
                            </div>
                            """, unsafe_allow_html=True)
                            st.rerun()
                    else:
                        st.error("❌ Invalid portfolio file format.")
                
                elif uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                    
                    required_columns = ['Symbol', 'Shares', 'Asset_Type']
                    if all(col in df.columns for col in required_columns):
                        st.markdown("""
                        <div class="success-card">
                            <strong>✅ Valid CSV file detected!</strong><br>
                            Ready to import your portfolio data.
                        </div>
                        """, unsafe_allow_html=True)
                        
                        with st.expander("👀 Preview Import Data"):
                            st.dataframe(df)
                        
                        csv_import_option = st.selectbox(
                            "Import Option",
                            ["Replace Current Portfolio", "Merge with Current Portfolio"],
                            key="csv_import_option"
                        )
                        
                        if st.button("🚀 Import CSV Portfolio", type="primary"):
                            new_portfolio = {}
                            
                            for _, row in df.iterrows():
                                symbol = row['Symbol']
                                new_portfolio[symbol] = {
                                    'shares': float(row['Shares']),
                                    'asset_type': row['Asset_Type'],
                                    'purchase_price': float(row.get('Purchase_Price', 100.0)),
                                    'added_date': datetime.now().isoformat()
                                }
                            
                            if csv_import_option == "Replace Current Portfolio":
                                st.session_state.portfolio = new_portfolio
                            else:
                                for symbol, asset_data in new_portfolio.items():
                                    st.session_state.portfolio[symbol] = asset_data
                            
                            portfolios = load_portfolios()
                            portfolios[st.session_state.username] = st.session_state.portfolio
                            save_portfolios(portfolios)
                            
                            st.markdown("""
                            <div class="success-card">
                                <strong>🎉 CSV portfolio imported successfully!</strong><br>
                                Your portfolio has been updated with the imported data.
                            </div>
                            """, unsafe_allow_html=True)
                            st.rerun()
                    else:
                        st.error(f"❌ CSV file must contain columns: {', '.join(required_columns)}")
                        
            except Exception as e:
                st.error(f"❌ Error importing file: {str(e)}")

if __name__ == "__main__":
    main(), 'HK"{return_pct:+.1f}%",
                'P&L': f"${current_value - invested_value:+,.2f}",
                'Asset Type': data['asset_type'],
                'Weight %': f"{(current_value / metrics['total_current_value']) * 100:.1f}%" if metrics['total_current_value'] > 0 else "0.0%"
            })
    
    if portfolio_data:
        st.markdown("### 📊 Holdings Breakdown")
        df = pd.DataFrame(portfolio_data)
        st.dataframe(df, use_container_width=True, height=400)
        
        if PLOTLY_AVAILABLE:
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                st.markdown("#### 🥧 Portfolio Allocation")
                # Extract numeric values safely
                current_values = []
                symbols = []
                
                for item in portfolio_data:
                    try:
                        value_str = item['Current Value'].replace('$', '').replace(',', '')
                        current_values.append(float(value_str))
                        symbols.append(item['Symbol'])
                    except:
                        continue
                
                if current_values:
                    fig_pie = px.pie(
                        values=current_values,
                        names=symbols,
                        title="Current Holdings Distribution",
                        hole=0.4
                    )
                    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                    fig_pie.update_layout(height=400)
                    st.plotly_chart(fig_pie, use_container_width=True)
            
            with col_chart2:
                st.markdown("#### 📈 Performance by Asset")
                return_values = []
                
                for item in portfolio_data:
                    try:
                        return_str = item['Return %'].replace('%', '').replace('+', '')
                        return_values.append(float(return_str))
                    except:
                        return_values.append(0.0)
                
                if return_values and symbols:
                    fig_returns = px.bar(
                        x=symbols,
                        y=return_values,
                        title="Return % by Holding",
                        color=return_values,
                        color_continuous_scale=['red', 'yellow', 'green']
                    )
                    fig_returns.update_layout(height=400, xaxis_title="Assets", yaxis_title="Return %")
                    st.plotly_chart(fig_returns, use_container_width=True)
        else:
            st.markdown("#### 📊 Simple Visualization")
            current_values = []
            symbols = []
            
            for item in portfolio_data:
                try:
                    value_str = item['Current Value'].replace('$', '').replace(',', '')
                    current_values.append(float(value_str))
                    symbols.append(item['Symbol'])
                except:
                    continue
            
            if current_values and symbols:
                chart_data = pd.DataFrame({
                    'Symbol': symbols,
                    'Current Value': current_values
                })
                st.bar_chart(chart_data.set_index('Symbol'))

def show_asset_management():
    """Display asset management page"""
    
    st.markdown("### 🎯 Manage Portfolio Assets")
    
    if st.session_state.learning_mode:
        st.markdown("""
        <div class="info-card">
            <strong>🎯 Asset Management Guide:</strong><br>
            Use this section to build and maintain your investment portfolio. Add assets, track purchase prices, and manage your holdings.
        </div>
        """, unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["➕ Add Assets", "➖ Remove Assets"])
    
    with tab1:
        st.markdown("### Add New Investment")
        
        col1, col2 = st.columns(2)
        
        with col1:
            popular_assets = get_popular_assets()
            
            # Create categories for better organization
            asset_categories = {
                "🏢 Large Cap Stocks": [k for k in popular_assets.keys() if k in ["AAPL", "GOOGL", "GOOG", "MSFT", "AMZN", "TSLA", "NVDA", "META", "NFLX", "JPM", "JNJ", "V", "WMT", "PG", "UNH", "HD", "MA", "BAC", "ADBE", "CRM", "XOM", "CVX", "KO", "PFE", "INTC", "CSCO", "VZ", "MRK", "ABT", "TMO"]],
                "🚀 Growth & Tech": [k for k in popular_assets.keys() if k in ["AMD", "PYPL", "SHOP", "SQ", "ROKU", "ZM", "SNOW", "PLTR", "CRWD", "NET"]],
                "🏛️ Vanguard ETFs": [k for k in popular_assets.keys() if k.startswith("V") and k in ["VTI", "VEA", "VWO", "VTV", "VUG", "VIG", "VYM", "VXUS", "VOO", "VT", "VB", "VO", "VGT", "VHT", "VFH", "VDE", "VAW", "VIS", "VCR", "VDC", "VPU", "VNQ"]],
                "⚫ BlackRock iShares": [k for k in popular_assets.keys() if k in ["QQQ", "IWM", "EFA", "EEM", "IVV", "IEFA", "IEMG", "IJH", "IJR", "IVW", "IVE", "IWF", "IWD", "ITOT", "IXUS", "IYY", "IWB", "IWV", "ACWI", "ACWX"]],
                "🎯 Sector ETFs": [k for k in popular_assets.keys() if k.startswith("XL") or k in ["IBB", "SOXX", "SKYY", "HACK", "ROBO", "ARKK", "ARKQ", "ARKW", "ARKG", "ARKF"]],
                "🏦 Bonds & Fixed Income": [k for k in popular_assets.keys() if k in ["AGG", "BND", "TLT", "IEF", "SHY", "LQD", "HYG", "JNK", "TIP", "VTEB", "MUB", "EMB", "BNDX", "VGIT", "VGSH", "VGLT", "VCIT", "VCSH", "BSV", "BIV", "BLV"]],
                "💎 Fidelity Funds": [k for k in popular_assets.keys() if k.startswith("F") and len(k) > 2],
                "🥇 Commodities": [k for k in popular_assets.keys() if k in ["GLD", "SLV", "USO", "UNG", "DBA", "DBC", "PDBC", "GSG", "COMT", "IAU", "SGOL", "PPLT", "PALL", "CORN", "WEAT", "SOYB", "CANE", "COW"]],
                "🌍 International": [k for k in popular_assets.keys() if k.startswith("EW") or k in ["FXI", "INDA", "RSX", "ASHR", "MCHI", "KWEB", "YINN"]],
                "₿ Major Cryptocurrencies": [k for k in popular_assets.keys() if k.endswith("-USD") and k in ["BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "ADA-USD", "SOL-USD", "DOGE-USD", "DOT-USD", "MATIC-USD", "SHIB-USD", "LTC-USD", "TRX-USD", "AVAX-USD", "LINK-USD", "ATOM-USD"]],
                "🔗 DeFi & Web3": [k for k in popular_assets.keys() if k.endswith("-USD") and k in ["XLM-USD", "ALGO-USD", "VET-USD", "ICP-USD", "FIL-USD", "SAND-USD", "MANA-USD", "AXS-USD", "ENJ-USD", "CHZ-USD", "FLOW-USD", "THETA-USD", "XTZ-USD", "EGLD-USD", "NEAR-USD"]],
                "📈 Crypto ETFs": [k for k in popular_assets.keys() if k in ["BITO", "BITI", "GBTC", "ETHE", "GDLC", "BITQ", "BLOK", "LEGR"]],
                "💱 Currency ETFs": [k for k in popular_assets.keys() if k.startswith("FX") or k in ["UUP", "UDN"]],
                "💰 Dividend ETFs": [k for k in popular_assets.keys() if k in ["SCHD", "DVY", "VYM", "NOBL", "DGRO", "HDV", "SPHD", "PEY", "RDVY", "FDL"]],
                "🌱 ESG & Clean Energy": [k for k in popular_assets.keys() if k in ["ESG", "ESGU", "ESGD", "ESGE", "VSGX", "ESGV", "SUSA", "DSI", "ICLN", "PBW", "QCLN"]],
                "📊 Volatility & Leverage": [k for k in popular_assets.keys() if k in ["VIX", "UVXY", "SVXY", "VXX", "VIXY", "TQQQ", "SQQQ", "UPRO", "SPXU", "TNA", "TZA", "TECL", "TECS", "FAS", "FAZ"]]
            }
            
            selected_category = st.selectbox(
                "Choose Asset Category",
                [""] + list(asset_categories.keys()),
                help="Browse assets by category for easier selection"
            )
            
            if selected_category:
                available_assets = asset_categories[selected_category]
                selected_popular = st.selectbox(
                    f"Assets in {selected_category}",
                    [""] + available_assets,
                    format_func=lambda x: f"{x} - {popular_assets[x]}" if x else "",
                    help="Select from assets in this category"
                )
            else:
                selected_popular = st.selectbox(
                    "Or search all assets",
                    [""] + list(popular_assets.keys()),
                    format_func=lambda x: f"{x} - {popular_assets[x]}" if x else "",
                    help="Search through all available assets"
                )
            
            custom_symbol = st.text_input(
                "Or Enter Custom Symbol",
                placeholder="e.g., AAPL, BTC-USD, GLD.L (London)",
                help="Enter any valid symbol including international exchanges"
            )
            
            symbol_to_use = ""
            if selected_popular:
                symbol_to_use = selected_popular
            elif custom_symbol:
                symbol_to_use = custom_symbol.upper()
        
        with col2:
            shares = st.number_input(
                "Number of Shares/Units",
                min_value=0.001,
                value=1.0,
                step=0.1,
                help="Enter the quantity you own"
            )
            
            purchase_price = st.number_input(
                "Purchase Price per Share ($)",
                min_value=0.01,
                value=100.0,
                step=0.01,
                help="Enter the average price you paid per share"
            )
            
            # NEW: Currency selection
            currencies = get_currency_list()
            selected_currency = st.selectbox(
                "Currency",
                list(currencies.keys()),
                index=0,  # Default to USD
                format_func=lambda x: f"{x} - {currencies[x]['name']}",
                help="Select the currency for this asset"
            )
            
            asset_type = st.selectbox(
                "Asset Type",
                ["Stock", "ETF", "Cryptocurrency", "Bond", "Commodity", "Index Fund", "Mutual Fund", "REIT", "Other"]
            )
            
            if st.session_state.learning_mode:
                st.markdown("""
                <div class="info-card">
                    <strong>💡 Pro Tip:</strong><br>
                    • Select the correct currency for international assets<br>
                    • Purchase price should be in the selected currency<br>
                    • The system will convert to your base currency for analysis
                </div>
                """, unsafe_allow_html=True)
        
        if symbol_to_use:
            asset_info = fetch_asset_data(symbol_to_use)
            if asset_info:
                current_price = asset_info['current_price']
                
                # Convert prices for display if different currency
                if selected_currency != "USD":
                    exchange_rate = get_exchange_rate("USD", selected_currency)
                    current_price_display = current_price * exchange_rate
                    purchase_price_display = purchase_price
                else:
                    current_price_display = current_price
                    purchase_price_display = purchase_price
                
                potential_return = ((current_price_display - purchase_price_display) / purchase_price_display) * 100 if purchase_price_display > 0 else 0
                
                # Create three columns for metrics display
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                
                with metric_col1:
                    currency_symbol = currencies[selected_currency]["symbol"]
                    st.metric("Current Price", f"{currency_symbol}{current_price_display:.2f}")
                with metric_col2:
                    st.metric("Your Purchase Price", f"{currency_symbol}{purchase_price_display:.2f}")
                with metric_col3:
                    st.metric("Potential Return", f"{potential_return:+.1f}%", 
                             delta=f"{currency_symbol}{(current_price_display - purchase_price_display) * shares:+.2f}")
                
                if st.button("✅ Add to Portfolio", type="primary", use_container_width=True):
                    st.session_state.portfolio[symbol_to_use] = {
                        'shares': shares,
                        'asset_type': asset_type,
                        'purchase_price': purchase_price,
                        'currency': selected_currency,  # NEW: Store currency
                        'added_date': datetime.now().isoformat()
                    }
                    
                    portfolios = load_portfolios()
                    portfolios[st.session_state.username] = st.session_state.portfolio
                    save_portfolios(portfolios)
                    
                    st.markdown("""
                    <div class="success-card">
                        <strong>🎉 Asset Added Successfully!</strong><br>
                        Your portfolio metrics are being recalculated with the new asset allocation and currency conversion.
                    </div>
                    """, unsafe_allow_html=True)
                    st.rerun()
            else:
                st.error(f"❌ Could not find asset data for '{symbol_to_use}'. Please check the symbol.")
    
    with tab2:
        st.markdown("### Remove Assets from Portfolio")
        
        if st.session_state.portfolio:
            current_values = []
            for symbol, data in st.session_state.portfolio.items():
                asset_info = fetch_asset_data(symbol)
                if asset_info:
                    current_value = data['shares'] * asset_info['current_price']
                    current_values.append({
                        'Symbol': symbol,
                        'Shares': f"{data['shares']:.3f}",
                        'Current Value': f"${current_value:,.2f}",
                        'Asset Type': data['asset_type']
                    })
            
            if current_values:
                df_current = pd.DataFrame(current_values)
                st.dataframe(df_current, hide_index=True)
            
            assets_to_remove = st.multiselect(
                "Select assets to remove:",
                list(st.session_state.portfolio.keys()),
                help="Choose one or more assets to remove from your portfolio"
            )
            
            if assets_to_remove:
                st.warning(f"⚠️ You are about to remove {len(assets_to_remove)} asset(s) from your portfolio.")
                
                if st.button("🗑️ Remove Selected Assets", type="secondary"):
                    for asset in assets_to_remove:
                        del st.session_state.portfolio[asset]
                    
                    portfolios = load_portfolios()
                    portfolios[st.session_state.username] = st.session_state.portfolio
                    save_portfolios(portfolios)
                    
                    st.markdown("""
                    <div class="success-card">
                        <strong>✅ Assets Removed Successfully!</strong><br>
                        Your portfolio has been updated and metrics recalculated.
                    </div>
                    """, unsafe_allow_html=True)
                    st.rerun()
        else:
            st.markdown("""
            <div class="info-card">
                <strong>📝 No assets to remove</strong><br>
                Your portfolio is currently empty. Add some investments first!
            </div>
            """, unsafe_allow_html=True)

def show_analytics_dashboard():
    """Display analytics dashboard"""
    
    st.markdown("### 📊 Advanced Analytics Dashboard")
    
    if st.session_state.learning_mode:
        st.markdown("""
        <div class="info-card">
            <strong>📈 Analytics Dashboard Guide:</strong><br>
            This dashboard provides professional-grade analysis tools including performance metrics, technical analysis, and AI-powered insights.
        </div>
        """, unsafe_allow_html=True)
    
    if not st.session_state.portfolio:
        st.warning("⚠️ Add assets to your portfolio to see advanced analytics.")
        return
    
    metrics = calculate_portfolio_metrics_advanced(st.session_state.portfolio)
    
    st.markdown("### 📈 Advanced Performance Metrics")
    
    perf_col1, perf_col2, perf_col3, perf_col4, perf_col5 = st.columns(5)
    
    with perf_col1:
        st.metric(
            "Annualized Return", 
            f"{metrics['annualized_return']:+.1f}%",
            help="Expected annual return based on current performance"
        )
    
    with perf_col2:
        st.metric(
            "Portfolio Beta", 
            f"{metrics['beta']:.2f}",
            delta="Market Risk" if metrics['beta'] > 1 else "Lower Risk",
            help="Portfolio volatility relative to market"
        )
    
    with perf_col3:
        st.metric(
            "Sharpe Ratio", 
            f"{metrics['sharpe_ratio']:.2f}",
            delta="Excellent" if metrics['sharpe_ratio'] > 1 else "Good" if metrics['sharpe_ratio'] > 0.5 else "Needs Improvement",
            help="Risk-adjusted return measure"
        )
    
    with perf_col4:
        st.metric(
            "Max Drawdown", 
            f"{metrics['max_drawdown']:.1f}%",
            delta="Risk Level",
            delta_color="inverse",
            help="Largest peak-to-trough decline"
        )
    
    with perf_col5:
        st.metric(
            "Portfolio Volatility", 
            f"{metrics['volatility']:.1f}%",
            help="Measure of price fluctuation"
        )
    
    # Technical analysis for top holdings
    st.markdown("### 🔍 Technical Analysis - Top Holdings")
    
    portfolio_values = []
    for symbol, data in st.session_state.portfolio.items():
        asset_info = fetch_asset_data(symbol)
        if asset_info:
            value = data['shares'] * asset_info['current_price']
            portfolio_values.append((symbol, value))
    
    portfolio_values.sort(key=lambda x: x[1], reverse=True)
    top_holdings = [item[0] for item in portfolio_values[:3]]
    
    for i, symbol in enumerate(top_holdings):
        with st.expander(f"📊 {symbol} - Technical Analysis", expanded=(i == 0)):
            indicators = calculate_technical_indicators(symbol)
            
            if indicators is not None:
                tech_col1, tech_col2, tech_col3, tech_col4 = st.columns(4)
                
                current_price = indicators['Close'].iloc[-1]
                ma20 = indicators['MA_20'].iloc[-1]
                current_rsi = indicators['RSI'].iloc[-1]
                current_macd = indicators['MACD'].iloc[-1]
                current_signal = indicators['MACD_Signal'].iloc[-1]
                
                with tech_col1:
                    price_trend = "Bullish" if current_price > ma20 else "Bearish"
                    st.metric("Price Trend", price_trend, f"${current_price:.2f}")
                
                with tech_col2:
                    rsi_status = "Overbought" if current_rsi > 70 else "Oversold" if current_rsi < 30 else "Neutral"
                    rsi_color = "🔴" if current_rsi > 70 else "🟢" if current_rsi < 30 else "🟡"
                    st.metric("RSI Signal", f"{rsi_color} {rsi_status}", f"{current_rsi:.1f}")
                
                with tech_col3:
                    macd_trend = "Bullish" if current_macd > current_signal else "Bearish"
                    macd_color = "🟢" if current_macd > current_signal else "🔴"
                    st.metric("MACD Signal", f"{macd_color} {macd_trend}")
                
                with tech_col4:
                    support_level = current_price * 0.95
                    resistance_level = current_price * 1.05
                    st.metric("Support/Resistance", f"${support_level:.2f} / ${resistance_level:.2f}")
                
                if PLOTLY_AVAILABLE:
                    fig = make_subplots(
                        rows=3, cols=1,
                        subplot_titles=(
                            f'{symbol} - Price & Moving Averages',
                            'Relative Strength Index (RSI)',
                            'MACD & Signal Line'
                        ),
                        vertical_spacing=0.08,
                        row_heights=[0.5, 0.25, 0.25]
                    )
                    
                    fig.add_trace(
                        go.Scatter(x=indicators.index, y=indicators['Close'], name='Price', line=dict(color='blue', width=2)),
                        row=1, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=indicators.index, y=indicators['MA_20'], name='MA 20', line=dict(color='orange')),
                        row=1, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=indicators.index, y=indicators['MA_50'], name='MA 50', line=dict(color='red')),
                        row=1, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(x=indicators.index, y=indicators['RSI'], name='RSI', line=dict(color='purple')),
                        row=2, col=1
                    )
                    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
                    
                    fig.add_trace(
                        go.Scatter(x=indicators.index, y=indicators['MACD'], name='MACD', line=dict(color='blue')),
                        row=3, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=indicators.index, y=indicators['MACD_Signal'], name='Signal', line=dict(color='red')),
                        row=3, col=1
                    )
                    
                    fig.update_layout(height=700, showlegend=True, title_text=f"{symbol} Technical Analysis")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.markdown("#### 📈 Price Chart")
                    st.line_chart(indicators[['Close', 'MA_20', 'MA_50']])
                    
                    simple_col1, simple_col2 = st.columns(2)
                    with simple_col1:
                        st.markdown("#### RSI")
                        st.line_chart(indicators['RSI'])
                    with simple_col2:
                        st.markdown("#### MACD")
                        st.line_chart(indicators[['MACD', 'MACD_Signal']])
    
    # Investment suggestions
    st.markdown("### 🤖 AI-Powered Investment Insights")
    suggestions = generate_investment_suggestions(st.session_state.portfolio)
    
    enhanced_suggestions = []
    
    if metrics['total_return'] > 10:
        enhanced_suggestions.append({
            'type': 'success',
            'message': f'🎉 Excellent performance! Your portfolio is up {metrics["total_return"]:.1f}%. Consider taking some profits and rebalancing.'
        })
    elif metrics['total_return'] < -5:
        enhanced_suggestions.append({
            'type': 'warning',
            'message': f'📉 Portfolio is down {abs(metrics["total_return"]):.1f}%. Review underperforming assets and consider strategic rebalancing.'
        })
    
    if metrics['volatility'] > 25:
        enhanced_suggestions.append({
            'type': 'warning',
            'message': f'⚡ High volatility detected ({metrics["volatility"]:.1f}%). Consider adding stable assets like bonds or dividend stocks.'
        })
    
    all_suggestions = enhanced_suggestions + suggestions
    
    for suggestion in all_suggestions[:6]:
        if suggestion['type'] == 'diversification':
            st.markdown(f"""
            <div class="info-card">
                <strong>🎯 Diversification:</strong> {suggestion['message']}
            </div>
            """, unsafe_allow_html=True)
        elif suggestion['type'] == 'rebalancing':
            st.markdown(f"""
            <div class="warning-card">
                <strong>⚖️ Rebalancing:</strong> {suggestion['message']}
            </div>
            """, unsafe_allow_html=True)
        elif suggestion['type'] == 'opportunity':
            st.markdown(f"""
            <div class="success-card">
                <strong>🚀 Opportunity:</strong> {suggestion['message']}
            </div>
            """, unsafe_allow_html=True)
        elif suggestion['type'] == 'success':
            st.markdown(f"""
            <div class="success-card">
                <strong>✅ Success:</strong> {suggestion['message']}
            </div>
            """, unsafe_allow_html=True)
        elif suggestion['type'] == 'warning':
            st.markdown(f"""
            <div class="warning-card">
                <strong>⚠️ Warning:</strong> {suggestion['message']}
            </div>
            """, unsafe_allow_html=True)

def show_export_import():
    """Display export/import functionality"""
    
    st.markdown("### 📁 Export & Import Portfolio")
    
    if st.session_state.learning_mode:
        st.markdown("""
        <div class="info-card">
            <strong>📁 Export/Import Guide:</strong><br>
            Backup and restore your portfolio data. Export as JSON or CSV for safekeeping, or import previously saved portfolios.
        </div>
        """, unsafe_allow_html=True)
    
    export_tab, import_tab = st.tabs(["📤 Export Portfolio", "📥 Import Portfolio"])
    
    with export_tab:
        st.markdown("### Export Your Portfolio Data")
        
        if not st.session_state.portfolio:
            st.markdown("""
            <div class="warning-card">
                <strong>📝 No portfolio data to export</strong><br>
                Add some investments to your portfolio first, then return here to export your data.
            </div>
            """, unsafe_allow_html=True)
            return
        
        export_format = st.selectbox("Choose Export Format", ["JSON", "CSV"])
        
        if export_format == "JSON":
            export_data = {
                'username': st.session_state.username,
                'export_date': datetime.now().isoformat(),
                'portfolio': st.session_state.portfolio,
                'version': '2.0'
            }
            
            json_string = json.dumps(export_data, indent=2)
            
            st.download_button(
                label="📥 Download Portfolio (JSON)",
                data=json_string,
                file_name=f"portfolio_{st.session_state.username}_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )
            
            with st.expander("👀 Preview JSON Data"):
                st.code(json_string, language="json")
        
        else:  # CSV format
            csv_data = []
            for symbol, data in st.session_state.portfolio.items():
                asset_info = fetch_asset_data(symbol)
                if asset_info:
                    csv_data.append({
                        'Symbol': symbol,
                        'Name': asset_info['name'],
                        'Shares': data['shares'],
                        'Asset_Type': data['asset_type'],
                        'Purchase_Price': data.get('purchase_price', asset_info['current_price']),
                        'Current_Price': asset_info['current_price'],
                        'Current_Value': data['shares'] * asset_info['current_price'],
                        'Added_Date': data['added_date']
                    })
            
            if csv_data:
                df = pd.DataFrame(csv_data)
                csv_string = df.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download Portfolio (CSV)",
                    data=csv_string,
                    file_name=f"portfolio_{st.session_state.username}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
                
                with st.expander("👀 Preview CSV Data"):
                    st.dataframe(df)
    
    with import_tab:
        st.markdown("### Import Portfolio Data")
        
        uploaded_file = st.file_uploader(
            "Choose a portfolio file",
            type=['json', 'csv'],
            help="Upload a previously exported portfolio file"
        )
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.json'):
                    content = uploaded_file.read()
                    data = json.loads(content)
                    
                    if 'portfolio' in data:
                        st.markdown("""
                        <div class="success-card">
                            <strong>✅ Valid portfolio file detected!</strong><br>
                            Ready to import your portfolio data.
                        </div>
                        """, unsafe_allow_html=True)
                        
                        with st.expander("👀 Preview Import Data"):
                            st.json(data['portfolio'])
                        
                        import_option = st.selectbox(
                            "Import Option",
                            ["Replace Current Portfolio", "Merge with Current Portfolio"]
                        )
                        
                        if st.button("🚀 Import Portfolio", type="primary"):
                            if import_option == "Replace Current Portfolio":
                                st.session_state.portfolio = data['portfolio']
                            else:
                                for symbol, asset_data in data['portfolio'].items():
                                    st.session_state.portfolio[symbol] = asset_data
                            
                            portfolios = load_portfolios()
                            portfolios[st.session_state.username] = st.session_state.portfolio
                            save_portfolios(portfolios)
                            
                            st.markdown("""
                            <div class="success-card">
                                <strong>🎉 Portfolio imported successfully!</strong><br>
                                Your portfolio has been updated with the imported data.
                            </div>
                            """, unsafe_allow_html=True)
                            st.rerun()
                    else:
                        st.error("❌ Invalid portfolio file format.")
                
                elif uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                    
                    required_columns = ['Symbol', 'Shares', 'Asset_Type']
                    if all(col in df.columns for col in required_columns):
                        st.markdown("""
                        <div class="success-card">
                            <strong>✅ Valid CSV file detected!</strong><br>
                            Ready to import your portfolio data.
                        </div>
                        """, unsafe_allow_html=True)
                        
                        with st.expander("👀 Preview Import Data"):
                            st.dataframe(df)
                        
                        csv_import_option = st.selectbox(
                            "Import Option",
                            ["Replace Current Portfolio", "Merge with Current Portfolio"],
                            key="csv_import_option"
                        )
                        
                        if st.button("🚀 Import CSV Portfolio", type="primary"):
                            new_portfolio = {}
                            
                            for _, row in df.iterrows():
                                symbol = row['Symbol']
                                new_portfolio[symbol] = {
                                    'shares': float(row['Shares']),
                                    'asset_type': row['Asset_Type'],
                                    'purchase_price': float(row.get('Purchase_Price', 100.0)),
                                    'added_date': datetime.now().isoformat()
                                }
                            
                            if csv_import_option == "Replace Current Portfolio":
                                st.session_state.portfolio = new_portfolio
                            else:
                                for symbol, asset_data in new_portfolio.items():
                                    st.session_state.portfolio[symbol] = asset_data
                            
                            portfolios = load_portfolios()
                            portfolios[st.session_state.username] = st.session_state.portfolio
                            save_portfolios(portfolios)
                            
                            st.markdown("""
                            <div class="success-card">
                                <strong>🎉 CSV portfolio imported successfully!</strong><br>
                                Your portfolio has been updated with the imported data.
                            </div>
                            """, unsafe_allow_html=True)
                            st.rerun()
                    else:
                        st.error(f"❌ CSV file must contain columns: {', '.join(required_columns)}")
                        
            except Exception as e:
                st.error(f"❌ Error importing file: {str(e)}")

if __name__ == "__main__":
    main(), 'kr', 'zł', 'Kč', 'Ft']:
                        value_str = value_str.replace(symbol_char, '')
                    value_str = value_str.replace(',', '')
                    current_values.append(float(value_str))
                    symbols.append(item['Symbol'])
                except:
                    continue
            
            if current_values and symbols:
                chart_data = pd.DataFrame({
                    'Symbol': symbols,
                    'Current Value': current_values
                })
                st.bar_chart(chart_data.set_index('Symbol'))"{return_pct:+.1f}%",
                'P&L': f"${current_value - invested_value:+,.2f}",
                'Asset Type': data['asset_type'],
                'Weight %': f"{(current_value / metrics['total_current_value']) * 100:.1f}%" if metrics['total_current_value'] > 0 else "0.0%"
            })
    
    if portfolio_data:
        st.markdown("### 📊 Holdings Breakdown")
        df = pd.DataFrame(portfolio_data)
        st.dataframe(df, use_container_width=True, height=400)
        
        if PLOTLY_AVAILABLE:
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                st.markdown("#### 🥧 Portfolio Allocation")
                # Extract numeric values safely
                current_values = []
                symbols = []
                
                for item in portfolio_data:
                    try:
                        value_str = item['Current Value'].replace('$', '').replace(',', '')
                        current_values.append(float(value_str))
                        symbols.append(item['Symbol'])
                    except:
                        continue
                
                if current_values:
                    fig_pie = px.pie(
                        values=current_values,
                        names=symbols,
                        title="Current Holdings Distribution",
                        hole=0.4
                    )
                    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                    fig_pie.update_layout(height=400)
                    st.plotly_chart(fig_pie, use_container_width=True)
            
            with col_chart2:
                st.markdown("#### 📈 Performance by Asset")
                return_values = []
                
                for item in portfolio_data:
                    try:
                        return_str = item['Return %'].replace('%', '').replace('+', '')
                        return_values.append(float(return_str))
                    except:
                        return_values.append(0.0)
                
                if return_values and symbols:
                    fig_returns = px.bar(
                        x=symbols,
                        y=return_values,
                        title="Return % by Holding",
                        color=return_values,
                        color_continuous_scale=['red', 'yellow', 'green']
                    )
                    fig_returns.update_layout(height=400, xaxis_title="Assets", yaxis_title="Return %")
                    st.plotly_chart(fig_returns, use_container_width=True)
        else:
            st.markdown("#### 📊 Simple Visualization")
            current_values = []
            symbols = []
            
            for item in portfolio_data:
                try:
                    value_str = item['Current Value'].replace('$', '').replace(',', '')
                    current_values.append(float(value_str))
                    symbols.append(item['Symbol'])
                except:
                    continue
            
            if current_values and symbols:
                chart_data = pd.DataFrame({
                    'Symbol': symbols,
                    'Current Value': current_values
                })
                st.bar_chart(chart_data.set_index('Symbol'))

def show_asset_management():
    """Display asset management page"""
    
    st.markdown("### 🎯 Manage Portfolio Assets")
    
    if st.session_state.learning_mode:
        st.markdown("""
        <div class="info-card">
            <strong>🎯 Asset Management Guide:</strong><br>
            Use this section to build and maintain your investment portfolio. Add assets, track purchase prices, and manage your holdings.
        </div>
        """, unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["➕ Add Assets", "➖ Remove Assets"])
    
    with tab1:
        st.markdown("### Add New Investment")
        
        col1, col2 = st.columns(2)
        
        with col1:
            popular_assets = get_popular_assets()
            
            # Create categories for better organization
            asset_categories = {
                "🏢 Large Cap Stocks": [k for k in popular_assets.keys() if k in ["AAPL", "GOOGL", "GOOG", "MSFT", "AMZN", "TSLA", "NVDA", "META", "NFLX", "JPM", "JNJ", "V", "WMT", "PG", "UNH", "HD", "MA", "BAC", "ADBE", "CRM", "XOM", "CVX", "KO", "PFE", "INTC", "CSCO", "VZ", "MRK", "ABT", "TMO"]],
                "🚀 Growth & Tech": [k for k in popular_assets.keys() if k in ["AMD", "PYPL", "SHOP", "SQ", "ROKU", "ZM", "SNOW", "PLTR", "CRWD", "NET"]],
                "🏛️ Vanguard ETFs": [k for k in popular_assets.keys() if k.startswith("V") and k in ["VTI", "VEA", "VWO", "VTV", "VUG", "VIG", "VYM", "VXUS", "VOO", "VT", "VB", "VO", "VGT", "VHT", "VFH", "VDE", "VAW", "VIS", "VCR", "VDC", "VPU", "VNQ"]],
                "⚫ BlackRock iShares": [k for k in popular_assets.keys() if k in ["QQQ", "IWM", "EFA", "EEM", "IVV", "IEFA", "IEMG", "IJH", "IJR", "IVW", "IVE", "IWF", "IWD", "ITOT", "IXUS", "IYY", "IWB", "IWV", "ACWI", "ACWX"]],
                "🎯 Sector ETFs": [k for k in popular_assets.keys() if k.startswith("XL") or k in ["IBB", "SOXX", "SKYY", "HACK", "ROBO", "ARKK", "ARKQ", "ARKW", "ARKG", "ARKF"]],
                "🏦 Bonds & Fixed Income": [k for k in popular_assets.keys() if k in ["AGG", "BND", "TLT", "IEF", "SHY", "LQD", "HYG", "JNK", "TIP", "VTEB", "MUB", "EMB", "BNDX", "VGIT", "VGSH", "VGLT", "VCIT", "VCSH", "BSV", "BIV", "BLV"]],
                "💎 Fidelity Funds": [k for k in popular_assets.keys() if k.startswith("F") and len(k) > 2],
                "🥇 Commodities": [k for k in popular_assets.keys() if k in ["GLD", "SLV", "USO", "UNG", "DBA", "DBC", "PDBC", "GSG", "COMT", "IAU", "SGOL", "PPLT", "PALL", "CORN", "WEAT", "SOYB", "CANE", "COW"]],
                "🌍 International": [k for k in popular_assets.keys() if k.startswith("EW") or k in ["FXI", "INDA", "RSX", "ASHR", "MCHI", "KWEB", "YINN"]],
                "₿ Major Cryptocurrencies": [k for k in popular_assets.keys() if k.endswith("-USD") and k in ["BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "ADA-USD", "SOL-USD", "DOGE-USD", "DOT-USD", "MATIC-USD", "SHIB-USD", "LTC-USD", "TRX-USD", "AVAX-USD", "LINK-USD", "ATOM-USD"]],
                "🔗 DeFi & Web3": [k for k in popular_assets.keys() if k.endswith("-USD") and k in ["XLM-USD", "ALGO-USD", "VET-USD", "ICP-USD", "FIL-USD", "SAND-USD", "MANA-USD", "AXS-USD", "ENJ-USD", "CHZ-USD", "FLOW-USD", "THETA-USD", "XTZ-USD", "EGLD-USD", "NEAR-USD"]],
                "📈 Crypto ETFs": [k for k in popular_assets.keys() if k in ["BITO", "BITI", "GBTC", "ETHE", "GDLC", "BITQ", "BLOK", "LEGR"]],
                "💱 Currency ETFs": [k for k in popular_assets.keys() if k.startswith("FX") or k in ["UUP", "UDN"]],
                "💰 Dividend ETFs": [k for k in popular_assets.keys() if k in ["SCHD", "DVY", "VYM", "NOBL", "DGRO", "HDV", "SPHD", "PEY", "RDVY", "FDL"]],
                "🌱 ESG & Clean Energy": [k for k in popular_assets.keys() if k in ["ESG", "ESGU", "ESGD", "ESGE", "VSGX", "ESGV", "SUSA", "DSI", "ICLN", "PBW", "QCLN"]],
                "📊 Volatility & Leverage": [k for k in popular_assets.keys() if k in ["VIX", "UVXY", "SVXY", "VXX", "VIXY", "TQQQ", "SQQQ", "UPRO", "SPXU", "TNA", "TZA", "TECL", "TECS", "FAS", "FAZ"]]
            }
            
            selected_category = st.selectbox(
                "Choose Asset Category",
                [""] + list(asset_categories.keys()),
                help="Browse assets by category for easier selection"
            )
            
            if selected_category:
                available_assets = asset_categories[selected_category]
                selected_popular = st.selectbox(
                    f"Assets in {selected_category}",
                    [""] + available_assets,
                    format_func=lambda x: f"{x} - {popular_assets[x]}" if x else "",
                    help="Select from assets in this category"
                )
            else:
                selected_popular = st.selectbox(
                    "Or search all assets",
                    [""] + list(popular_assets.keys()),
                    format_func=lambda x: f"{x} - {popular_assets[x]}" if x else "",
                    help="Search through all available assets"
                )
            
            custom_symbol = st.text_input(
                "Or Enter Custom Symbol",
                placeholder="e.g., AAPL, BTC-USD, GLD.L (London)",
                help="Enter any valid symbol including international exchanges"
            )
            
            symbol_to_use = ""
            if selected_popular:
                symbol_to_use = selected_popular
            elif custom_symbol:
                symbol_to_use = custom_symbol.upper()
        
        with col2:
            shares = st.number_input(
                "Number of Shares/Units",
                min_value=0.001,
                value=1.0,
                step=0.1,
                help="Enter the quantity you own"
            )
            
            purchase_price = st.number_input(
                "Purchase Price per Share ($)",
                min_value=0.01,
                value=100.0,
                step=0.01,
                help="Enter the average price you paid per share"
            )
            
            # NEW: Currency selection
            currencies = get_currency_list()
            selected_currency = st.selectbox(
                "Currency",
                list(currencies.keys()),
                index=0,  # Default to USD
                format_func=lambda x: f"{x} - {currencies[x]['name']}",
                help="Select the currency for this asset"
            )
            
            asset_type = st.selectbox(
                "Asset Type",
                ["Stock", "ETF", "Cryptocurrency", "Bond", "Commodity", "Index Fund", "Mutual Fund", "REIT", "Other"]
            )
            
            if st.session_state.learning_mode:
                st.markdown("""
                <div class="info-card">
                    <strong>💡 Pro Tip:</strong><br>
                    • Select the correct currency for international assets<br>
                    • Purchase price should be in the selected currency<br>
                    • The system will convert to your base currency for analysis
                </div>
                """, unsafe_allow_html=True)
        
        if symbol_to_use:
            asset_info = fetch_asset_data(symbol_to_use)
            if asset_info:
                current_price = asset_info['current_price']
                
                # Convert prices for display if different currency
                if selected_currency != "USD":
                    exchange_rate = get_exchange_rate("USD", selected_currency)
                    current_price_display = current_price * exchange_rate
                    purchase_price_display = purchase_price
                else:
                    current_price_display = current_price
                    purchase_price_display = purchase_price
                
                potential_return = ((current_price_display - purchase_price_display) / purchase_price_display) * 100 if purchase_price_display > 0 else 0
                
                # Create three columns for metrics display
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                
                with metric_col1:
                    currency_symbol = currencies[selected_currency]["symbol"]
                    st.metric("Current Price", f"{currency_symbol}{current_price_display:.2f}")
                with metric_col2:
                    st.metric("Your Purchase Price", f"{currency_symbol}{purchase_price_display:.2f}")
                with metric_col3:
                    st.metric("Potential Return", f"{potential_return:+.1f}%", 
                             delta=f"{currency_symbol}{(current_price_display - purchase_price_display) * shares:+.2f}")
                
                if st.button("✅ Add to Portfolio", type="primary", use_container_width=True):
                    st.session_state.portfolio[symbol_to_use] = {
                        'shares': shares,
                        'asset_type': asset_type,
                        'purchase_price': purchase_price,
                        'currency': selected_currency,  # NEW: Store currency
                        'added_date': datetime.now().isoformat()
                    }
                    
                    portfolios = load_portfolios()
                    portfolios[st.session_state.username] = st.session_state.portfolio
                    save_portfolios(portfolios)
                    
                    st.markdown("""
                    <div class="success-card">
                        <strong>🎉 Asset Added Successfully!</strong><br>
                        Your portfolio metrics are being recalculated with the new asset allocation and currency conversion.
                    </div>
                    """, unsafe_allow_html=True)
                    st.rerun()
            else:
                st.error(f"❌ Could not find asset data for '{symbol_to_use}'. Please check the symbol.")
    
    with tab2:
        st.markdown("### Remove Assets from Portfolio")
        
        if st.session_state.portfolio:
            current_values = []
            for symbol, data in st.session_state.portfolio.items():
                asset_info = fetch_asset_data(symbol)
                if asset_info:
                    current_value = data['shares'] * asset_info['current_price']
                    current_values.append({
                        'Symbol': symbol,
                        'Shares': f"{data['shares']:.3f}",
                        'Current Value': f"${current_value:,.2f}",
                        'Asset Type': data['asset_type']
                    })
            
            if current_values:
                df_current = pd.DataFrame(current_values)
                st.dataframe(df_current, hide_index=True)
            
            assets_to_remove = st.multiselect(
                "Select assets to remove:",
                list(st.session_state.portfolio.keys()),
                help="Choose one or more assets to remove from your portfolio"
            )
            
            if assets_to_remove:
                st.warning(f"⚠️ You are about to remove {len(assets_to_remove)} asset(s) from your portfolio.")
                
                if st.button("🗑️ Remove Selected Assets", type="secondary"):
                    for asset in assets_to_remove:
                        del st.session_state.portfolio[asset]
                    
                    portfolios = load_portfolios()
                    portfolios[st.session_state.username] = st.session_state.portfolio
                    save_portfolios(portfolios)
                    
                    st.markdown("""
                    <div class="success-card">
                        <strong>✅ Assets Removed Successfully!</strong><br>
                        Your portfolio has been updated and metrics recalculated.
                    </div>
                    """, unsafe_allow_html=True)
                    st.rerun()
        else:
            st.markdown("""
            <div class="info-card">
                <strong>📝 No assets to remove</strong><br>
                Your portfolio is currently empty. Add some investments first!
            </div>
            """, unsafe_allow_html=True)

def show_analytics_dashboard():
    """Display analytics dashboard"""
    
    st.markdown("### 📊 Advanced Analytics Dashboard")
    
    if st.session_state.learning_mode:
        st.markdown("""
        <div class="info-card">
            <strong>📈 Analytics Dashboard Guide:</strong><br>
            This dashboard provides professional-grade analysis tools including performance metrics, technical analysis, and AI-powered insights.
        </div>
        """, unsafe_allow_html=True)
    
    if not st.session_state.portfolio:
        st.warning("⚠️ Add assets to your portfolio to see advanced analytics.")
        return
    
    metrics = calculate_portfolio_metrics_advanced(st.session_state.portfolio)
    
    st.markdown("### 📈 Advanced Performance Metrics")
    
    perf_col1, perf_col2, perf_col3, perf_col4, perf_col5 = st.columns(5)
    
    with perf_col1:
        st.metric(
            "Annualized Return", 
            f"{metrics['annualized_return']:+.1f}%",
            help="Expected annual return based on current performance"
        )
    
    with perf_col2:
        st.metric(
            "Portfolio Beta", 
            f"{metrics['beta']:.2f}",
            delta="Market Risk" if metrics['beta'] > 1 else "Lower Risk",
            help="Portfolio volatility relative to market"
        )
    
    with perf_col3:
        st.metric(
            "Sharpe Ratio", 
            f"{metrics['sharpe_ratio']:.2f}",
            delta="Excellent" if metrics['sharpe_ratio'] > 1 else "Good" if metrics['sharpe_ratio'] > 0.5 else "Needs Improvement",
            help="Risk-adjusted return measure"
        )
    
    with perf_col4:
        st.metric(
            "Max Drawdown", 
            f"{metrics['max_drawdown']:.1f}%",
            delta="Risk Level",
            delta_color="inverse",
            help="Largest peak-to-trough decline"
        )
    
    with perf_col5:
        st.metric(
            "Portfolio Volatility", 
            f"{metrics['volatility']:.1f}%",
            help="Measure of price fluctuation"
        )
    
    # Technical analysis for top holdings
    st.markdown("### 🔍 Technical Analysis - Top Holdings")
    
    portfolio_values = []
    for symbol, data in st.session_state.portfolio.items():
        asset_info = fetch_asset_data(symbol)
        if asset_info:
            value = data['shares'] * asset_info['current_price']
            portfolio_values.append((symbol, value))
    
    portfolio_values.sort(key=lambda x: x[1], reverse=True)
    top_holdings = [item[0] for item in portfolio_values[:3]]
    
    for i, symbol in enumerate(top_holdings):
        with st.expander(f"📊 {symbol} - Technical Analysis", expanded=(i == 0)):
            indicators = calculate_technical_indicators(symbol)
            
            if indicators is not None:
                tech_col1, tech_col2, tech_col3, tech_col4 = st.columns(4)
                
                current_price = indicators['Close'].iloc[-1]
                ma20 = indicators['MA_20'].iloc[-1]
                current_rsi = indicators['RSI'].iloc[-1]
                current_macd = indicators['MACD'].iloc[-1]
                current_signal = indicators['MACD_Signal'].iloc[-1]
                
                with tech_col1:
                    price_trend = "Bullish" if current_price > ma20 else "Bearish"
                    st.metric("Price Trend", price_trend, f"${current_price:.2f}")
                
                with tech_col2:
                    rsi_status = "Overbought" if current_rsi > 70 else "Oversold" if current_rsi < 30 else "Neutral"
                    rsi_color = "🔴" if current_rsi > 70 else "🟢" if current_rsi < 30 else "🟡"
                    st.metric("RSI Signal", f"{rsi_color} {rsi_status}", f"{current_rsi:.1f}")
                
                with tech_col3:
                    macd_trend = "Bullish" if current_macd > current_signal else "Bearish"
                    macd_color = "🟢" if current_macd > current_signal else "🔴"
                    st.metric("MACD Signal", f"{macd_color} {macd_trend}")
                
                with tech_col4:
                    support_level = current_price * 0.95
                    resistance_level = current_price * 1.05
                    st.metric("Support/Resistance", f"${support_level:.2f} / ${resistance_level:.2f}")
                
                if PLOTLY_AVAILABLE:
                    fig = make_subplots(
                        rows=3, cols=1,
                        subplot_titles=(
                            f'{symbol} - Price & Moving Averages',
                            'Relative Strength Index (RSI)',
                            'MACD & Signal Line'
                        ),
                        vertical_spacing=0.08,
                        row_heights=[0.5, 0.25, 0.25]
                    )
                    
                    fig.add_trace(
                        go.Scatter(x=indicators.index, y=indicators['Close'], name='Price', line=dict(color='blue', width=2)),
                        row=1, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=indicators.index, y=indicators['MA_20'], name='MA 20', line=dict(color='orange')),
                        row=1, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=indicators.index, y=indicators['MA_50'], name='MA 50', line=dict(color='red')),
                        row=1, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(x=indicators.index, y=indicators['RSI'], name='RSI', line=dict(color='purple')),
                        row=2, col=1
                    )
                    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
                    
                    fig.add_trace(
                        go.Scatter(x=indicators.index, y=indicators['MACD'], name='MACD', line=dict(color='blue')),
                        row=3, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=indicators.index, y=indicators['MACD_Signal'], name='Signal', line=dict(color='red')),
                        row=3, col=1
                    )
                    
                    fig.update_layout(height=700, showlegend=True, title_text=f"{symbol} Technical Analysis")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.markdown("#### 📈 Price Chart")
                    st.line_chart(indicators[['Close', 'MA_20', 'MA_50']])
                    
                    simple_col1, simple_col2 = st.columns(2)
                    with simple_col1:
                        st.markdown("#### RSI")
                        st.line_chart(indicators['RSI'])
                    with simple_col2:
                        st.markdown("#### MACD")
                        st.line_chart(indicators[['MACD', 'MACD_Signal']])
    
    # Investment suggestions
    st.markdown("### 🤖 AI-Powered Investment Insights")
    suggestions = generate_investment_suggestions(st.session_state.portfolio)
    
    enhanced_suggestions = []
    
    if metrics['total_return'] > 10:
        enhanced_suggestions.append({
            'type': 'success',
            'message': f'🎉 Excellent performance! Your portfolio is up {metrics["total_return"]:.1f}%. Consider taking some profits and rebalancing.'
        })
    elif metrics['total_return'] < -5:
        enhanced_suggestions.append({
            'type': 'warning',
            'message': f'📉 Portfolio is down {abs(metrics["total_return"]):.1f}%. Review underperforming assets and consider strategic rebalancing.'
        })
    
    if metrics['volatility'] > 25:
        enhanced_suggestions.append({
            'type': 'warning',
            'message': f'⚡ High volatility detected ({metrics["volatility"]:.1f}%). Consider adding stable assets like bonds or dividend stocks.'
        })
    
    all_suggestions = enhanced_suggestions + suggestions
    
    for suggestion in all_suggestions[:6]:
        if suggestion['type'] == 'diversification':
            st.markdown(f"""
            <div class="info-card">
                <strong>🎯 Diversification:</strong> {suggestion['message']}
            </div>
            """, unsafe_allow_html=True)
        elif suggestion['type'] == 'rebalancing':
            st.markdown(f"""
            <div class="warning-card">
                <strong>⚖️ Rebalancing:</strong> {suggestion['message']}
            </div>
            """, unsafe_allow_html=True)
        elif suggestion['type'] == 'opportunity':
            st.markdown(f"""
            <div class="success-card">
                <strong>🚀 Opportunity:</strong> {suggestion['message']}
            </div>
            """, unsafe_allow_html=True)
        elif suggestion['type'] == 'success':
            st.markdown(f"""
            <div class="success-card">
                <strong>✅ Success:</strong> {suggestion['message']}
            </div>
            """, unsafe_allow_html=True)
        elif suggestion['type'] == 'warning':
            st.markdown(f"""
            <div class="warning-card">
                <strong>⚠️ Warning:</strong> {suggestion['message']}
            </div>
            """, unsafe_allow_html=True)

def show_export_import():
    """Display export/import functionality"""
    
    st.markdown("### 📁 Export & Import Portfolio")
    
    if st.session_state.learning_mode:
        st.markdown("""
        <div class="info-card">
            <strong>📁 Export/Import Guide:</strong><br>
            Backup and restore your portfolio data. Export as JSON or CSV for safekeeping, or import previously saved portfolios.
        </div>
        """, unsafe_allow_html=True)
    
    export_tab, import_tab = st.tabs(["📤 Export Portfolio", "📥 Import Portfolio"])
    
    with export_tab:
        st.markdown("### Export Your Portfolio Data")
        
        if not st.session_state.portfolio:
            st.markdown("""
            <div class="warning-card">
                <strong>📝 No portfolio data to export</strong><br>
                Add some investments to your portfolio first, then return here to export your data.
            </div>
            """, unsafe_allow_html=True)
            return
        
        export_format = st.selectbox("Choose Export Format", ["JSON", "CSV"])
        
        if export_format == "JSON":
            export_data = {
                'username': st.session_state.username,
                'export_date': datetime.now().isoformat(),
                'portfolio': st.session_state.portfolio,
                'version': '2.0'
            }
            
            json_string = json.dumps(export_data, indent=2)
            
            st.download_button(
                label="📥 Download Portfolio (JSON)",
                data=json_string,
                file_name=f"portfolio_{st.session_state.username}_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )
            
            with st.expander("👀 Preview JSON Data"):
                st.code(json_string, language="json")
        
        else:  # CSV format
            csv_data = []
            for symbol, data in st.session_state.portfolio.items():
                asset_info = fetch_asset_data(symbol)
                if asset_info:
                    csv_data.append({
                        'Symbol': symbol,
                        'Name': asset_info['name'],
                        'Shares': data['shares'],
                        'Asset_Type': data['asset_type'],
                        'Purchase_Price': data.get('purchase_price', asset_info['current_price']),
                        'Current_Price': asset_info['current_price'],
                        'Current_Value': data['shares'] * asset_info['current_price'],
                        'Added_Date': data['added_date']
                    })
            
            if csv_data:
                df = pd.DataFrame(csv_data)
                csv_string = df.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download Portfolio (CSV)",
                    data=csv_string,
                    file_name=f"portfolio_{st.session_state.username}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
                
                with st.expander("👀 Preview CSV Data"):
                    st.dataframe(df)
    
    with import_tab:
        st.markdown("### Import Portfolio Data")
        
        uploaded_file = st.file_uploader(
            "Choose a portfolio file",
            type=['json', 'csv'],
            help="Upload a previously exported portfolio file"
        )
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.json'):
                    content = uploaded_file.read()
                    data = json.loads(content)
                    
                    if 'portfolio' in data:
                        st.markdown("""
                        <div class="success-card">
                            <strong>✅ Valid portfolio file detected!</strong><br>
                            Ready to import your portfolio data.
                        </div>
                        """, unsafe_allow_html=True)
                        
                        with st.expander("👀 Preview Import Data"):
                            st.json(data['portfolio'])
                        
                        import_option = st.selectbox(
                            "Import Option",
                            ["Replace Current Portfolio", "Merge with Current Portfolio"]
                        )
                        
                        if st.button("🚀 Import Portfolio", type="primary"):
                            if import_option == "Replace Current Portfolio":
                                st.session_state.portfolio = data['portfolio']
                            else:
                                for symbol, asset_data in data['portfolio'].items():
                                    st.session_state.portfolio[symbol] = asset_data
                            
                            portfolios = load_portfolios()
                            portfolios[st.session_state.username] = st.session_state.portfolio
                            save_portfolios(portfolios)
                            
                            st.markdown("""
                            <div class="success-card">
                                <strong>🎉 Portfolio imported successfully!</strong><br>
                                Your portfolio has been updated with the imported data.
                            </div>
                            """, unsafe_allow_html=True)
                            st.rerun()
                    else:
                        st.error("❌ Invalid portfolio file format.")
                
                elif uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                    
                    required_columns = ['Symbol', 'Shares', 'Asset_Type']
                    if all(col in df.columns for col in required_columns):
                        st.markdown("""
                        <div class="success-card">
                            <strong>✅ Valid CSV file detected!</strong><br>
                            Ready to import your portfolio data.
                        </div>
                        """, unsafe_allow_html=True)
                        
                        with st.expander("👀 Preview Import Data"):
                            st.dataframe(df)
                        
                        csv_import_option = st.selectbox(
                            "Import Option",
                            ["Replace Current Portfolio", "Merge with Current Portfolio"],
                            key="csv_import_option"
                        )
                        
                        if st.button("🚀 Import CSV Portfolio", type="primary"):
                            new_portfolio = {}
                            
                            for _, row in df.iterrows():
                                symbol = row['Symbol']
                                new_portfolio[symbol] = {
                                    'shares': float(row['Shares']),
                                    'asset_type': row['Asset_Type'],
                                    'purchase_price': float(row.get('Purchase_Price', 100.0)),
                                    'added_date': datetime.now().isoformat()
                                }
                            
                            if csv_import_option == "Replace Current Portfolio":
                                st.session_state.portfolio = new_portfolio
                            else:
                                for symbol, asset_data in new_portfolio.items():
                                    st.session_state.portfolio[symbol] = asset_data
                            
                            portfolios = load_portfolios()
                            portfolios[st.session_state.username] = st.session_state.portfolio
                            save_portfolios(portfolios)
                            
                            st.markdown("""
                            <div class="success-card">
                                <strong>🎉 CSV portfolio imported successfully!</strong><br>
                                Your portfolio has been updated with the imported data.
                            </div>
                            """, unsafe_allow_html=True)
                            st.rerun()
                    else:
                        st.error(f"❌ CSV file must contain columns: {', '.join(required_columns)}")
                        
            except Exception as e:
                st.error(f"❌ Error importing file: {str(e)}")

if __name__ == "__main__":
    main()
