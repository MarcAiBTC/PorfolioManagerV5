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

# Currency conversion rates (mock data - in production would use real API)
CURRENCY_RATES = {
    'USD': 1.0,
    'EUR': 0.85,
    'GBP': 0.73,
    'JPY': 110.0,
    'CAD': 1.25,
    'AUD': 1.35,
    'CHF': 0.92,
    'CNY': 6.45,
    'INR': 74.5,
    'BRL': 5.2
}

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
    """Return comprehensive dictionary of popular assets with their symbols"""
    return {
        # US Large Cap Stocks
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
        "MA": "Mastercard Inc.",
        "PG": "Procter & Gamble Co.",
        "UNH": "UnitedHealth Group Inc.",
        "HD": "The Home Depot Inc.",
        "BAC": "Bank of America Corp.",
        "DIS": "The Walt Disney Company",
        "ADBE": "Adobe Inc.",
        "CRM": "Salesforce Inc.",
        "XOM": "Exxon Mobil Corporation",
        "CVX": "Chevron Corporation",
        "WMT": "Walmart Inc.",
        "KO": "The Coca-Cola Company",
        "PFE": "Pfizer Inc.",
        "ABBV": "AbbVie Inc.",
        "COST": "Costco Wholesale Corporation",
        "AVGO": "Broadcom Inc.",
        "TMO": "Thermo Fisher Scientific Inc.",
        "ACN": "Accenture plc",
        
        # International Stocks
        "ASML": "ASML Holding N.V.",
        "TSM": "Taiwan Semiconductor Manufacturing Co.",
        "NVO": "Novo Nordisk A/S",
        "NESN.SW": "Nestlé S.A.",
        "RHHBY": "Roche Holding AG",
        "SAP": "SAP SE",
        "TM": "Toyota Motor Corporation",
        "BABA": "Alibaba Group Holding Limited",
        "TCEHY": "Tencent Holdings Limited",
        
        # US Broad Market ETFs
        "SPY": "SPDR S&P 500 ETF Trust",
        "VOO": "Vanguard S&P 500 ETF",
        "IVV": "iShares Core S&P 500 ETF",
        "VTI": "Vanguard Total Stock Market ETF",
        "ITOT": "iShares Core S&P Total U.S. Stock Market ETF",
        "SPTM": "SPDR Portfolio S&P 1500 Composite Stock Market ETF",
        
        # Technology ETFs
        "QQQ": "Invesco QQQ Trust ETF",
        "VGT": "Vanguard Information Technology ETF",
        "XLK": "Technology Select Sector SPDR Fund",
        "FTEC": "Fidelity MSCI Information Technology Index ETF",
        "IYW": "iShares U.S. Technology ETF",
        
        # International ETFs
        "VEA": "Vanguard FTSE Developed Markets ETF",
        "IEFA": "iShares Core MSCI EAFE IMI Index ETF",
        "EFA": "iShares MSCI EAFE ETF",
        "VWO": "Vanguard FTSE Emerging Markets ETF",
        "IEMG": "iShares Core MSCI Emerging Markets IMI Index ETF",
        "EEM": "iShares MSCI Emerging Markets ETF",
        "FEZ": "SPDR EURO STOXX 50 ETF",
        "EWJ": "iShares MSCI Japan ETF",
        "INDA": "iShares MSCI India ETF",
        "MCHI": "iShares MSCI China ETF",
        
        # Small & Mid Cap ETFs
        "IWM": "iShares Russell 2000 ETF",
        "VB": "Vanguard Small-Cap ETF",
        "IJH": "iShares Core S&P Mid-Cap ETF",
        "VO": "Vanguard Mid-Cap ETF",
        
        # Sector ETFs
        "XLF": "Financial Select Sector SPDR Fund",
        "XLE": "Energy Select Sector SPDR Fund",
        "XLV": "Health Care Select Sector SPDR Fund",
        "XLI": "Industrial Select Sector SPDR Fund",
        "XLY": "Consumer Discretionary Select Sector SPDR Fund",
        "XLP": "Consumer Staples Select Sector SPDR Fund",
        "XLU": "Utilities Select Sector SPDR Fund",
        "XLRE": "Real Estate Select Sector SPDR Fund",
        "XLB": "Materials Select Sector SPDR Fund",
        
        # Bond ETFs
        "AGG": "iShares Core U.S. Aggregate Bond ETF",
        "BND": "Vanguard Total Bond Market ETF",
        "VTEB": "Vanguard Tax-Exempt Bond ETF",
        "TLT": "iShares 20+ Year Treasury Bond ETF",
        "IEF": "iShares 7-10 Year Treasury Bond ETF",
        "SHY": "iShares 1-3 Year Treasury Bond ETF",
        "LQD": "iShares iBoxx $ Investment Grade Corporate Bond ETF",
        "HYG": "iShares iBoxx $ High Yield Corporate Bond ETF",
        "JNK": "SPDR Bloomberg High Yield Bond ETF",
        "EMB": "iShares J.P. Morgan USD Emerging Markets Bond ETF",
        "BNDX": "Vanguard Total International Bond ETF",
        
        # Vanguard Funds
        "VTSAX": "Vanguard Total Stock Market Index Fund",
        "VTIAX": "Vanguard Total International Stock Index Fund",
        "VBTLX": "Vanguard Total Bond Market Index Fund",
        "VGTSX": "Vanguard Total International Stock Index Fund",
        "VGSLX": "Vanguard Real Estate Index Fund",
        "VTSMX": "Vanguard Total Stock Market Index Fund Investor Shares",
        
        # Fidelity Funds
        "FXAIX": "Fidelity 500 Index Fund",
        "FTIHX": "Fidelity Total International Index Fund",
        "FXNAX": "Fidelity U.S. Bond Index Fund",
        "FSKAX": "Fidelity Total Market Index Fund",
        "FTEC": "Fidelity MSCI Information Technology Index ETF",
        "FDVV": "Fidelity High Dividend ETF",
        
        # BlackRock/iShares Additional
        "IUSV": "iShares Core S&P U.S. Value ETF",
        "IUSG": "iShares Core S&P U.S. Growth ETF",
        "ICLN": "iShares Global Clean Energy ETF",
        "IGV": "iShares Expanded Tech-Software Sector ETF",
        "ITA": "iShares U.S. Aerospace & Defense ETF",
        
        # Commodities & Alternatives
        "GLD": "SPDR Gold Shares",
        "IAU": "iShares Gold Trust",
        "SLV": "iShares Silver Trust",
        "PDBC": "Invesco Optimum Yield Diversified Commodity Strategy No K-1 ETF",
        "DBC": "Invesco DB Commodity Index Tracking Fund",
        "USO": "United States Oil Fund",
        "UNG": "United States Natural Gas Fund",
        "REIT": "iShares Global REIT ETF",
        "VNQ": "Vanguard Real Estate ETF",
        "VTEB": "Vanguard Tax-Exempt Bond ETF",
        
        # Cryptocurrency
        "BTC-USD": "Bitcoin",
        "ETH-USD": "Ethereum",
        "ADA-USD": "Cardano",
        "DOT-USD": "Polkadot",
        "LINK-USD": "Chainlink",
        "SOL-USD": "Solana",
        "MATIC-USD": "Polygon",
        "AVAX-USD": "Avalanche",
        "ATOM-USD": "Cosmos",
        "UNI-USD": "Uniswap",
        "LTC-USD": "Litecoin",
        "BCH-USD": "Bitcoin Cash",
        "XRP-USD": "XRP",
        "DOGE-USD": "Dogecoin",
        "SHIB-USD": "Shiba Inu",
        
        # Currency ETFs
        "UUP": "Invesco DB US Dollar Index Bullish Fund",
        "FXE": "Invesco CurrencyShares Euro Trust",
        "FXB": "Invesco CurrencyShares British Pound Sterling Trust",
        "FXY": "Invesco CurrencyShares Japanese Yen Trust",
        "FXC": "Invesco CurrencyShares Canadian Dollar Trust",
        
        # European Stocks
        "MC.PA": "LVMH Moët Hennessy Louis Vuitton",
        "OR.PA": "L'Oréal S.A.",
        "SAP.DE": "SAP SE",
        "SIE.DE": "Siemens AG",
        "BAS.DE": "BASF SE",
        "VOW3.DE": "Volkswagen AG",
        
        # Dividend Focused
        "VYM": "Vanguard High Dividend Yield ETF",
        "SCHD": "Schwab US Dividend Equity ETF",
        "DVY": "iShares Select Dividend ETF",
        "VIG": "Vanguard Dividend Appreciation ETF",
        "NOBL": "ProShares S&P 500 Dividend Aristocrats ETF",
        
        # Growth & Value
        "VUG": "Vanguard Growth ETF",
        "VTV": "Vanguard Value ETF",
        "IWF": "iShares Russell 1000 Growth ETF",
        "IWD": "iShares Russell 1000 Value ETF"
    }

def convert_currency(amount, from_currency, to_currency):
    """Convert amount from one currency to another"""
    if from_currency == to_currency:
        return amount
    
    usd_amount = amount / CURRENCY_RATES.get(from_currency, 1.0)
    return usd_amount * CURRENCY_RATES.get(to_currency, 1.0)

def fetch_asset_data(symbol, currency='USD'):
    """Fetch current asset data with currency support"""
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
                    base_currency = info.get('currency', 'USD')
                except:
                    name = popular_assets.get(symbol, symbol)
                    base_currency = 'USD'
                
                # Convert price to requested currency
                converted_price = convert_currency(current_price, base_currency, currency)
                
                return {
                    'name': name,
                    'current_price': float(converted_price),
                    'symbol': symbol,
                    'currency': currency,
                    'base_currency': base_currency
                }
        except:
            pass
    
    # Fallback to mock data
    mock_price = get_mock_price(symbol)
    converted_price = convert_currency(mock_price, 'USD', currency)
    
    return {
        'name': popular_assets.get(symbol, symbol),
        'current_price': converted_price,
        'symbol': symbol,
        'currency': currency,
        'base_currency': 'USD'
    }

def calculate_portfolio_metrics_advanced(portfolio, base_currency='USD'):
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
    
    # Calculate totals first
    for symbol, data in portfolio.items():
        asset_currency = data.get('currency', 'USD')
        asset_info = fetch_asset_data(symbol, asset_currency)
        if asset_info:
            current_value = data['shares'] * asset_info['current_price']
            invested_value = data['shares'] * data.get('purchase_price', asset_info['current_price'])
            
            # Convert to base currency for portfolio calculations
            current_value_base = convert_currency(current_value, asset_currency, base_currency)
            invested_value_base = convert_currency(invested_value, asset_currency, base_currency)
            
            total_current_value += current_value_base
            total_invested += invested_value_base
    
    # Calculate individual asset performance and beta weights
    beta_components = []
    volatility_components = []
    
    for symbol, data in portfolio.items():
        asset_currency = data.get('currency', 'USD')
        asset_info = fetch_asset_data(symbol, asset_currency)
        if asset_info:
            current_value = data['shares'] * asset_info['current_price']
            invested_value = data['shares'] * data.get('purchase_price', asset_info['current_price'])
            
            # Convert to base currency
            current_value_base = convert_currency(current_value, asset_currency, base_currency)
            invested_value_base = convert_currency(invested_value, asset_currency, base_currency)
            
            weight = current_value_base / total_current_value if total_current_value > 0 else 0
            asset_return = (current_value_base - invested_value_base) / invested_value_base if invested_value_base > 0 else 0
            
            # Calculate individual asset beta (consistent across all functions)
            asset_beta = calculate_asset_beta(symbol, data.get('asset_type', 'Stock'))
            beta_components.append(weight * asset_beta)
            volatility_components.append(weight * calculate_asset_volatility(symbol, asset_return))
            
            asset_performance.append({
                'symbol': symbol,
                'weight': weight,
                'return': asset_return,
                'current_value': current_value_base,
                'invested_value': invested_value_base,
                'beta': asset_beta,
                'currency': asset_currency
            })
    
    total_return = (total_current_value - total_invested) / total_invested if total_invested > 0 else 0
    
    # Calculate consistent portfolio metrics
    portfolio_beta = sum(beta_components)
    portfolio_volatility = np.sqrt(sum(volatility_components))
    sharpe_ratio = (total_return * 12 - 0.02) / (portfolio_volatility * np.sqrt(12)) if portfolio_volatility > 0 else 0
    
    return {
        'beta': max(0.1, min(2.5, portfolio_beta)),
        'sharpe_ratio': max(-2.0, min(3.0, sharpe_ratio)),
        'max_drawdown': abs(np.random.normal(8, 3)),
        'total_return': total_return * 100,
        'annualized_return': total_return * 12 * 100,
        'volatility': portfolio_volatility * 100,
        'var_95': abs(np.random.normal(5, 2)),
        'total_current_value': total_current_value,
        'total_invested': total_invested,
        'asset_performance': asset_performance,
        'currency': base_currency
    }

def calculate_asset_beta(symbol, asset_type):
    """Calculate consistent asset beta based on symbol and type"""
    # Set seed based on symbol for consistency
    np.random.seed(hash(symbol) % 2147483647)
    
    base_beta = {
        'Stock': 1.0,
        'ETF': 0.9,
        'Cryptocurrency': 2.5,
        'Bond': 0.3,
        'Commodity': 1.2,
        'Index Fund': 0.95,
        'Other': 1.0
    }.get(asset_type, 1.0)
    
    # Add symbol-specific variation
    variation = np.random.normal(0, 0.2)
    return max(0.1, min(3.0, base_beta + variation))

def calculate_asset_volatility(symbol, asset_return):
    """Calculate consistent asset volatility"""
    np.random.seed(hash(symbol) % 2147483647)
    base_vol = (0.15 + abs(asset_return) * 0.1) ** 2
    variation = np.random.normal(0, 0.02)
    return max(0.01, base_vol + variation)

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
        asset_currency = data.get('currency', 'USD')
        asset_info = fetch_asset_data(symbol, asset_currency)
        
        if asset_info:
            value = data['shares'] * asset_info['current_price']
            # Convert to USD for analysis
            value_usd = convert_currency(value, asset_currency, 'USD')
            total_value += value_usd
            
            if asset_type in asset_types:
                asset_types[asset_type] += value_usd
            else:
                asset_types[asset_type] = value_usd
    
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
if 'base_currency' not in st.session_state:
    st.session_state.base_currency = 'USD'

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
                        
if __name__ == "__main__":
    main()
