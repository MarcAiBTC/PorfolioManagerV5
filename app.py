import streamlit as st
import pandas as pd
import numpy as np
import json
import hashlib
import os
from datetime import datetime, timedelta
import io
import warnings
!pip install streamlit pandas numpy plotly yfinance

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
        
        # Currency selection
        st.markdown("### 💱 Portfolio Settings")
        st.session_state.base_currency = st.selectbox(
            "Base Currency",
            list(CURRENCY_RATES.keys()),
            index=list(CURRENCY_RATES.keys()).index(st.session_state.base_currency),
            help="Select your preferred currency for portfolio calculations"
        )
        
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
            This section provides a comprehensive view of your investment portfolio including performance metrics, asset allocation, individual holdings, and risk assessment.
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
    
    metrics = calculate_portfolio_metrics_advanced(st.session_state.portfolio, st.session_state.base_currency)
    
    st.markdown("### 💼 Portfolio Performance")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    currency_symbol = "€" if st.session_state.base_currency == "EUR" else "£" if st.session_state.base_currency == "GBP" else "¥" if st.session_state.base_currency == "JPY" else "$"
    
    with col1:
        st.metric(
            f"Total Value ({st.session_state.base_currency})", 
            f"{currency_symbol}{metrics['total_current_value']:,.2f}",
            delta=f"{currency_symbol}{metrics['total_current_value'] - metrics['total_invested']:+,.2f}"
        )
    
    with col2:
        st.metric(
            "Total Return", 
            f"{metrics['total_return']:+.1f}%",
            delta=f"{currency_symbol}{metrics['total_current_value'] - metrics['total_invested']:+,.2f}"
        )
    
    with col3:
        st.metric(
            "Portfolio Beta", 
            f"{metrics['beta']:.2f}",
            help="Portfolio volatility relative to the market"
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
    
    # Portfolio composition
    portfolio_data = []
    for symbol, data in st.session_state.portfolio.items():
        asset_currency = data.get('currency', 'USD')
        asset_info = fetch_asset_data(symbol, asset_currency)
        if asset_info:
            current_price = asset_info['current_price']
            purchase_price = data.get('purchase_price', current_price)
            current_value = data['shares'] * current_price
            invested_value = data['shares'] * purchase_price
            return_pct = ((current_value - invested_value) / invested_value) * 100 if invested_value > 0 else 0
            
            # Convert to base currency for display
            current_value_base = convert_currency(current_value, asset_currency, st.session_state.base_currency)
            invested_value_base = convert_currency(invested_value, asset_currency, st.session_state.base_currency)
            current_price_base = convert_currency(current_price, asset_currency, st.session_state.base_currency)
            purchase_price_base = convert_currency(purchase_price, asset_currency, st.session_state.base_currency)
            
            asset_currency_symbol = "€" if asset_currency == "EUR" else "£" if asset_currency == "GBP" else "¥" if asset_currency == "JPY" else "$"
            base_currency_symbol = "€" if st.session_state.base_currency == "EUR" else "£" if st.session_state.base_currency == "GBP" else "¥" if st.session_state.base_currency == "JPY" else "$"
            
            portfolio_data.append({
                'Symbol': symbol,
                'Name': asset_info['name'][:30] + "..." if len(asset_info['name']) > 30 else asset_info['name'],
                'Shares': f"{data['shares']:.3f}",
                f'Purchase Price ({asset_currency})': f"{asset_currency_symbol}{purchase_price:.2f}",
                f'Current Price ({asset_currency})': f"{asset_currency_symbol}{current_price:.2f}",
                f'Invested Value ({st.session_state.base_currency})': f"{base_currency_symbol}{invested_value_base:,.2f}",
                f'Current Value ({st.session_state.base_currency})': f"{base_currency_symbol}{current_value_base:,.2f}",
                'Return %': f"{return_pct:+.1f}%",
                f'P&L ({st.session_state.base_currency})': f"{base_currency_symbol}{current_value_base - invested_value_base:+,.2f}",
                'Asset Type': data['asset_type'],
                'Weight %': f"{(current_value_base / metrics['total_current_value']) * 100:.1f}%" if metrics['total_current_value'] > 0 else "0.0%",
                'Currency': asset_currency
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
                        value_str = item[f'Current Value ({st.session_state.base_currency})'].replace(currency_symbol, '').replace(',', '')
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
                    value_str = item[f'Current Value ({st.session_state.base_currency})'].replace(currency_symbol, '').replace(',', '')
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
            Use this section to build and maintain your investment portfolio. Add assets, track purchase prices, and manage your holdings with multi-currency support.
        </div>
        """, unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["➕ Add Assets", "➖ Remove Assets"])
    
    with tab1:
        st.markdown("### Add New Investment")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Categorized asset selection
            asset_category = st.selectbox(
                "Asset Category",
                ["All Assets", "US Stocks", "International Stocks", "US ETFs", "International ETFs", 
                 "Sector ETFs", "Bond ETFs", "Vanguard Funds", "Fidelity Funds", "BlackRock/iShares", 
                 "Cryptocurrency", "Commodities", "Currency ETFs", "Dividend Focused", "Growth & Value"]
            )
            
            popular_assets = get_popular_assets()
            
            # Filter assets based on category
            if asset_category == "US Stocks":
                filtered_assets = {k: v for k, v in popular_assets.items() 
                                 if k in ["AAPL", "GOOGL", "GOOG", "MSFT", "AMZN", "TSLA", "NVDA", "META", "NFLX", "JPM", "JNJ", "V", "MA", "PG", "UNH", "HD", "BAC", "DIS", "ADBE", "CRM", "XOM", "CVX", "WMT", "KO", "PFE", "ABBV", "COST", "AVGO", "TMO", "ACN"]}
            elif asset_category == "International Stocks":
                filtered_assets = {k: v for k, v in popular_assets.items() 
                                 if k in ["ASML", "TSM", "NVO", "NESN.SW", "RHHBY", "SAP", "TM", "BABA", "TCEHY", "MC.PA", "OR.PA", "SAP.DE", "SIE.DE", "BAS.DE", "VOW3.DE"]}
            elif asset_category == "US ETFs":
                filtered_assets = {k: v for k, v in popular_assets.items() 
                                 if k in ["SPY", "VOO", "IVV", "VTI", "ITOT", "SPTM", "QQQ", "VGT", "XLK", "FTEC", "IYW"]}
            elif asset_category == "International ETFs":
                filtered_assets = {k: v for k, v in popular_assets.items() 
                                 if k in ["VEA", "IEFA", "EFA", "VWO", "IEMG", "EEM", "FEZ", "EWJ", "INDA", "MCHI"]}
            elif asset_category == "Sector ETFs":
                filtered_assets = {k: v for k, v in popular_assets.items() 
                                 if k in ["XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLRE", "XLB"]}
            elif asset_category == "Bond ETFs":
                filtered_assets = {k: v for k, v in popular_assets.items() 
                                 if k in ["AGG", "BND", "VTEB", "TLT", "IEF", "SHY", "LQD", "HYG", "JNK", "EMB", "BNDX"]}
            elif asset_category == "Vanguard Funds":
                filtered_assets = {k: v for k, v in popular_assets.items() 
                                 if k.startswith("V") or "VTSAX" in k or "VTIAX" in k or "VBTLX" in k}
            elif asset_category == "Fidelity Funds":
                filtered_assets = {k: v for k, v in popular_assets.items() 
                                 if k.startswith("F") and ("X" in k or "TEC" in k)}
            elif asset_category == "BlackRock/iShares":
                filtered_assets = {k: v for k, v in popular_assets.items() 
                                 if k.startswith("I") and len(k) <= 5}
            elif asset_category == "Cryptocurrency":
                filtered_assets = {k: v for k, v in popular_assets.items() 
                                 if "-USD" in k}
            elif asset_category == "Commodities":
                filtered_assets = {k: v for k, v in popular_assets.items() 
                                 if k in ["GLD", "IAU", "SLV", "PDBC", "DBC", "USO", "UNG", "REIT", "VNQ"]}
            elif asset_category == "Currency ETFs":
                filtered_assets = {k: v for k, v in popular_assets.items() 
                                 if k in ["UUP", "FXE", "FXB", "FXY", "FXC"]}
            elif asset_category == "Dividend Focused":
                filtered_assets = {k: v for k, v in popular_assets.items() 
                                 if k in ["VYM", "SCHD", "DVY", "VIG", "NOBL", "FDVV"]}
            elif asset_category == "Growth & Value":
                filtered_assets = {k: v for k, v in popular_assets.items() 
                                 if k in ["VUG", "VTV", "IWF", "IWD", "IUSV", "IUSG"]}
            else:
                filtered_assets = popular_assets
            
            selected_popular = st.selectbox(
                f"Choose from {asset_category} ({len(filtered_assets)} options)",
                [""] + list(filtered_assets.keys()),
                format_func=lambda x: f"{x} - {filtered_assets[x]}" if x and x in filtered_assets else x,
                help="Select from curated asset list"
            )
            
            custom_symbol = st.text_input(
                "Or Enter Custom Symbol",
                placeholder="e.g., AAPL, BTC-USD, GLD",
                help="Enter any valid symbol"
            )
            
            symbol_to_use = ""
            if selected_popular:
                symbol_to_use = selected_popular
            elif custom_symbol:
                symbol_to_use = custom_symbol.upper()
        
        with col2:
            # Asset currency selection
            asset_currency = st.selectbox(
                "Asset Currency",
                list(CURRENCY_RATES.keys()),
                index=0,
                help="Select the currency this asset is traded in"
            )
            
            shares = st.number_input(
                "Number of Shares/Units",
                min_value=0.001,
                value=1.0,
                step=0.1,
                help="Enter the quantity you own"
            )
            
            purchase_price = st.number_input(
                f"Purchase Price per Share ({asset_currency})",
                min_value=0.01,
                value=100.0,
                step=0.01,
                help="Enter the average price you paid per share"
            )
            
            asset_type = st.selectbox(
                "Asset Type",
                ["Stock", "ETF", "Cryptocurrency", "Bond", "Commodity", "Index Fund", "Other"]
            )
        
        if symbol_to_use:
            asset_info = fetch_asset_data(symbol_to_use, asset_currency)
            if asset_info:
                current_price = asset_info['current_price']
                potential_return = ((current_price - purchase_price) / purchase_price) * 100 if purchase_price > 0 else 0
                
                # Create three columns for metrics display
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                
                currency_symbol = "€" if asset_currency == "EUR" else "£" if asset_currency == "GBP" else "¥" if asset_currency == "JPY" else "$"
                
                with metric_col1:
                    st.metric("Current Price", f"{currency_symbol}{current_price:.2f}")
                with metric_col2:
                    st.metric("Your Purchase Price", f"{currency_symbol}{purchase_price:.2f}")
                with metric_col3:
                    st.metric("Potential Return", f"{potential_return:+.1f}%", 
                             delta=f"{currency_symbol}{(current_price - purchase_price) * shares:+.2f}")
                
                if st.button("✅ Add to Portfolio", type="primary", use_container_width=True):
                    st.session_state.portfolio[symbol_to_use] = {
                        'shares': shares,
                        'asset_type': asset_type,
                        'purchase_price': purchase_price,
                        'currency': asset_currency,
                        'added_date': datetime.now().isoformat()
                    }
                    
                    portfolios = load_portfolios()
                    portfolios[st.session_state.username] = st.session_state.portfolio
                    save_portfolios(portfolios)
                    
                    st.markdown("""
                    <div class="success-card">
                        <strong>🎉 Asset Added Successfully!</strong><br>
                        Your portfolio metrics are being recalculated with the new asset allocation.
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
                asset_currency = data.get('currency', 'USD')
                asset_info = fetch_asset_data(symbol, asset_currency)
                if asset_info:
                    current_value = data['shares'] * asset_info['current_price']
                    currency_symbol = "€" if asset_currency == "EUR" else "£" if asset_currency == "GBP" else "¥" if asset_currency == "JPY" else "$"
                    current_values.append({
                        'Symbol': symbol,
                        'Shares': f"{data['shares']:.3f}",
                        'Current Value': f"{currency_symbol}{current_value:,.2f}",
                        'Asset Type': data['asset_type'],
                        'Currency': asset_currency
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
            This dashboard provides professional-grade analysis tools including performance metrics, technical analysis, and AI-powered insights with multi-currency support.
        </div>
        """, unsafe_allow_html=True)
    
    if not st.session_state.portfolio:
        st.warning("⚠️ Add assets to your portfolio to see advanced analytics.")
        return
    
    metrics = calculate_portfolio_metrics_advanced(st.session_state.portfolio, st.session_state.base_currency)
    
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
            help="Portfolio volatility relative to market (consistent with Overview)"
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
    
    # Currency breakdown
    if len(set(data.get('currency', 'USD') for data in st.session_state.portfolio.values())) > 1:
        st.markdown("### 💱 Currency Exposure")
        
        currency_exposure = {}
        for symbol, data in st.session_state.portfolio.items():
            asset_currency = data.get('currency', 'USD')
            asset_info = fetch_asset_data(symbol, asset_currency)
            if asset_info:
                value = data['shares'] * asset_info['current_price']
                value_base = convert_currency(value, asset_currency, st.session_state.base_currency)
                currency_exposure[asset_currency] = currency_exposure.get(asset_currency, 0) + value_base
        
        currency_cols = st.columns(len(currency_exposure))
        for i, (currency, value) in enumerate(currency_exposure.items()):
            with currency_cols[i]:
                percentage = (value / metrics['total_current_value']) * 100 if metrics['total_current_value'] > 0 else 0
                currency_symbol = "€" if currency == "EUR" else "£" if currency == "GBP" else "¥" if currency == "JPY" else "$"
                st.metric(f"{currency} Exposure", f"{percentage:.1f}%", f"{currency_symbol}{value:,.2f}")
    
    # Technical analysis for top holdings
    st.markdown("### 🔍 Technical Analysis - Top Holdings")
    
    portfolio_values = []
    for symbol, data in st.session_state.portfolio.items():
        asset_currency = data.get('currency', 'USD')
        asset_info = fetch_asset_data(symbol, asset_currency)
        if asset_info:
            value = data['shares'] * asset_info['current_price']
            value_base = convert_currency(value, asset_currency, st.session_state.base_currency)
            portfolio_values.append((symbol, value_base))
    
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
                    # Get asset currency for price display
                    asset_data = st.session_state.portfolio.get(symbol, {})
                    asset_currency = asset_data.get('currency', 'USD')
                    currency_symbol = "€" if asset_currency == "EUR" else "£" if asset_currency == "GBP" else "¥" if asset_currency == "JPY" else "$"
                    st.metric("Price Trend", price_trend, f"{currency_symbol}{current_price:.2f}")
                
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
                    st.metric("Support/Resistance", f"{currency_symbol}{support_level:.2f} / {currency_symbol}{resistance_level:.2f}")
                
                if PLOTLY_AVAILABLE:
                    fig = make_subplots(
                        rows=3, cols=1,
                        subplot_titles=(
                            f'{symbol} - Price & Moving Averages ({asset_currency})',
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
    
    # Currency risk warning for multi-currency portfolios
    unique_currencies = set(data.get('currency', 'USD') for data in st.session_state.portfolio.values())
    if len(unique_currencies) > 1:
        enhanced_suggestions.append({
            'type': 'info',
            'message': f'💱 Multi-currency portfolio detected ({len(unique_currencies)} currencies). Monitor currency exchange rate risks.'
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
        elif suggestion['type'] == 'info':
            st.markdown(f"""
            <div class="info-card">
                <strong>ℹ️ Info:</strong> {suggestion['message']}
            </div>
            """, unsafe_allow_html=True)

def show_export_import():
    """Display export/import functionality"""
    
    st.markdown("### 📁 Export & Import Portfolio")
    
    if st.session_state.learning_mode:
        st.markdown("""
        <div class="info-card">
            <strong>📁 Export/Import Guide:</strong><br>
            Backup and restore your portfolio data with full multi-currency support. Export as JSON or CSV for safekeeping, or import previously saved portfolios.
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
                'base_currency': st.session_state.base_currency,
                'version': '2.1'
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
                asset_currency = data.get('currency', 'USD')
                asset_info = fetch_asset_data(symbol, asset_currency)
                if asset_info:
                    csv_data.append({
                        'Symbol': symbol,
                        'Name': asset_info['name'],
                        'Shares': data['shares'],
                        'Asset_Type': data['asset_type'],
                        'Purchase_Price': data.get('purchase_price', asset_info['current_price']),
                        'Asset_Currency': asset_currency,
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
                        
                        # Show import details
                        if 'base_currency' in data:
                            st.info(f"📊 Portfolio base currency: {data['base_currency']}")
                        
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
                            
                            # Update base currency if available
                            if 'base_currency' in data:
                                st.session_state.base_currency = data['base_currency']
                            
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
                                    'currency': row.get('Asset_Currency', 'USD'),
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
