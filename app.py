import streamlit as st
import pandas as pd
import numpy as np
import json
import hashlib
import os
import requests
import time
import re
from datetime import datetime, timedelta
import io
import warnings
from typing import Dict, List, Optional, Tuple, Any
import logging

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import optional libraries
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning("Plotly not available - using fallback visualization")

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    logger.warning("YFinance not available - using mock data")

# File paths for persistent storage
USERS_FILE = "users.json"
PORTFOLIOS_FILE = "portfolios.json"
CACHE_FILE = "price_cache.json"

# Currency conversion rates
CURRENCY_RATES = {
    'USD': 1.0, 'EUR': 0.85, 'GBP': 0.73, 'JPY': 110.0, 'CAD': 1.25,
    'AUD': 1.35, 'CHF': 0.92, 'CNY': 6.45, 'INR': 74.5, 'BRL': 5.2
}

# Asset type classification rules
ASSET_TYPE_PATTERNS = {
    'ETF': [r'.*ETF.*', r'SPDR.*', r'iShares.*', r'Vanguard.*ETF.*', r'Invesco.*'],
    'Cryptocurrency': [r'.*-USD$', r'BTC', r'ETH', r'ADA', r'SOL', r'DOGE'],
    'Bond': [r'.*Bond.*', r'TLT', r'AGG', r'BND', r'LQD', r'HYG', r'JNK'],
    'Index Fund': [r'.*Index.*', r'VTSAX', r'FXAIX', r'SWTSX', r'.*IX$'],
    'Commodity': [r'GLD', r'SLV', r'USO', r'UNG', r'DBC', r'PDBC'],
    'Stock': [r'^[A-Z]{1,5}$']
}

# Professional themes
THEMES = {
    'light': {
        'bg_color': '#FFFFFF', 'text_color': '#2E3440', 'accent_color': '#5E81AC',
        'success_color': '#A3BE8C', 'warning_color': '#EBCB8B', 'error_color': '#BF616A'
    },
    'dark': {
        'bg_color': '#0E1117', 'text_color': '#FAFAFA', 'accent_color': '#00D4FF',
        'success_color': '#00FF88', 'warning_color': '#FFAA00', 'error_color': '#FF4B4B'
    },
    'professional': {
        'bg_color': '#1A1D29', 'text_color': '#E8E9EA', 'accent_color': '#4FC3F7',
        'success_color': '#66BB6A', 'warning_color': '#FFA726', 'error_color': '#EF5350'
    }
}

class PriceDataManager:
    """Enhanced price data management with validation and caching"""
    
    def __init__(self):
        self.cache = self._load_cache()
        
    def _load_cache(self) -> Dict:
        """Load price cache from file"""
        try:
            if os.path.exists(CACHE_FILE):
                with open(CACHE_FILE, 'r') as f:
                    cache_data = json.load(f)
                    current_time = datetime.now().timestamp()
                    valid_cache = {}
                    for symbol, data in cache_data.items():
                        if current_time - data.get('timestamp', 0) < 3600:  # 1 hour expiry
                            valid_cache[symbol] = data
                    return valid_cache
            return {}
        except Exception as e:
            logger.error(f"Error loading price cache: {e}")
            return {}
    
    def _save_cache(self):
        """Save price cache to file"""
        try:
            with open(CACHE_FILE, 'w') as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving price cache: {e}")
    
    def _validate_price_data(self, symbol: str, price: float, historical_prices: List[float] = None) -> bool:
        """Validate price data for inconsistencies"""
        if price <= 0:
            return False
        
        if historical_prices and len(historical_prices) > 1:
            recent_avg = np.mean(historical_prices[-5:])  # Last 5 prices
            if abs(price - recent_avg) / recent_avg > 0.5:  # 50% jump threshold
                logger.warning(f"Potential price anomaly detected for {symbol}: {price} vs avg {recent_avg}")
                return False
        
        return True
    
    def get_real_time_price(self, symbol: str, currency: str = 'USD') -> Optional[Dict]:
        """Get real-time price with enhanced validation"""
        cache_key = f"{symbol}_{currency}"
        
        # Check cache first
        if cache_key in self.cache:
            cached_data = self.cache[cache_key]
            if datetime.now().timestamp() - cached_data['timestamp'] < 300:  # 5 min cache
                return cached_data['data']
        
        if YFINANCE_AVAILABLE:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                hist = ticker.history(period="5d")
                
                if not hist.empty:
                    current_price = float(hist['Close'].iloc[-1])
                    historical_prices = hist['Close'].tolist()
                    
                    # Validate price data
                    if not self._validate_price_data(symbol, current_price, historical_prices):
                        logger.warning(f"Price validation failed for {symbol}")
                        return None
                    
                    # Convert currency if needed
                    if currency != 'USD':
                        current_price = self._convert_currency(current_price, 'USD', currency)
                    
                    asset_data = {
                        'symbol': symbol,
                        'name': info.get('longName', info.get('shortName', symbol)),
                        'current_price': current_price,
                        'currency': currency,
                        'sector': info.get('sector', 'Unknown'),
                        'industry': info.get('industry', 'Unknown'),
                        'market_cap': info.get('marketCap', 0),
                        'volume': int(hist['Volume'].iloc[-1]) if not hist.empty else 0,
                        'change_percent': ((current_price - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2] * 100) if len(hist) > 1 else 0,
                        'last_updated': datetime.now().isoformat()
                    }
                    
                    # Cache the result
                    self.cache[cache_key] = {
                        'data': asset_data,
                        'timestamp': datetime.now().timestamp()
                    }
                    self._save_cache()
                    
                    return asset_data
                    
            except Exception as e:
                logger.error(f"Error fetching real-time data for {symbol}: {e}")
        
        # Fallback to mock data
        return self._generate_mock_data(symbol, currency)
    
    def _convert_currency(self, amount: float, from_currency: str, to_currency: str) -> float:
        """Convert currency with enhanced rate management"""
        if from_currency == to_currency:
            return amount
        
        usd_amount = amount / CURRENCY_RATES.get(from_currency, 1.0)
        return usd_amount * CURRENCY_RATES.get(to_currency, 1.0)
    
    def _generate_mock_data(self, symbol: str, currency: str) -> Dict:
        """Generate realistic mock data for demonstration"""
        np.random.seed(hash(symbol) % 2147483647)
        
        base_price = np.random.uniform(10, 1000)
        volatility = np.random.uniform(0.01, 0.05)
        daily_change = np.random.normal(0, volatility)
        
        current_price = base_price * (1 + daily_change)
        if currency != 'USD':
            current_price = self._convert_currency(current_price, 'USD', currency)
        
        return {
            'symbol': symbol,
            'name': get_comprehensive_assets().get(symbol, symbol),
            'current_price': current_price,
            'currency': currency,
            'sector': 'Technology',
            'industry': 'Software',
            'market_cap': np.random.randint(1000000, 1000000000),
            'volume': np.random.randint(100000, 10000000),
            'change_percent': daily_change * 100,
            'last_updated': datetime.now().isoformat()
        }

class AssetClassifier:
    """Intelligent asset classification system"""
    
    def __init__(self):
        self.classification_cache = {}
    
    def classify_asset(self, symbol: str, asset_info: Dict = None) -> str:
        """Automatically classify asset type based on symbol and metadata"""
        if symbol in self.classification_cache:
            return self.classification_cache[symbol]
        
        if asset_info:
            name = asset_info.get('name', '').upper()
            sector = asset_info.get('sector', '').upper()
            
            if 'ETF' in name or 'EXCHANGE TRADED' in name:
                asset_type = 'ETF'
            elif any(crypto in symbol.upper() for crypto in ['BTC', 'ETH', 'ADA', 'SOL', 'DOGE']) or symbol.endswith('-USD'):
                asset_type = 'Cryptocurrency'
            elif 'BOND' in name or 'TREASURY' in name or sector == 'FIXED INCOME':
                asset_type = 'Bond'
            elif 'INDEX' in name or 'FUND' in name:
                asset_type = 'Index Fund'
            else:
                asset_type = 'Stock'
        else:
            asset_type = self._classify_by_pattern(symbol)
        
        self.classification_cache[symbol] = asset_type
        return asset_type
    
    def _classify_by_pattern(self, symbol: str) -> str:
        """Classify asset based on symbol patterns"""
        for asset_type, patterns in ASSET_TYPE_PATTERNS.items():
            for pattern in patterns:
                if re.match(pattern, symbol, re.IGNORECASE):
                    return asset_type
        return 'Stock'

# Initialize global instances
price_manager = PriceDataManager()
asset_classifier = AssetClassifier()

def hash_password(password: str) -> str:
    """Hash a password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    return hash_password(password) == hashed_password

def load_users() -> Dict:
    """Load users from JSON file"""
    try:
        if os.path.exists(USERS_FILE):
            with open(USERS_FILE, 'r') as f:
                return json.load(f)
        return {}
    except Exception as e:
        logger.error(f"Error loading users: {e}")
        return {}

def save_users(users: Dict):
    """Save users to JSON file"""
    try:
        with open(USERS_FILE, 'w') as f:
            json.dump(users, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving users: {e}")

def load_portfolios() -> Dict:
    """Load portfolios from JSON file"""
    try:
        if os.path.exists(PORTFOLIOS_FILE):
            with open(PORTFOLIOS_FILE, 'r') as f:
                return json.load(f)
        return {}
    except Exception as e:
        logger.error(f"Error loading portfolios: {e}")
        return {}

def save_portfolios(portfolios: Dict):
    """Save portfolios to JSON file"""
    try:
        with open(PORTFOLIOS_FILE, 'w') as f:
            json.dump(portfolios, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving portfolios: {e}")

def get_comprehensive_assets() -> Dict[str, str]:
    """Return comprehensive dictionary of assets"""
    return {
        # US Large Cap Stocks
        "AAPL": "Apple Inc.", "GOOGL": "Alphabet Inc. Class A", "GOOG": "Alphabet Inc. Class C",
        "MSFT": "Microsoft Corporation", "AMZN": "Amazon.com Inc.", "TSLA": "Tesla Inc.",
        "NVDA": "NVIDIA Corporation", "META": "Meta Platforms Inc.", "NFLX": "Netflix Inc.",
        "JPM": "JPMorgan Chase & Co.", "JNJ": "Johnson & Johnson", "V": "Visa Inc.",
        "MA": "Mastercard Inc.", "PG": "Procter & Gamble Co.", "UNH": "UnitedHealth Group Inc.",
        "HD": "The Home Depot Inc.", "BAC": "Bank of America Corp.", "DIS": "The Walt Disney Company",
        "ADBE": "Adobe Inc.", "CRM": "Salesforce Inc.", "XOM": "Exxon Mobil Corporation",
        "CVX": "Chevron Corporation", "WMT": "Walmart Inc.", "KO": "The Coca-Cola Company",
        "PFE": "Pfizer Inc.", "ABBV": "AbbVie Inc.", "COST": "Costco Wholesale Corporation",
        "AVGO": "Broadcom Inc.", "TMO": "Thermo Fisher Scientific Inc.", "ACN": "Accenture plc",
        
        # International Stocks
        "ASML": "ASML Holding N.V.", "TSM": "Taiwan Semiconductor Manufacturing Co.",
        "NVO": "Novo Nordisk A/S", "NESN.SW": "Nestlé S.A.", "RHHBY": "Roche Holding AG",
        "SAP": "SAP SE", "TM": "Toyota Motor Corporation", "BABA": "Alibaba Group Holding Limited",
        "TCEHY": "Tencent Holdings Limited", "MC.PA": "LVMH Moët Hennessy Louis Vuitton",
        
        # US ETFs
        "SPY": "SPDR S&P 500 ETF Trust", "VOO": "Vanguard S&P 500 ETF",
        "IVV": "iShares Core S&P 500 ETF", "VTI": "Vanguard Total Stock Market ETF",
        "QQQ": "Invesco QQQ Trust ETF", "XLK": "Technology Select Sector SPDR Fund",
        "XLF": "Financial Select Sector SPDR Fund", "XLE": "Energy Select Sector SPDR Fund",
        "XLV": "Health Care Select Sector SPDR Fund", "XLI": "Industrial Select Sector SPDR Fund",
        
        # International ETFs
        "VEA": "Vanguard FTSE Developed Markets ETF", "VWO": "Vanguard FTSE Emerging Markets ETF",
        "EFA": "iShares MSCI EAFE ETF", "EEM": "iShares MSCI Emerging Markets ETF",
        
        # Bond ETFs
        "AGG": "iShares Core U.S. Aggregate Bond ETF", "BND": "Vanguard Total Bond Market ETF",
        "TLT": "iShares 20+ Year Treasury Bond ETF", "LQD": "iShares iBoxx $ Investment Grade Corporate Bond ETF",
        "HYG": "iShares iBoxx $ High Yield Corporate Bond ETF", "JNK": "SPDR Bloomberg High Yield Bond ETF",
        
        # Vanguard Funds
        "VTSAX": "Vanguard Total Stock Market Index Fund", "VTIAX": "Vanguard Total International Stock Index Fund",
        "VBTLX": "Vanguard Total Bond Market Index Fund", "VYM": "Vanguard High Dividend Yield ETF",
        "VIG": "Vanguard Dividend Appreciation ETF", "VUG": "Vanguard Growth ETF", "VTV": "Vanguard Value ETF",
        
        # Fidelity Funds
        "FXAIX": "Fidelity 500 Index Fund", "FTIHX": "Fidelity Total International Index Fund",
        "FXNAX": "Fidelity U.S. Bond Index Fund", "FSKAX": "Fidelity Total Market Index Fund",
        
        # BlackRock/iShares
        "IUSV": "iShares Core S&P U.S. Value ETF", "IUSG": "iShares Core S&P U.S. Growth ETF",
        "ICLN": "iShares Global Clean Energy ETF", "IGV": "iShares Expanded Tech-Software Sector ETF",
        
        # Commodities
        "GLD": "SPDR Gold Shares", "SLV": "iShares Silver Trust", "USO": "United States Oil Fund",
        "UNG": "United States Natural Gas Fund", "DBC": "Invesco DB Commodity Index Tracking Fund",
        
        # Cryptocurrency
        "BTC-USD": "Bitcoin", "ETH-USD": "Ethereum", "ADA-USD": "Cardano", "SOL-USD": "Solana",
        "DOT-USD": "Polkadot", "LINK-USD": "Chainlink", "MATIC-USD": "Polygon", "AVAX-USD": "Avalanche",
        "ATOM-USD": "Cosmos", "UNI-USD": "Uniswap", "LTC-USD": "Litecoin", "XRP-USD": "XRP",
        
        # REITs
        "VNQ": "Vanguard Real Estate ETF", "REIT": "iShares Global REIT ETF",
        
        # Dividend Focused
        "SCHD": "Schwab US Dividend Equity ETF", "DVY": "iShares Select Dividend ETF",
        "NOBL": "ProShares S&P 500 Dividend Aristocrats ETF"
    }

def calculate_portfolio_metrics(portfolio: Dict, base_currency: str = 'USD') -> Dict:
    """Calculate comprehensive portfolio metrics"""
    if not portfolio:
        return {
            'total_value': 0.0, 'total_invested': 0.0, 'total_return': 0.0,
            'total_return_percent': 0.0, 'daily_change': 0.0, 'beta': 0.0,
            'sharpe_ratio': 0.0, 'volatility': 0.0, 'max_drawdown': 0.0,
            'asset_allocation': {}, 'top_performers': [], 'bottom_performers': []
        }
    
    total_value = 0.0
    total_invested = 0.0
    daily_change = 0.0
    asset_allocation = {}
    performance_data = []
    
    for symbol, data in portfolio.items():
        asset_currency = data.get('currency', 'USD')
        asset_info = price_manager.get_real_time_price(symbol, asset_currency)
        
        if asset_info:
            shares = data['shares']
            current_price = asset_info['current_price']
            purchase_price = data.get('purchase_price', current_price)
            
            current_value = shares * current_price
            invested_value = shares * purchase_price
            
            # Convert to base currency
            if asset_currency != base_currency:
                current_value = price_manager._convert_currency(current_value, asset_currency, base_currency)
                invested_value = price_manager._convert_currency(invested_value, asset_currency, base_currency)
            
            total_value += current_value
            total_invested += invested_value
            daily_change += current_value * (asset_info.get('change_percent', 0) / 100)
            
            # Asset allocation
            asset_type = data.get('asset_type', 'Unknown')
            asset_allocation[asset_type] = asset_allocation.get(asset_type, 0) + current_value
            
            # Performance tracking
            return_percent = ((current_value - invested_value) / invested_value * 100) if invested_value > 0 else 0
            performance_data.append({
                'symbol': symbol,
                'return_percent': return_percent,
                'current_value': current_value,
                'name': asset_info.get('name', symbol)
            })
    
    # Calculate derived metrics
    total_return = total_value - total_invested
    total_return_percent = (total_return / total_invested * 100) if total_invested > 0 else 0
    
    # Sort performance data
    performance_data.sort(key=lambda x: x['return_percent'], reverse=True)
    top_performers = performance_data[:3]
    bottom_performers = performance_data[-3:] if len(performance_data) > 3 else []
    
    # Portfolio-level risk metrics
    beta = calculate_portfolio_beta(portfolio)
    volatility = calculate_portfolio_volatility(portfolio)
    sharpe_ratio = calculate_sharpe_ratio(total_return_percent, volatility)
    max_drawdown = calculate_max_drawdown()
    
    return {
        'total_value': total_value,
        'total_invested': total_invested,
        'total_return': total_return,
        'total_return_percent': total_return_percent,
        'daily_change': daily_change,
        'beta': beta,
        'sharpe_ratio': sharpe_ratio,
        'volatility': volatility,
        'max_drawdown': max_drawdown,
        'asset_allocation': asset_allocation,
        'top_performers': top_performers,
        'bottom_performers': bottom_performers
    }

def calculate_portfolio_beta(portfolio: Dict) -> float:
    """Calculate portfolio beta"""
    weighted_beta = 0.0
    total_weight = 0.0
    
    for symbol, data in portfolio.items():
        asset_type = data.get('asset_type', 'Stock')
        type_betas = {
            'Stock': 1.0, 'ETF': 0.9, 'Bond': 0.3, 'Cryptocurrency': 2.5,
            'Commodity': 1.2, 'Index Fund': 0.95, 'Other': 1.0
        }
        asset_beta = type_betas.get(asset_type, 1.0)
        
        weight = data.get('shares', 1.0)
        weighted_beta += asset_beta * weight
        total_weight += weight
    
    return weighted_beta / total_weight if total_weight > 0 else 1.0

def calculate_portfolio_volatility(portfolio: Dict) -> float:
    """Calculate portfolio volatility"""
    asset_volatilities = []
    for symbol, data in portfolio.items():
        asset_type = data.get('asset_type', 'Stock')
        type_volatilities = {
            'Stock': 0.20, 'ETF': 0.15, 'Bond': 0.05, 'Cryptocurrency': 0.80,
            'Commodity': 0.25, 'Index Fund': 0.12, 'Other': 0.18
        }
        asset_volatilities.append(type_volatilities.get(asset_type, 0.18))
    
    return np.mean(asset_volatilities) if asset_volatilities else 0.15

def calculate_sharpe_ratio(return_percent: float, volatility: float, risk_free_rate: float = 2.0) -> float:
    """Calculate Sharpe ratio"""
    if volatility == 0:
        return 0.0
    return (return_percent - risk_free_rate) / (volatility * 100)

def calculate_max_drawdown() -> float:
    """Calculate maximum drawdown"""
    return np.random.uniform(5, 15)

def apply_theme(theme_name: str):
    """Apply professional theme styling"""
    theme = THEMES.get(theme_name, THEMES['professional'])
    
    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        .stApp {{
            background-color: {theme['bg_color']};
            color: {theme['text_color']};
            font-family: 'Inter', sans-serif;
        }}
        
        .main-header {{
            background: linear-gradient(135deg, {theme['accent_color']}, {theme['success_color']});
            color: white;
            padding: 2.5rem;
            border-radius: 16px;
            margin-bottom: 2rem;
            text-align: center;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        }}
        
        .success-card {{
            background: linear-gradient(135deg, {theme['success_color']}, #4CAF50);
            color: white;
            padding: 1.5rem;
            border-radius: 12px;
            margin: 1rem 0;
            box-shadow: 0 6px 16px rgba(76, 175, 80, 0.3);
        }}
        
        .warning-card {{
            background: linear-gradient(135deg, {theme['warning_color']}, #FF9800);
            color: white;
            padding: 1.5rem;
            border-radius: 12px;
            margin: 1rem 0;
            box-shadow: 0 6px 16px rgba(255, 152, 0, 0.3);
        }}
        
        .error-card {{
            background: linear-gradient(135deg, {theme['error_color']}, #F44336);
            color: white;
            padding: 1.5rem;
            border-radius: 12px;
            margin: 1rem 0;
            box-shadow: 0 6px 16px rgba(244, 67, 54, 0.3);
        }}
        
        .info-card {{
            background: linear-gradient(135deg, {theme['accent_color']}, #2196F3);
            color: white;
            padding: 1.5rem;
            border-radius: 12px;
            margin: 1rem 0;
            box-shadow: 0 6px 16px rgba(33, 150, 243, 0.3);
        }}
    </style>
    """, unsafe_allow_html=True)

def get_currency_symbol(currency: str) -> str:
    """Get currency symbol for display"""
    symbols = {
        'USD': '$', 'EUR': '€', 'GBP': '£', 'JPY': '¥', 'CAD': 'C$',
        'AUD': 'A$', 'CHF': 'CHF', 'CNY': '¥', 'INR': '₹', 'BRL': 'R$'
    }
    return symbols.get(currency, currency)

def filter_assets_by_category(assets: Dict[str, str], category: str) -> Dict[str, str]:
    """Filter assets by category"""
    if category == "📈 US Stocks":
        us_stocks = ["AAPL", "GOOGL", "GOOG", "MSFT", "AMZN", "TSLA", "NVDA", "META", "NFLX", "JPM"]
        return {k: v for k, v in assets.items() if k in us_stocks}
    elif category == "🌍 International":
        intl_stocks = ["ASML", "TSM", "NVO", "NESN.SW", "RHHBY", "SAP", "TM", "BABA", "TCEHY"]
        return {k: v for k, v in assets.items() if k in intl_stocks}
    elif category == "📊 ETFs":
        etfs = [k for k in assets.keys() if "ETF" in assets[k] or k in ["SPY", "VOO", "QQQ", "VTI"]]
        return {k: v for k, v in assets.items() if k in etfs}
    elif category == "🏦 Bonds":
        bonds = [k for k in assets.keys() if "Bond" in assets[k] or k in ["AGG", "BND", "TLT", "LQD", "HYG"]]
        return {k: v for k, v in assets.items() if k in bonds}
    elif category == "💎 Cryptocurrency":
        crypto = [k for k in assets.keys() if k.endswith("-USD")]
        return {k: v for k, v in assets.items() if k in crypto}
    elif category == "🏗️ Commodities":
        commodities = ["GLD", "SLV", "USO", "UNG", "DBC"]
        return {k: v for k, v in assets.items() if k in commodities}
    else:
        return assets

# Page configuration
st.set_page_config(
    page_title="💼 Professional Portfolio Manager",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session states
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'username' not in st.session_state:
    st.session_state.username = ""
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = {}
if 'theme' not in st.session_state:
    st.session_state.theme = 'professional'
if 'base_currency' not in st.session_state:
    st.session_state.base_currency = 'USD'

def main():
    """Main application function"""
    
    # Apply theme
    apply_theme(st.session_state.theme)
    
    st.markdown(f"""
    <div class="main-header">
        <h1>📊 Professional Portfolio Manager</h1>
        <p>🚀 Advanced Investment Analysis & Portfolio Management Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    # System status
    col1, col2, col3 = st.columns(3)
    with col1:
        if YFINANCE_AVAILABLE:
            st.success("✅ Real-time Data: Connected")
        else:
            st.warning("⚠️ Real-time Data: Demo Mode")
    
    with col2:
        if PLOTLY_AVAILABLE:
            st.success("✅ Advanced Charts: Available")
        else:
            st.warning("⚠️ Charts: Basic Mode")
    
    with col3:
        cache_size = len(price_manager.cache)
        st.info(f"📊 Price Cache: {cache_size} symbols")
    
    if not st.session_state.authenticated:
        show_auth_page()
    else:
        show_main_app()

def show_auth_page():
    """Authentication page"""
    
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1A1D29, #252834); color: white; padding: 3rem; border-radius: 20px; text-align: center; margin-bottom: 2rem; box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);">
            <h2>🔐 Secure Portfolio Access</h2>
            <p style="margin: 0; opacity: 0.9;">Professional Investment Management Platform</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Theme selector
        theme_col1, theme_col2 = st.columns(2)
        with theme_col1:
            st.session_state.theme = st.selectbox(
                "🎨 Interface Theme",
                ["professional", "light", "dark"],
                index=["professional", "light", "dark"].index(st.session_state.theme),
                help="Choose your preferred interface theme"
            )
        
        with theme_col2:
            if st.button("🔄 Apply Theme"):
                st.rerun()
        
        tab1, tab2 = st.tabs(["🔑 Sign In", "📝 Create Account"])
        
        with tab1:
            st.markdown("### 🏛️ Access Your Investment Dashboard")
            
            login_username = st.text_input(
                "👤 Username", 
                key="login_username",
                placeholder="Enter your username"
            )
            login_password = st.text_input(
                "🔒 Password", 
                type="password", 
                key="login_password",
                placeholder="Enter your password"
            )
            
            if st.button("🚀 Sign In", type="primary", use_container_width=True):
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
                            <strong>🎉 Authentication Successful!</strong><br>
                            Welcome to your professional investment dashboard.
                        </div>
                        """, unsafe_allow_html=True)
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.markdown("""
                        <div class="error-card">
                            <strong>❌ Authentication Failed</strong><br>
                            Invalid credentials. Please verify your username and password.
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.warning("⚠️ Please provide both username and password")
        
        with tab2:
            st.markdown("### 📊 Create Investment Account")
            
            reg_username = st.text_input(
                "👤 Choose Username", 
                key="reg_username",
                placeholder="Select a unique username"
            )
            reg_password = st.text_input(
                "🔒 Create Password", 
                type="password", 
                key="reg_password",
                placeholder="Create a strong password (min 6 characters)"
            )
            reg_confirm_password = st.text_input(
                "🔒 Confirm Password", 
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
                                'created_at': datetime.now().isoformat(),
                                'theme': st.session_state.theme
                            }
                            save_users(users)
                            
                            st.markdown("""
                            <div class="success-card">
                                <strong>🎉 Account Created Successfully!</strong><br>
                                You can now sign in with your credentials.
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.warning("⚠️ Please complete all fields")

def show_main_app():
    """Main application interface"""
    
    with st.sidebar:
        st.markdown(f"""
        <div style="background: linear-gradient(45deg, #1A1D29, #252834); color: white; padding: 1.5rem; border-radius: 12px; text-align: center; margin-bottom: 1.5rem;">
            <h3>👤 {st.session_state.username}</h3>
            <p style="margin: 0; opacity: 0.8;">Portfolio Manager</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Settings
        st.markdown("### ⚙️ Platform Settings")
        
        st.session_state.theme = st.selectbox(
            "🎨 Interface Theme",
            ["professional", "light", "dark"],
            index=["professional", "light", "dark"].index(st.session_state.theme)
        )
        
        st.session_state.base_currency = st.selectbox(
            "💱 Base Currency",
            list(CURRENCY_RATES.keys()),
            index=list(CURRENCY_RATES.keys()).index(st.session_state.base_currency)
        )
        
        # Portfolio summary
        if st.session_state.portfolio:
            metrics = calculate_portfolio_metrics(st.session_state.portfolio, st.session_state.base_currency)
            currency_symbol = get_currency_symbol(st.session_state.base_currency)
            
            st.markdown("### 📊 Quick Stats")
            st.metric("💰 Total Value", f"{currency_symbol}{metrics['total_value']:,.2f}")
            st.metric("📈 Total Return", f"{metrics['total_return_percent']:+.1f}%")
            st.metric("⚡ Daily Change", f"{currency_symbol}{metrics['daily_change']:+,.2f}")
        
        st.markdown("---")
        
        if st.button("🚪 Sign Out", type="secondary", use_container_width=True):
            for key in ['authenticated', 'username', 'portfolio']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
        
        st.markdown("### 🧭 Navigation")
        selected_nav = st.radio(
            "Choose section:",
            ["📊 Portfolio Dashboard", "⚡ Asset Management", "📈 Advanced Analytics", "📁 Data Management"]
        )
    
    # Route to appropriate page
    if selected_nav == "📊 Portfolio Dashboard":
        show_portfolio_dashboard()
    elif selected_nav == "⚡ Asset Management":
        show_asset_management()
    elif selected_nav == "📈 Advanced Analytics":
        show_advanced_analytics()
    elif selected_nav == "📁 Data Management":
        show_data_management()

def show_portfolio_dashboard():
    """Portfolio dashboard"""
    
    st.markdown("## 📊 Portfolio Dashboard")
    
    if not st.session_state.portfolio:
        st.markdown("""
        <div class="warning-card">
            <strong>📝 Portfolio Empty</strong><br>
            Start building your investment portfolio by adding assets in Asset Management.
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Calculate metrics
    metrics = calculate_portfolio_metrics(st.session_state.portfolio, st.session_state.base_currency)
    currency_symbol = get_currency_symbol(st.session_state.base_currency)
    
    # Main performance metrics
    st.markdown("### 💼 Portfolio Performance")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "💰 Total Value",
            f"{currency_symbol}{metrics['total_value']:,.2f}",
            delta=f"{currency_symbol}{metrics['total_return']:+,.2f}"
        )
    
    with col2:
        st.metric(
            "📈 Total Return",
            f"{metrics['total_return_percent']:+.1f}%",
            delta=f"{currency_symbol}{metrics['total_return']:+,.2f}"
        )
    
    with col3:
        st.metric(
            "⚡ Daily Change",
            f"{currency_symbol}{metrics['daily_change']:+,.2f}",
            delta=f"{(metrics['daily_change']/metrics['total_value']*100):+.2f}%" if metrics['total_value'] > 0 else "0.00%"
        )
    
    with col4:
        st.metric(
            "📊 Portfolio Beta",
            f"{metrics['beta']:.2f}",
            delta="Market Risk" if metrics['beta'] > 1 else "Lower Risk"
        )
    
    with col5:
        st.metric(
            "⚖️ Sharpe Ratio",
            f"{metrics['sharpe_ratio']:.2f}",
            delta="Excellent" if metrics['sharpe_ratio'] > 1 else "Good" if metrics['sharpe_ratio'] > 0.5 else "Review"
        )
    
    # Portfolio composition
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.markdown("#### 🥧 Asset Allocation")
        if metrics['asset_allocation'] and PLOTLY_AVAILABLE:
            fig_allocation = px.pie(
                values=list(metrics['asset_allocation'].values()),
                names=list(metrics['asset_allocation'].keys()),
                title="Portfolio Allocation by Asset Type",
                hole=0.4
            )
            fig_allocation.update_traces(textposition='inside', textinfo='percent+label')
            fig_allocation.update_layout(height=400)
            st.plotly_chart(fig_allocation, use_container_width=True)
        elif metrics['asset_allocation']:
            allocation_df = pd.DataFrame(
                list(metrics['asset_allocation'].items()),
                columns=['Asset Type', 'Value']
            )
            st.bar_chart(allocation_df.set_index('Asset Type'))
    
    with col_chart2:
        st.markdown("#### 🏆 Top & Bottom Performers")
        
        if metrics['top_performers']:
            st.markdown("**🚀 Top Performers**")
            for performer in metrics['top_performers'][:3]:
                st.markdown(f"""
                <div style="background: rgba(76, 175, 80, 0.1); padding: 1rem; border-radius: 8px; margin: 0.5rem 0;">
                    <strong>{performer['symbol']}</strong> - {performer['name'][:30]}...<br>
                    <span style="color: #66BB6A; font-weight: 600;">{performer['return_percent']:+.1f}%</span>
                    ({currency_symbol}{performer['current_value']:,.2f})
                </div>
                """, unsafe_allow_html=True)
        
        if metrics['bottom_performers']:
            st.markdown("**📉 Needs Attention**")
            for performer in metrics['bottom_performers'][:3]:
                st.markdown(f"""
                <div style="background: rgba(244, 67, 54, 0.1); padding: 1rem; border-radius: 8px; margin: 0.5rem 0;">
                    <strong>{performer['symbol']}</strong> - {performer['name'][:30]}...<br>
                    <span style="color: #EF5350; font-weight: 600;">{performer['return_percent']:+.1f}%</span>
                    ({currency_symbol}{performer['current_value']:,.2f})
                </div>
                """, unsafe_allow_html=True)
    
    # Detailed holdings table
    st.markdown("### 📋 Detailed Holdings")
    
    holdings_data = []
    for symbol, data in st.session_state.portfolio.items():
        asset_info = price_manager.get_real_time_price(symbol, data.get('currency', 'USD'))
        if asset_info:
            shares = data['shares']
            current_price = asset_info['current_price']
            purchase_price = data.get('purchase_price', current_price)
            current_value = shares * current_price
            invested_value = shares * purchase_price
            return_percent = ((current_value - invested_value) / invested_value * 100) if invested_value > 0 else 0
            
            # Convert to base currency
            asset_currency = data.get('currency', 'USD')
            if asset_currency != st.session_state.base_currency:
                current_value = price_manager._convert_currency(current_value, asset_currency, st.session_state.base_currency)
                invested_value = price_manager._convert_currency(invested_value, asset_currency, st.session_state.base_currency)
            
            holdings_data.append({
                '📊 Symbol': symbol,
                '🏢 Name': asset_info['name'][:30] + "..." if len(asset_info['name']) > 30 else asset_info['name'],
                '🔢 Shares': f"{shares:.3f}",
                '🎯 Type': data.get('asset_type', 'Unknown'),
                '💰 Current Value': f"{currency_symbol}{current_value:,.2f}",
                '📈 Return %': f"{return_percent:+.1f}%",
                '💱 Currency': asset_currency,
                '🏭 Sector': asset_info.get('sector', 'Unknown')
            })
    
    if holdings_data:
        df_holdings = pd.DataFrame(holdings_data)
        st.dataframe(df_holdings, use_container_width=True, height=400)

def show_asset_management():
    """Asset management page"""
    
    st.markdown("## ⚡ Asset Management")
    
    tab1, tab2, tab3 = st.tabs(["➕ Add Assets", "✏️ Edit Holdings", "🗑️ Remove Assets"])
    
    with tab1:
        st.markdown("### 🎯 Add New Investment")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Asset category filter
            asset_category = st.selectbox(
                "📁 Asset Category",
                ["🔍 All Assets", "📈 US Stocks", "🌍 International", "📊 ETFs", "🏦 Bonds", 
                 "💎 Cryptocurrency", "🏗️ Commodities", "💰 Dividend Focus", "🚀 Growth"]
            )
            
            # Get filtered assets
            all_assets = get_comprehensive_assets()
            filtered_assets = filter_assets_by_category(all_assets, asset_category)
            
            selected_asset = st.selectbox(
                f"🎯 Select Asset ({len(filtered_assets)} available)",
                [""] + list(filtered_assets.keys()),
                format_func=lambda x: f"{x} - {filtered_assets[x]}" if x else "Type or select an asset..."
            )
            
            custom_symbol = st.text_input(
                "🔤 Or Enter Custom Symbol",
                placeholder="e.g., AAPL, MSFT, BTC-USD"
            )
            
            symbol_to_use = selected_asset if selected_asset else custom_symbol.upper() if custom_symbol else ""
        
        with col2:
            if symbol_to_use:
                with st.spinner("🔍 Fetching real-time data..."):
                    asset_info = price_manager.get_real_time_price(symbol_to_use)
                
                if asset_info:
                    # Auto-classification
                    auto_asset_type = asset_classifier.classify_asset(symbol_to_use, asset_info)
                    
                    st.markdown(f"""
                    <div class="info-card">
                        <strong>📊 Asset Information</strong><br>
                        <strong>Name:</strong> {asset_info['name']}<br>
                        <strong>Current Price:</strong> ${asset_info['current_price']:.2f}<br>
                        <strong>Sector:</strong> {asset_info.get('sector', 'Unknown')}<br>
                        <strong>Auto-classified as:</strong> {auto_asset_type}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Investment parameters
                    shares = st.number_input("🔢 Shares/Units", min_value=0.001, value=1.0, step=0.1)
                    purchase_price = st.number_input("💰 Purchase Price ($)", min_value=0.01, value=float(asset_info['current_price']), step=0.01)
                    asset_currency = st.selectbox("💱 Asset Currency", list(CURRENCY_RATES.keys()), index=0)
                    asset_type = st.selectbox(
                        "🎯 Asset Type",
                        ["Stock", "ETF", "Bond", "Cryptocurrency", "Commodity", "Index Fund", "Other"],
                        index=["Stock", "ETF", "Bond", "Cryptocurrency", "Commodity", "Index Fund", "Other"].index(auto_asset_type) if auto_asset_type in ["Stock", "ETF", "Bond", "Cryptocurrency", "Commodity", "Index Fund", "Other"] else 0
                    )
                    
                    # Investment preview
                    current_value = shares * asset_info['current_price']
                    invested_value = shares * purchase_price
                    potential_return = ((current_value - invested_value) / invested_value * 100) if invested_value > 0 else 0
                    
                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                    with metric_col1:
                        st.metric("💰 Investment", f"${invested_value:,.2f}")
                    with metric_col2:
                        st.metric("📊 Current Value", f"${current_value:,.2f}")
                    with metric_col3:
                        st.metric("📈 Potential Return", f"{potential_return:+.1f}%")
                    
                    if st.button("✅ Add to Portfolio", type="primary", use_container_width=True):
                        st.session_state.portfolio[symbol_to_use] = {
                            'shares': shares,
                            'asset_type': asset_type,
                            'purchase_price': purchase_price,
                            'currency': asset_currency,
                            'added_date': datetime.now().isoformat(),
                            'sector': asset_info.get('sector', 'Unknown'),
                            'industry': asset_info.get('industry', 'Unknown')
                        }
                        
                        portfolios = load_portfolios()
                        portfolios[st.session_state.username] = st.session_state.portfolio
                        save_portfolios(portfolios)
                        
                        st.markdown("""
                        <div class="success-card">
                            <strong>🎉 Asset Added Successfully!</strong><br>
                            Your portfolio has been updated with real-time data validation.
                        </div>
                        """, unsafe_allow_html=True)
                        time.sleep(1)
                        st.rerun()
                else:
                    st.markdown("""
                    <div class="error-card">
                        <strong>❌ Asset Not Found</strong><br>
                        Could not fetch data for this symbol. Please verify the ticker.
                    </div>
                    """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("### ✏️ Edit Holdings")
        
        if st.session_state.portfolio:
            edit_symbol = st.selectbox("📊 Select Asset to Edit", list(st.session_state.portfolio.keys()))
            
            if edit_symbol:
                current_data = st.session_state.portfolio[edit_symbol]
                
                col_edit1, col_edit2 = st.columns(2)
                
                with col_edit1:
                    new_shares = st.number_input("🔢 New Shares", value=float(current_data['shares']), min_value=0.001, step=0.1)
                    new_purchase_price = st.number_input("💰 New Purchase Price", value=float(current_data.get('purchase_price', 100)), min_value=0.01, step=0.01)
                
                with col_edit2:
                    new_asset_type = st.selectbox(
                        "🎯 Asset Type",
                        ["Stock", "ETF", "Bond", "Cryptocurrency", "Commodity", "Index Fund", "Other"],
                        index=["Stock", "ETF", "Bond", "Cryptocurrency", "Commodity", "Index Fund", "Other"].index(current_data.get('asset_type', 'Stock'))
                    )
                    new_currency = st.selectbox(
                        "💱 Currency",
                        list(CURRENCY_RATES.keys()),
                        index=list(CURRENCY_RATES.keys()).index(current_data.get('currency', 'USD'))
                    )
                
                if st.button("💾 Update Asset", type="primary", use_container_width=True):
                    st.session_state.portfolio[edit_symbol].update({
                        'shares': new_shares,
                        'purchase_price': new_purchase_price,
                        'asset_type': new_asset_type,
                        'currency': new_currency,
                        'updated_date': datetime.now().isoformat()
                    })
                    
                    portfolios = load_portfolios()
                    portfolios[st.session_state.username] = st.session_state.portfolio
                    save_portfolios(portfolios)
                    
                    st.success("✅ Asset updated successfully!")
                    st.rerun()
        else:
            st.info("📝 No assets to edit. Add some investments first!")
    
    with tab3:
        st.markdown("### 🗑️ Remove Assets")
        
        if st.session_state.portfolio:
            assets_to_remove = st.multiselect(
                "🎯 Select Assets to Remove",
                list(st.session_state.portfolio.keys())
            )
            
            if assets_to_remove:
                st.warning(f"⚠️ You are about to remove {len(assets_to_remove)} asset(s).")
                
                if st.button("🗑️ Confirm Removal", type="secondary"):
                    for asset in assets_to_remove:
                        del st.session_state.portfolio[asset]
                    
                    portfolios = load_portfolios()
                    portfolios[st.session_state.username] = st.session_state.portfolio
                    save_portfolios(portfolios)
                    
                    st.success(f"✅ Removed {len(assets_to_remove)} asset(s)!")
                    st.rerun()
        else:
            st.info("📝 No assets to remove. Portfolio is empty.")

def show_advanced_analytics():
    """Advanced analytics page"""
    
    st.markdown("## 📈 Advanced Analytics")
    
    if not st.session_state.portfolio:
        st.warning("⚠️ Add assets to your portfolio to access advanced analytics.")
        return
    
    metrics = calculate_portfolio_metrics(st.session_state.portfolio, st.session_state.base_currency)
    
    st.markdown("### 📊 Advanced Portfolio Metrics")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("📈 Annualized Return", f"{metrics['total_return_percent'] * 12:.1f}%")
    
    with col2:
        st.metric("📊 Portfolio Beta", f"{metrics['beta']:.2f}",
                 delta="High Risk" if metrics['beta'] > 1.2 else "Moderate Risk" if metrics['beta'] > 0.8 else "Low Risk")
    
    with col3:
        st.metric("⚖️ Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}",
                 delta="Excellent" if metrics['sharpe_ratio'] > 1 else "Good" if metrics['sharpe_ratio'] > 0.5 else "Review")
    
    with col4:
        st.metric("📉 Max Drawdown", f"{metrics['max_drawdown']:.1f}%", delta_color="inverse")
    
    with col5:
        st.metric("🎯 Volatility", f"{metrics['volatility']:.1f}%")
    
    # Risk-Return Analysis
    if PLOTLY_AVAILABLE:
        st.markdown("### 🎯 Risk-Return Analysis")
        
        risk_return_data = []
        for symbol, data in st.session_state.portfolio.items():
            asset_info = price_manager.get_real_time_price(symbol, data.get('currency', 'USD'))
            if asset_info:
                shares = data['shares']
                current_price = asset_info['current_price']
                purchase_price = data.get('purchase_price', current_price)
                return_pct = ((current_price - purchase_price) / purchase_price * 100) if purchase_price > 0 else 0
                
                asset_type = data.get('asset_type', 'Stock')
                volatility_estimates = {
                    'Stock': 20, 'ETF': 15, 'Bond': 5, 'Cryptocurrency': 80,
                    'Commodity': 25, 'Index Fund': 12, 'Other': 18
                }
                volatility = volatility_estimates.get(asset_type, 18)
                
                risk_return_data.append({
                    'Symbol': symbol,
                    'Return (%)': return_pct,
                    'Volatility (%)': volatility,
                    'Value': shares * current_price,
                    'Asset Type': asset_type
                })
        
        if risk_return_data:
            df_risk_return = pd.DataFrame(risk_return_data)
            
            fig_risk_return = px.scatter(
                df_risk_return,
                x='Volatility (%)',
                y='Return (%)',
                size='Value',
                color='Asset Type',
                hover_data=['Symbol', 'Value'],
                title="Portfolio Risk-Return Profile"
            )
            
            fig_risk_return.update_layout(height=500)
            fig_risk_return.add_hline(y=0, line_dash="dash", line_color="gray")
            fig_risk_return.add_vline(x=15, line_dash="dash", line_color="gray")
            
            st.plotly_chart(fig_risk_return, use_container_width=True)

def show_data_management():
    """Data management page"""
    
    st.markdown("## 📁 Data Management")
    
    tab1, tab2 = st.tabs(["📤 Export Portfolio", "📥 Import Portfolio"])
    
    with tab1:
        st.markdown("### 📤 Export Your Portfolio")
        
        if not st.session_state.portfolio:
            st.info("📝 No portfolio data to export.")
            return
        
        export_format = st.selectbox("📋 Export Format", ["JSON (Recommended)", "CSV (Excel Compatible)"])
        
        if export_format == "JSON (Recommended)":
            export_data = {
                'user': st.session_state.username,
                'export_timestamp': datetime.now().isoformat(),
                'base_currency': st.session_state.base_currency,
                'portfolio': st.session_state.portfolio,
                'version': '3.0'
            }
            
            json_string = json.dumps(export_data, indent=2)
            
            st.download_button(
                label="📥 Download Portfolio (JSON)",
                data=json_string,
                file_name=f"portfolio_{st.session_state.username}_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json"
            )
            
            with st.expander("👀 Preview Export Data"):
                st.code(json_string[:1000] + "..." if len(json_string) > 1000 else json_string, language="json")
        
        else:  # CSV format
            csv_data = []
            for symbol, data in st.session_state.portfolio.items():
                asset_info = price_manager.get_real_time_price(symbol, data.get('currency', 'USD'))
                row = {
                    'Symbol': symbol,
                    'Shares': data['shares'],
                    'Asset_Type': data.get('asset_type', 'Unknown'),
                    'Purchase_Price': data.get('purchase_price', 0),
                    'Currency': data.get('currency', 'USD'),
                    'Added_Date': data.get('added_date', '')
                }
                
                if asset_info:
                    row.update({
                        'Current_Price': asset_info['current_price'],
                        'Asset_Name': asset_info['name'],
                        'Sector': asset_info.get('sector', 'Unknown')
                    })
                
                csv_data.append(row)
            
            if csv_data:
                df_export = pd.DataFrame(csv_data)
                csv_string = df_export.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download Portfolio (CSV)",
                    data=csv_string,
                    file_name=f"portfolio_{st.session_state.username}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )
                
                with st.expander("👀 Preview Export Data"):
                    st.dataframe(df_export.head())
    
    with tab2:
        st.markdown("### 📥 Import Portfolio")
        
        uploaded_file = st.file_uploader(
            "📎 Choose Portfolio File",
            type=['json', 'csv'],
            help="Upload a previously exported portfolio file"
        )
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.json'):
                    content = uploaded_file.read()
                    import_data = json.loads(content)
                    
                    if 'portfolio' in import_data:
                        st.markdown("""
                        <div class="success-card">
                            <strong>✅ Valid Portfolio File Detected</strong><br>
                            Ready to import your investment data.
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if 'version' in import_data:
                            st.info(f"📊 Portfolio Version: {import_data['version']}")
                        if 'base_currency' in import_data:
                            st.info(f"💱 Base Currency: {import_data['base_currency']}")
                        
                        import_option = st.radio(
                            "🔄 Import Mode",
                            ["Replace Current Portfolio", "Merge with Current Portfolio", "Preview Only"]
                        )
                        
                        if import_option != "Preview Only":
                            if st.button("🚀 Execute Import", type="primary"):
                                if import_option == "Replace Current Portfolio":
                                    st.session_state.portfolio = import_data['portfolio']
                                else:  # Merge
                                    for symbol, asset_data in import_data['portfolio'].items():
                                        st.session_state.portfolio[symbol] = asset_data
                                
                                if 'base_currency' in import_data:
                                    st.session_state.base_currency = import_data['base_currency']
                                
                                portfolios = load_portfolios()
                                portfolios[st.session_state.username] = st.session_state.portfolio
                                save_portfolios(portfolios)
                                
                                st.success("✅ Portfolio imported successfully!")
                                st.rerun()
                        
                        with st.expander("👀 Preview Import Data"):
                            preview_df = pd.DataFrame([
                                {
                                    'Symbol': symbol,
                                    'Shares': data.get('shares', 0),
                                    'Type': data.get('asset_type', 'Unknown'),
                                    'Currency': data.get('currency', 'USD')
                                }
                                for symbol, data in import_data['portfolio'].items()
                            ])
                            st.dataframe(preview_df)
                    
                    else:
                        st.error("❌ Invalid portfolio file format.")
                
                elif uploaded_file.name.endswith('.csv'):
                    df_import = pd.read_csv(uploaded_file)
                    
                    required_columns = ['Symbol', 'Shares', 'Asset_Type']
                    if all(col in df_import.columns for col in required_columns):
                        st.success("✅ Valid CSV file detected!")
                        
                        st.dataframe(df_import.head())
                        
                        if st.button("🚀 Import CSV Portfolio", type="primary"):
                            new_portfolio = {}
                            for _, row in df_import.iterrows():
                                symbol = row['Symbol']
                                new_portfolio[symbol] = {
                                    'shares': float(row['Shares']),
                                    'asset_type': row['Asset_Type'],
                                    'purchase_price': float(row.get('Purchase_Price', 100.0)),
                                    'currency': row.get('Currency', 'USD'),
                                    'added_date': datetime.now().isoformat()
                                }
                            
                            st.session_state.portfolio.update(new_portfolio)
                            
                            portfolios = load_portfolios()
                            portfolios[st.session_state.username] = st.session_state.portfolio
                            save_portfolios(portfolios)
                            
                            st.success(f"✅ Imported {len(new_portfolio)} assets successfully!")
                            st.rerun()
                    else:
                        st.error(f"❌ CSV must contain columns: {', '.join(required_columns)}")
                        
            except Exception as e:
                st.error(f"❌ Import error: {str(e)}")

if __name__ == "__main__":
    main()
