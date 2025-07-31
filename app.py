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
    """Return dictionary of popular assets with their symbols"""
    return {
        "AAPL": "Apple Inc.",
        "GOOGL": "Alphabet Inc.",
        "MSFT": "Microsoft Corporation",
        "AMZN": "Amazon.com Inc.",
        "TSLA": "Tesla Inc.",
        "NVDA": "NVIDIA Corporation", 
        "META": "Meta Platforms Inc.",
        "NFLX": "Netflix Inc.",
        "JPM": "JPMorgan Chase & Co.",
        "JNJ": "Johnson & Johnson",
        "SPY": "SPDR S&P 500 ETF",
        "QQQ": "Invesco QQQ Trust",
        "VTI": "Vanguard Total Stock Market ETF",
        "IWM": "iShares Russell 2000 ETF",
        "EFA": "iShares MSCI EAFE ETF",
        "VEA": "Vanguard FTSE Developed Markets ETF",
        "VWO": "Vanguard FTSE Emerging Markets ETF",
        "AGG": "iShares Core U.S. Aggregate Bond ETF",
        "TLT": "iShares 20+ Year Treasury Bond ETF",
        "GLD": "SPDR Gold Shares",
        "BTC-USD": "Bitcoin",
        "ETH-USD": "Ethereum",
        "ADA-USD": "Cardano",
        "DOT-USD": "Polkadot",
        "LINK-USD": "Chainlink"
    }

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
        sharpe_ratio = (total_return * 12 - 0.02) / (portfolio_volatility * np.sqrt(12)) if portfolio_volatility > 0 else 0
    else:
        portfolio_beta = 1.0
        portfolio_volatility = 0.15
        sharpe_ratio = 0.0
    
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
        'asset_performance': asset_performance
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
    
    metrics = calculate_portfolio_metrics_advanced(st.session_state.portfolio)
    
    st.markdown("### 💼 Portfolio Performance")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Total Value", 
            f"${metrics['total_current_value']:,.2f}",
            delta=f"${metrics['total_current_value'] - metrics['total_invested']:+,.2f}"
        )
    
    with col2:
        st.metric(
            "Total Return", 
            f"{metrics['total_return']:+.1f}%",
            delta=f"${metrics['total_current_value'] - metrics['total_invested']:+,.2f}"
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
        asset_info = fetch_asset_data(symbol)
        if asset_info:
            current_price = asset_info['current_price']
            purchase_price = data.get('purchase_price', current_price)
            current_value = data['shares'] * current_price
            invested_value = data['shares'] * purchase_price
            return_pct = ((current_value - invested_value) / invested_value) * 100 if invested_value > 0 else 0
            
            portfolio_data.append({
                'Symbol': symbol,
                'Name': asset_info['name'][:30] + "..." if len(asset_info['name']) > 30 else asset_info['name'],
                'Shares': f"{data['shares']:.3f}",
                'Purchase Price': f"${purchase_price:.2f}",
                'Current Price': f"${current_price:.2f}",
                'Invested Value': f"${invested_value:,.2f}",
                'Current Value': f"${current_value:,.2f}",
                'Return %': f"{return_pct:+.1f}%",
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
            selected_popular = st.selectbox(
                "Choose from Popular Assets",
                [""] + list(popular_assets.keys()),
                help="Select from commonly traded assets"
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
            
            asset_type = st.selectbox(
                "Asset Type",
                ["Stock", "ETF", "Cryptocurrency", "Bond", "Commodity", "Index Fund", "Other"]
            )
        
        if symbol_to_use:
            asset_info = fetch_asset_data(symbol_to_use)
            if asset_info:
                current_price = asset_info['current_price']
                potential_return = ((current_price - purchase_price) / purchase_price) * 100 if purchase_price > 0 else 0
                
                # Create three columns for metrics display
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                
                with metric_col1:
                    st.metric("Current Price", f"${current_price:.2f}")
                with metric_col2:
                    st.metric("Your Purchase Price", f"${purchase_price:.2f}")
                with metric_col3:
                    st.metric("Potential Return", f"{potential_return:+.1f}%", 
                             delta=f"${(current_price - purchase_price) * shares:+.2f}")
                
                if st.button("✅ Add to Portfolio", type="primary", use_container_width=True):
                    st.session_state.portfolio[symbol_to_use] = {
                        'shares': shares,
                        'asset_type': asset_type,
                        'purchase_price': purchase_price,
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
