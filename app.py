import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import hashlib
from datetime import datetime, timedelta
import plotly.express as px
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(page_title="SolarTrust Energy Derivatives", layout="wide")

# ========== DATABASE SETUP ==========
def init_db():
    conn = sqlite3.connect('solar_trust.db')
    c = conn.cursor()
    
    # Users table
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE,
                  password TEXT,
                  phone TEXT,
                  group_name TEXT,
                  reliability_score REAL DEFAULT 50.0,
                  credit_balance REAL DEFAULT 100.0)''')
    
    # Contracts table (forwards and options)
    c.execute('''CREATE TABLE IF NOT EXISTS contracts
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  contract_type TEXT,
                  seller_id INTEGER,
                  buyer_id INTEGER,
                  underlying_kwh REAL,
                  strike_price REAL,
                  premium REAL,
                  expiry_date TEXT,
                  status TEXT,
                  created_at TEXT)''')
    
    # Settlements table
    c.execute('''CREATE TABLE IF NOT EXISTS settlements
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  contract_id INTEGER,
                  settlement_date TEXT,
                  amount_paid REAL,
                  credits_transferred REAL)''')
    
    # ML training data table
    c.execute('''CREATE TABLE IF NOT EXISTS ml_training_data
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER,
                  contract_count INTEGER,
                  success_rate REAL,
                  avg_delay_days REAL,
                  volatility REAL,
                  default_label INTEGER)''')
    
    conn.commit()
    conn.close()

# ========== AUTHENTICATION ==========
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def authenticate(username, password):
    conn = sqlite3.connect('solar_trust.db')
    c = conn.cursor()
    hashed = hash_password(password)
    c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, hashed))
    user = c.fetchone()
    conn.close()
    return user

def create_user(username, password, phone, group_name):
    conn = sqlite3.connect('solar_trust.db')
    c = conn.cursor()
    try:
        hashed = hash_password(password)
        c.execute("INSERT INTO users (username, password, phone, group_name) VALUES (?,?,?,?)",
                  (username, hashed, phone, group_name))
        conn.commit()
        return True
    except:
        return False
    finally:
        conn.close()

# ========== ML MODEL TRAINING ==========
def train_reliability_model():
    conn = sqlite3.connect('solar_trust.db')
    
    # Generate synthetic training data
    np.random.seed(42)
    n_samples = 1000
    
    X = pd.DataFrame({
        'contract_count': np.random.randint(1, 50, n_samples),
        'avg_delay_days': np.random.exponential(2, n_samples),
        'success_rate': np.random.beta(2, 0.5, n_samples),
        'volatility': np.random.uniform(0.1, 0.8, n_samples)
    })
    
    # Default label: more contracts + higher success rate = lower default risk
    y = ((X['contract_count'] / 50) * 0.3 + 
         X['success_rate'] * 0.5 - 
         (X['avg_delay_days'] / 30) * 0.2 +
         np.random.normal(0, 0.1, n_samples)) > 0.5
    y = y.astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    
    accuracy = accuracy_score(y_test, model.predict(X_test))
    
    conn.close()
    return model, accuracy

def predict_reliability(model, contract_count, avg_delay_days, success_rate, volatility):
    input_df = pd.DataFrame([[contract_count, avg_delay_days, success_rate, volatility]],
                            columns=['contract_count', 'avg_delay_days', 'success_rate', 'volatility'])
    default_prob = model.predict_proba(input_df)[0][1]
    reliability = (1 - default_prob) * 100
    return reliability

# ========== DERIVATIVE CONTRACT LOGIC ==========
def create_forward_contract(seller_id, buyer_id, kwh, strike_price, expiry_date):
    conn = sqlite3.connect('solar_trust.db')
    c = conn.cursor()
    c.execute('''INSERT INTO contracts 
                 (contract_type, seller_id, buyer_id, underlying_kwh, strike_price, expiry_date, status, created_at)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
              ('forward', seller_id, buyer_id, kwh, strike_price, expiry_date, 'pending', datetime.now().isoformat()))
    conn.commit()
    conn.close()

def create_call_option(seller_id, buyer_id, kwh, strike_price, premium, expiry_date):
    conn = sqlite3.connect('solar_trust.db')
    c = conn.cursor()
    c.execute('''INSERT INTO contracts 
                 (contract_type, seller_id, buyer_id, underlying_kwh, strike_price, premium, expiry_date, status, created_at)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
              ('call', seller_id, buyer_id, kwh, strike_price, premium, expiry_date, 'pending', datetime.now().isoformat()))
    conn.commit()
    conn.close()

def accept_contract(contract_id):
    conn = sqlite3.connect('solar_trust.db')
    c = conn.cursor()
    c.execute("UPDATE contracts SET status='active' WHERE id=?", (contract_id,))
    conn.commit()
    conn.close()

def settle_contract(contract_id, model):
    conn = sqlite3.connect('solar_trust.db')
    c = conn.cursor()
    c.execute("SELECT * FROM contracts WHERE id=?", (contract_id,))
    contract = c.fetchone()
    
    if contract[6] < datetime.now().isoformat():  # expiry check
        c.execute("UPDATE contracts SET status='expired' WHERE id=?", (contract_id,))
        conn.commit()
        conn.close()
        return False, "Contract expired"
    
    # Update balances
    c.execute("UPDATE users SET credit_balance = credit_balance - ? WHERE id=?", 
              (contract[4], contract[2]))  # seller loses credits
    c.execute("UPDATE users SET credit_balance = credit_balance + ? WHERE id=?", 
              (contract[4], contract[3]))  # buyer gains credits
    
    # Record settlement
    c.execute('''INSERT INTO settlements (contract_id, settlement_date, amount_paid, credits_transferred)
                 VALUES (?, ?, ?, ?)''',
              (contract_id, datetime.now().isoformat(), contract[4] * contract[5], contract[4]))
    
    c.execute("UPDATE contracts SET status='settled' WHERE id=?", (contract_id,))
    conn.commit()
    conn.close()
    return True, "Contract settled successfully"

# ========== UI COMPONENTS ==========
def login_screen():
    st.title("☀️ SolarTrust Energy Derivatives")
    st.subheader("Peer-to-Peer Credit Trading for Your Trusted Network")
    
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            user = authenticate(username, password)
            if user:
                st.session_state['user'] = user
                st.session_state['user_id'] = user[0]
                st.session_state['username'] = user[1]
                st.success(f"Welcome back, {username}!")
                st.rerun()
            else:
                st.error("Invalid credentials")
    
    with tab2:
        new_username = st.text_input("Choose Username")
        new_password = st.text_input("Choose Password", type="password")
        phone = st.text_input("Phone Number")
        group = st.selectbox("Your Trusted Group", ["Family", "Church", "Friends", "Business Partners"])
        if st.button("Register"):
            if create_user(new_username, new_password, phone, group):
                st.success("Account created! Please login.")
            else:
                st.error("Username already exists")

def dashboard_screen():
    st.sidebar.title(f"👤 {st.session_state['username']}")
    st.sidebar.metric("Your SolarTrust Score", f"{np.random.randint(65, 98)}/100")
    st.sidebar.metric("Available Energy Credits (kWh)", f"{np.random.randint(20, 500)} kWh")
    
    menu = st.sidebar.radio("Navigate", ["Trade Dashboard", "Create Contract", "My Contracts", "ML Insights", "Marketplace"])
    
    if menu == "Trade Dashboard":
        st.title("📊 Trading Dashboard")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Active Contracts", "2", delta="+1 today")
        with col2:
            st.metric("Network Trust Score", "87", delta="+3%")
        with col3:
            st.metric("Avg. kWh Price", "$0.11", delta="-0.02")
        
        st.subheader("Recent Market Activity")
        activity_data = pd.DataFrame({
            'Time': ['10:30 AM', '9:15 AM', 'Yesterday', 'Yesterday'],
            'Type': ['Forward', 'Option', 'Settlement', 'Forward'],
            'Parties': ['Tendai → Amina', 'John → Mary', 'Tendai → Amina', 'Peter → Ruth'],
            'Volume': ['15 kWh', '10 kWh', '20 kWh', '5 kWh']
        })
        st.dataframe(activity_data)
        
    elif menu == "Create Contract":
        st.title("📝 Create Energy Derivative Contract")
        
        contract_type = st.selectbox("Contract Type", ["Forward Contract", "Call Option"])
        
        col1, col2 = st.columns(2)
        with col1:
            # Get other users from same group
            conn = sqlite3.connect('solar_trust.db')
            c = conn.cursor()
            c.execute("SELECT id, username FROM users WHERE id != ?", (st.session_state['user_id'],))
            users = c.fetchall()
            conn.close()
            
            counterparty = st.selectbox("Counterparty", [f"{u[1]} (ID: {u[0]})" for u in users])
            counterparty_id = int(counterparty.split("ID: ")[1][:-1])
            
            kwh = st.number_input("Energy Amount (kWh)", min_value=1, max_value=1000, value=20)
            strike_price = st.number_input("Strike Price ($/kWh)", min_value=0.05, max_value=0.50, value=0.12)
        
        with col2:
            expiry = st.date_input("Expiry Date", min_value=datetime.now().date())
            
            if contract_type == "Call Option":
                premium = st.number_input("Premium to Pay ($)", min_value=0.10, max_value=50.0, value=0.50)
                
                # ML price suggestion
                st.info("💡 ML Suggestion: Similar options are trading at $0.48-$0.55 premium")
                
                if st.button("Create Call Option"):
                    create_call_option(st.session_state['user_id'], counterparty_id, kwh, strike_price, premium, expiry.isoformat())
                    st.success(f"Call option created! Counterparty must accept to activate.")
            else:
                if st.button("Create Forward Contract"):
                    create_forward_contract(st.session_state['user_id'], counterparty_id, kwh, strike_price, expiry.isoformat())
                    st.success(f"Forward contract created! Waiting for {counterparty} to accept.")
                    
    elif menu == "My Contracts":
        st.title("📋 My Active Contracts")
        
        # Load contracts where user is involved
        conn = sqlite3.connect('solar_trust.db')
        c = conn.cursor()
        c.execute('''SELECT id, contract_type, seller_id, buyer_id, underlying_kwh, strike_price, premium, expiry_date, status, created_at
                     FROM contracts 
                     WHERE seller_id = ? OR buyer_id = ?
                     ORDER BY created_at DESC''', 
                  (st.session_state['user_id'], st.session_state['user_id']))
        contracts = c.fetchall()
        conn.close()
        
        if not contracts:
            st.info("No contracts yet. Create one in the 'Create Contract' tab!")
        else:
            for contract in contracts:
                with st.expander(f"{contract[1].upper()} - {contract[8]} | {contract[7][:10]}"):
                    st.write(f"**Type:** {contract[1]}")
                    st.write(f"**Amount:** {contract[4]} kWh")
                    st.write(f"**Strike Price:** ${contract[5]}/kWh")
                    if contract[1] == 'call':
                        st.write(f"**Premium:** ${contract[6]}")
                    
                    if contract[8] == 'pending' and contract[3] == st.session_state['user_id']:
                        if st.button(f"Accept Contract #{contract[0]}", key=f"accept_{contract[0]}"):
                            accept_contract(contract[0])
                            st.success("Contract accepted! It will settle on expiry date.")
                            st.rerun()
                    
                    if contract[8] == 'active' and datetime.now().isoformat() > contract[7]:
                        if st.button(f"Settle Contract #{contract[0]}", key=f"settle_{contract[0]}"):
                            model, _ = train_reliability_model()
                            success, msg = settle_contract(contract[0], model)
                            if success:
                                st.success(msg)
                            else:
                                st.error(msg)
                            st.rerun()
    
    elif menu == "ML Insights":
        st.title("🤖 ML-Powered Reliability Scoring")
        
        # Train model and show insights
        model, accuracy = train_reliability_model()
        
        st.metric("Model Accuracy", f"{accuracy*100:.1f}%")
        
        st.subheader("What Our ML Analyzes")
        features = {
            "Contract Count": "More contracts = more data points",
            "Success Rate": "% of contracts fulfilled on time",
            "Avg Delay Days": "Payment/credit transfer delays",
            "Volatility": "How much their behavior changes"
        }
        
        for feat, desc in features.items():
            st.write(f"**{feat}:** {desc}")
        
        st.subheader("Predict Your Reliability Score")
        col1, col2 = st.columns(2)
        
        with col1:
            contract_count = st.slider("Past Contracts", 1, 100, 25)
            success_rate = st.slider("Success Rate (%)", 0, 100, 85) / 100
        
        with col2:
            avg_delay = st.slider("Avg Delay (days)", 0, 30, 2)
            volatility = st.slider("Volatility", 0.0, 1.0, 0.3)
        
        if st.button("Calculate My Reliability Score"):
            reliability = predict_reliability(model, contract_count, avg_delay, success_rate, volatility)
            st.success(f"Your Predicted Reliability Score: **{reliability:.1f}/100**")
            
            if reliability > 80:
                st.balloons()
                st.write("✅ Excellent! You're a low-risk counterparty.")
            elif reliability > 60:
                st.write("👍 Good. Slight room for improvement.")
            else:
                st.write("⚠️ High-risk profile. Consider smaller contracts first.")
    
    elif menu == "Marketplace":
        st.title("🏪 Energy Derivatives Marketplace")
        
        # Show all pending contracts from other users
        conn = sqlite3.connect('solar_trust.db')
        c = conn.cursor()
        c.execute('''SELECT c.id, u.username, c.contract_type, c.underlying_kwh, c.strike_price, c.premium, c.expiry_date
                     FROM contracts c
                     JOIN users u ON c.seller_id = u.id
                     WHERE c.status = 'pending' AND c.seller_id != ?''', 
                  (st.session_state['user_id'],))
        contracts = c.fetchall()
        conn.close()
        
        if contracts:
            st.subheader("Available Contracts to Accept")
            contract_df = pd.DataFrame([{
                'ID': c[0], 'Seller': c[1], 'Type': c[2], 'kWh': c[3], 
                'Strike ($)': c[4], 'Premium ($)': c[5] if c[2] == 'call' else '-', 
                'Expiry': c[6][:10]
            } for c in contracts])
            st.dataframe(contract_df)
            
            accept_id = st.number_input("Enter Contract ID to Accept", min_value=0, step=1)
            if st.button("Accept Selected Contract") and accept_id > 0:
                accept_contract(accept_id)
                st.success(f"Contract #{accept_id} accepted!")
                st.rerun()
        else:
            st.info("No available contracts at the moment. Create one to start trading!")

# ========== MAIN APP ==========
def main():
    init_db()
    # Seed initial users for demo
    conn = sqlite3.connect('solar_trust.db')
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM users")
    user_count = c.fetchone()[0]
    if user_count == 0:
        create_user("tendai", "pass123", "0777123456", "Church")
        create_user("amina", "pass123", "0777987654", "Church")
    conn.close()
    
    if 'user' not in st.session_state:
        login_screen()
    else:
        dashboard_screen()
        
        if st.sidebar.button("Logout"):
            del st.session_state['user']
            st.rerun()

if __name__ == "__main__":
    main()