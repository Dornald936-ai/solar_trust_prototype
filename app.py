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

# ========== PAGE CONFIG ==========
st.set_page_config(page_title="SolarTrust Energy Derivatives", page_icon="☀️", layout="wide")

# ========== MODERN CSS ==========
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,400;0,500;0,600;0,700;1,400&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .card {
        background: linear-gradient(135deg, #1e2a3a 0%, #0f172a 100%);
        border-radius: 1rem;
        padding: 1.5rem;
        border: 1px solid #334155;
        box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    .card:hover {
        transform: translateY(-4px);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #fbbf24, #f59e0b);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        letter-spacing: 0.05em;
    }
    .stButton button {
        background: linear-gradient(90deg, #f59e0b, #d97706);
        border: none;
        border-radius: 0.5rem;
        font-weight: 600;
        transition: all 0.2s;
    }
    .stButton button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 12px rgba(245,158,11,0.5);
    }
    [data-testid="stSidebar"] {
        background: #0f172a;
        border-right: 1px solid #1e293b;
    }
    .dataframe {
        overflow-x: auto;
        display: block;
    }
    button {
        min-height: 44px;
    }
    @media (max-width: 640px) {
        .card { padding: 0.75rem; }
        .metric-value { font-size: 1.5rem; }
        .stButton button { width: 100%; }
    }
</style>
""", unsafe_allow_html=True)

# ========== DATABASE SETUP ==========
def init_db():
    conn = sqlite3.connect('solar_trust.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE,
                  password TEXT,
                  phone TEXT,
                  group_name TEXT,
                  reliability_score REAL DEFAULT 50.0,
                  credit_balance REAL DEFAULT 100.0)''')
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
    c.execute('''CREATE TABLE IF NOT EXISTS settlements
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  contract_id INTEGER,
                  settlement_date TEXT,
                  amount_paid REAL,
                  credits_transferred REAL)''')
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

# ========== AUTH ==========
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

# ========== ML MODEL ==========
def train_reliability_model():
    conn = sqlite3.connect('solar_trust.db')
    np.random.seed(42)
    n_samples = 1000
    X = pd.DataFrame({
        'contract_count': np.random.randint(1, 50, n_samples),
        'avg_delay_days': np.random.exponential(2, n_samples),
        'success_rate': np.random.beta(2, 0.5, n_samples),
        'volatility': np.random.uniform(0.1, 0.8, n_samples)
    })
    y = ((X['contract_count'] / 50) * 0.3 + X['success_rate'] * 0.5 - (X['avg_delay_days'] / 30) * 0.2 +
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

def explain_reliability(contract_count, avg_delay_days, success_rate, volatility):
    reasons = []
    if contract_count < 10:
        reasons.append("🟡 Few past contracts → limited data")
    if avg_delay_days > 5:
        reasons.append("🔴 Frequent payment delays")
    if success_rate < 0.8:
        reasons.append("🔴 Low contract fulfillment rate")
    if volatility > 0.5:
        reasons.append("🟡 High variability in behavior")
    if not reasons:
        reasons.append("✅ Strong, consistent trading history")
    return reasons

# ========== AUTO SETTLE ==========
def auto_settle_expired_contracts():
    conn = sqlite3.connect('solar_trust.db')
    c = conn.cursor()
    now = datetime.now().isoformat()
    c.execute("SELECT id, seller_id, buyer_id, underlying_kwh, strike_price FROM contracts WHERE status='active' AND expiry_date < ?", (now,))
    expired = c.fetchall()
    for contract in expired:
        c.execute("UPDATE users SET credit_balance = credit_balance - ? WHERE id=?", (contract[3], contract[1]))
        c.execute("UPDATE users SET credit_balance = credit_balance + ? WHERE id=?", (contract[3], contract[2]))
        c.execute("UPDATE contracts SET status='settled_auto' WHERE id=?", (contract[0],))
        c.execute("INSERT INTO settlements (contract_id, settlement_date, amount_paid, credits_transferred) VALUES (?,?,?,?)",
                  (contract[0], now, contract[3]*contract[4], contract[3]))
    conn.commit()
    conn.close()
    return len(expired)

# ========== CONTRACT LOGIC ==========
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
    if contract[6] < datetime.now().isoformat():
        c.execute("UPDATE contracts SET status='expired' WHERE id=?", (contract_id,))
        conn.commit()
        conn.close()
        return False, "Contract expired"
    c.execute("UPDATE users SET credit_balance = credit_balance - ? WHERE id=?", (contract[4], contract[2]))
    c.execute("UPDATE users SET credit_balance = credit_balance + ? WHERE id=?", (contract[4], contract[3]))
    c.execute('''INSERT INTO settlements (contract_id, settlement_date, amount_paid, credits_transferred)
                 VALUES (?, ?, ?, ?)''',
              (contract_id, datetime.now().isoformat(), contract[4] * contract[5], contract[4]))
    c.execute("UPDATE contracts SET status='settled' WHERE id=?", (contract_id,))
    conn.commit()
    conn.close()
    return True, "Contract settled successfully"

# ========== LOGIN SCREEN ==========
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

# ========== DASHBOARD SCREEN (MODERN) ==========
def dashboard_screen():
    # Auto-settle expired contracts
    settled_count = auto_settle_expired_contracts()
    if settled_count > 0:
        st.toast(f"✅ {settled_count} contracts automatically settled!", icon="🤖")

    # Sidebar user info
    st.sidebar.image("https://img.icons8.com/fluency/96/solar-panel.png", width=60)
    st.sidebar.title(f"☀️ {st.session_state['username']}")
    conn = sqlite3.connect('solar_trust.db')
    c = conn.cursor()
    c.execute("SELECT credit_balance, reliability_score FROM users WHERE id=?", (st.session_state['user_id'],))
    user_data = c.fetchone()
    credit_balance = user_data[0] if user_data else 100
    reliability = user_data[1] if user_data else 50
    conn.close()
    st.sidebar.metric("🔋 Energy Credits", f"{credit_balance:.0f} kWh")
    st.sidebar.metric("🏆 Trust Score", f"{reliability:.0f}/100", delta="+5" if reliability > 70 else None)

    menu = st.sidebar.radio(
        "Navigate",
        ["📊 Dashboard", "📝 New Contract", "📄 My Contracts", "🧠 ML Insights", "🏪 Marketplace", "👥 Group Hub", "💳 Wallet"],
        format_func=lambda x: x
    )

    # ------------------- DASHBOARD -------------------
    if menu == "📊 Dashboard":
        st.title("⚡ Energy Derivatives Dashboard")
        st.caption(f"Welcome back, {st.session_state['username']}. Ready to trade?")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("<div class='card'><div class='metric-label'>ACTIVE CONTRACTS</div><div class='metric-value'>3</div></div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<div class='card'><div class='metric-label'>YOUR TRUST SCORE</div><div class='metric-value'>{reliability:.0f}</div></div>", unsafe_allow_html=True)
        with col3:
            st.markdown(f"<div class='card'><div class='metric-label'>AVAILABLE CREDITS</div><div class='metric-value'>{credit_balance:.0f} kWh</div></div>", unsafe_allow_html=True)
        with col4:
            st.markdown("<div class='card'><div class='metric-label'>AVG kWh PRICE</div><div class='metric-value'>$0.11</div></div>", unsafe_allow_html=True)
        st.subheader("📈 Market Trends")
        trend_data = pd.DataFrame({'Day': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri'], 'Avg Price': [0.125, 0.118, 0.112, 0.109, 0.115]})
        fig = px.line(trend_data, x='Day', y='Avg Price', title='Energy Credit Price (USD/kWh)', markers=True)
        fig.update_layout(plot_bgcolor='#0f172a', paper_bgcolor='#0f172a', font_color='white')
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("🕒 Recent Trades")
        recent = pd.DataFrame({'Time': ['10:30 AM', '09:15 AM', 'Yesterday'], 'Type': ['Forward', 'Option', 'Settlement'], 'Parties': ['Tendai → Amina', 'John → Mary', 'Tendai → Amina'], 'Volume (kWh)': [20, 15, 20]})
        st.dataframe(recent, use_container_width=True)

    # ------------------- NEW CONTRACT -------------------
    elif menu == "📝 New Contract":
        st.title("✍️ Create New Derivative")
        contract_type = st.selectbox("Contract Type", ["Forward Contract", "Call Option"], format_func=lambda x: "📜 Forward (future sale)" if x=="Forward Contract" else "🎫 Call Option (right to buy)")
        col1, col2 = st.columns(2)
        with col1:
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
                if st.button("Create Call Option"):
                    create_call_option(st.session_state['user_id'], counterparty_id, kwh, strike_price, premium, expiry.isoformat())
                    st.success("Call option created! Counterparty must accept.")
            else:
                if st.button("Create Forward Contract"):
                    create_forward_contract(st.session_state['user_id'], counterparty_id, kwh, strike_price, expiry.isoformat())
                    st.success("Forward contract created! Waiting for counterparty to accept.")
        st.info("💡 ML Tip: Current market price is $0.11/kWh. Consider strike price within ±10% for fair trade.")

    # ------------------- MY CONTRACTS -------------------
    elif menu == "📄 My Contracts":
        st.title("📋 Your Portfolio")
        conn = sqlite3.connect('solar_trust.db')
        c = conn.cursor()
        c.execute('''SELECT id, contract_type, seller_id, buyer_id, underlying_kwh, strike_price, premium, expiry_date, status, created_at
                     FROM contracts WHERE seller_id = ? OR buyer_id = ? ORDER BY created_at DESC''',
                  (st.session_state['user_id'], st.session_state['user_id']))
        contracts = c.fetchall()
        conn.close()
        if not contracts:
            st.info("No contracts yet. Create one in the 'New Contract' tab!")
        else:
            for contract in contracts:
                with st.expander(f"{contract[1].upper()} - {contract[8]} | expires {contract[7][:10]}"):
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

    # ------------------- ML INSIGHTS -------------------
    elif menu == "🧠 ML Insights":
        st.title("🤖 AI Reliability Engine")
        model, accuracy = train_reliability_model()
        st.metric("Model Accuracy", f"{accuracy*100:.1f}%")
        st.subheader("What Our ML Analyzes")
        features = {"Contract Count": "More contracts = more data points", "Success Rate": "% of contracts fulfilled on time", "Avg Delay Days": "Payment/credit transfer delays", "Volatility": "How much their behavior changes"}
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
            st.subheader("🔍 Why this score?")
            reasons = explain_reliability(contract_count, avg_delay, success_rate, volatility)
            for r in reasons:
                st.write(r)

    # ------------------- MARKETPLACE -------------------
    elif menu == "🏪 Marketplace":
        st.title("🏪 Public Offers")
        conn = sqlite3.connect('solar_trust.db')
        c = conn.cursor()
        c.execute('''SELECT c.id, u.username, c.contract_type, c.underlying_kwh, c.strike_price, c.premium, c.expiry_date
                     FROM contracts c JOIN users u ON c.seller_id = u.id
                     WHERE c.status = 'pending' AND c.seller_id != ?''', (st.session_state['user_id'],))
        offers = c.fetchall()
        conn.close()
        if offers:
            df_offers = pd.DataFrame([{'ID': o[0], 'Seller': o[1], 'Type': o[2], 'kWh': o[3], 'Strike ($)': o[4], 'Premium ($)': o[5] if o[2]=='call' else '-', 'Expiry': o[6][:10]} for o in offers])
            st.dataframe(df_offers, use_container_width=True)
            accept_id = st.number_input("Enter Contract ID to Accept", min_value=0, step=1)
            if st.button("Accept Selected Contract") and accept_id > 0:
                accept_contract(accept_id)
                st.success(f"Contract #{accept_id} accepted!")
                st.rerun()
        else:
            st.info("No pending offers. Create one in 'New Contract' tab!")

    # ------------------- GROUP HUB -------------------
    elif menu == "👥 Group Hub":
        st.title("👥 Your Trust Network")
        conn = sqlite3.connect('solar_trust.db')
        c = conn.cursor()
        c.execute("SELECT username, reliability_score, credit_balance FROM users WHERE id != ?", (st.session_state['user_id'],))
        members = c.fetchall()
        conn.close()
        if members:
            df_members = pd.DataFrame(members, columns=['Name', 'Trust Score', 'Credits (kWh)'])
            st.dataframe(df_members, use_container_width=True)
            best = df_members.loc[df_members['Trust Score'].idxmax(), 'Name']
            st.success(f"🏆 Top Trader: {best}")
        else:
            st.info("No other members yet. Invite friends to trade!")

    # ------------------- WALLET -------------------
    elif menu == "💳 Wallet":
        st.title("💳 My Energy Wallet")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Current Balance", f"{credit_balance} kWh")
            st.metric("Value at Market Price", f"${credit_balance * 0.11:.2f}")
        with col2:
            st.subheader("Receive Credits")
            st.code(f"SOLAR:{st.session_state['username']}:{st.session_state['user_id']}", language="text")
            st.caption("Share this code to receive energy credits from friends.")
        st.button("💸 Request Payment via EcoCash (Demo)", help="In production, this would open EcoCash USSD.")

# ========== MAIN ==========
def main():
    init_db()
    # Seed demo users if none exist
    conn = sqlite3.connect('solar_trust.db')
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM users")
    if c.fetchone()[0] == 0:
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