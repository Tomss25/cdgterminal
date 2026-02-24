import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import yfinance as yf
import streamlit.components.v1 as components

# =============================================================================
# 1. CORE CONFIG & UI INJECTION (L'Ecosistema Visivo)
# =============================================================================
st.set_page_config(page_title="Alpha Terminal V8.0", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    /* Sfondo globale e testo principale */
    .stApp { background-color: #0A0E17; color: #D1D5DB; font-family: 'Inter', -apple-system, sans-serif; }
    
    /* Headers (Titoli) */
    h1, h2, h3, h4 { color: #F8B400 !important; font-family: 'SF Pro Display', sans-serif; font-weight: 600; letter-spacing: -0.5px; }
    
    /* Stile dei pulsanti */
    .stButton>button { background-color: #1E293B; color: #60A5FA; border: 1px solid #3B82F6; border-radius: 4px; font-weight: bold; transition: all 0.2s; width: 100%; }
    .stButton>button:hover { background-color: #3B82F6; color: #FFFFFF; border-color: #60A5FA; }
    
    /* Input testuali */
    .stTextArea>div>div>textarea, .stTextInput>div>div>input { background-color: #111827; color: #10B981; font-family: 'Courier New', Courier, monospace; border: 1px solid #374151; border-radius: 4px; }
    .stTextArea>div>div>textarea:focus, .stTextInput>div>div>input:focus { border-color: #F8B400; box-shadow: none; }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { background-color: #0F172A; border-bottom: 2px solid #1E293B; }
    .stTabs [data-baseweb="tab"] { color: #94A3B8; font-weight: 600; padding-top: 1rem; padding-bottom: 1rem; }
    .stTabs [aria-selected="true"] { color: #F8B400; border-bottom-color: #F8B400; }
    
    /* Avvisi */
    .stAlert { background-color: #111827; border: 1px solid #374151; color: #D1D5DB; }
</style>
""", unsafe_allow_html=True)

st.title("ALPHA TERMINAL // PROPRIETARY ENGINE")
st.markdown("---")

# =============================================================================
# 2. SESSION STATE (La Memoria Persistente)
# =============================================================================
if 'df_stock' not in st.session_state:
    st.session_state['df_stock'] = None
if 'df_bond' not in st.session_state:
    st.session_state['df_bond'] = None
if 'api_calls' not in st.session_state:
    st.session_state['api_calls'] = 0

# =============================================================================
# 3. MOTORI DI ESTRAZIONE DATI (Niente side-effects qui dentro)
# =============================================================================
@st.cache_data(ttl=600)
def fetch_stock_data(symbol):
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        pe = info.get('trailingPE', np.nan)
        roe = info.get('returnOnEquity', np.nan)
        rev_growth = info.get('revenueGrowth', np.nan)
        debt_eq = info.get('debtToEquity', np.nan)
        
        return {
            'Symbol': symbol,
            'Price': info.get('currentPrice', np.nan),
            'P/E': pe,
            'ROE %': roe * 100 if pd.notnull(roe) else np.nan,
            'Rev Growth %': rev_growth * 100 if pd.notnull(rev_growth) else np.nan,
            'Debt/Eq': debt_eq,
            'Status': 'OK'
        }
    except Exception as e:
        return {'Symbol': symbol, 'Status': 'ERRORE', 'Price': np.nan, 'P/E': np.nan, 'ROE %': np.nan, 'Rev Growth %': np.nan, 'Debt/Eq': np.nan}

@st.cache_data(ttl=600)
def fetch_bond_data(isin):
    bond_db = {
        'FR0013327988': {'Price':101.2, 'YTM':3.2, 'Duration':2.1, 'Rating':'BBB+', 'Issuer':'Capgemini'},
        'XS2486589596': {'Price':100.5, 'YTM':2.8, 'Duration':1.8, 'Rating':'A-', 'Issuer':'HSBC'},
        'XS2388941077': {'Price':99.8, 'YTM':3.5, 'Duration':2.5, 'Rating':'BBB', 'Issuer':'Acciona'}
    }
    if isin in bond_db:
        data = bond_db[isin]
        data['ISIN'] = isin
        data['Status'] = 'OK'
        return data
    else:
        return {'ISIN': isin, 'Status': 'N/A', 'Price': np.nan, 'YTM': np.nan, 'Duration': np.nan, 'Rating': 'N/A', 'Issuer': 'N/A'}

# =============================================================================
# 4. INTERFACCIA A TAB
# =============================================================================
tab_home, tab_stock, tab_bond, tab_portfolio = st.tabs(["[🏠] MACRO DASHBOARD", "[EQ] EQUITY SCREENER", "[FI] FIXED INCOME", "[PRT] PORTFOLIO MATRIX"])

with tab_home:
    st.subheader("➤ GLOBAL MACRO & SENTIMENT (EXTERNAL FEED)")
    st.warning("⚠️ RETAIL FEED: Dati puramente visivi e isolati. Non interagiscono con il motore quantitativo backend.")
    
    col_dx, col_sx = st.columns([1.5, 1])
    
    with col_dx:
        st.markdown("**CALENDARIO ECONOMICO**")
        cal_html = """
        <iframe src="https://sslecal2.investing.com?ecoDayBackground=%230A0E17&defaultFont=%23D1D5DB&borderColor=%231E293B&ecoDayFontColor=%23F8B400&columns=exc_flags,exc_currency,exc_importance,exc_actual,exc_forecast,exc_previous&features=datepicker,timezone&countries=25,32,6,37,72,22,17,39,14,10,35,43,56,36,110,11,26,12,4,5&calType=week&timeZone=8&lang=1" width="100%" height="520" frameborder="0" allowtransparency="true" marginwidth="0" marginheight="0"></iframe>
        """
        components.html(cal_html, height=530)

    with col_sx:
        st.markdown("**LIVE CURRENCY CROSS RATES**")
        fx_html = """
        <iframe src="https://www.widgets.investing.com/live-currency-cross-rates?theme=darkTheme&roundedCorners=true&pairs=1,3,2,4,7,5,8,6,9,10,11" width="100%" height="220" frameborder="0" allowtransparency="true" marginwidth="0" marginheight="0"></iframe>
        """
        components.html(fx_html, height=230)
        
        st.markdown("**TOP CRYPTOCURRENCIES**")
        crypto_html = """
        <iframe src="https://www.widgets.investing.com/top-cryptocurrencies?theme=darkTheme&roundedCorners=true" width="100%" height="220" frameborder="0" allowtransparency="true" marginwidth="0" marginheight="0"></iframe>
        """
        components.html(crypto_html, height=230)

with tab_stock:
    st.subheader("➤ EQUITY COMMAND LINE")
    symbols_input = st.text_area("INSERISCI TICKER (Separati da a capo)", "AAPL\nMSFT\nENI.MI", height=100)
    
    if st.button("EXECUTE EQUITY QUERY"):
        symbols = [s.strip().upper() for s in symbols_input.split('\n') if s.strip()]
        st.session_state['api_calls'] += len(symbols)
        results = [fetch_stock_data(sym) for sym in symbols]
        
        st.session_state['df_stock'] = pd.DataFrame(results)
        st.success("Query eseguita. Dati bloccati in RAM.")
        
    if st.session_state['df_stock'] is not None:
        st.dataframe(st.session_state['df_stock'].style.format(na_rep='N/A', precision=2), use_container_width=True)

with tab_bond:
    st.subheader("➤ FIXED INCOME COMMAND LINE")
    isins_input = st.text_area("INSERISCI ISIN (Separati da a capo)", "XS2486589596\nFR0013327988\nIT000FAKE000", height=100)
    
    if st.button("EXECUTE BOND QUERY"):
        isins = [i.strip().upper() for i in isins_input.split('\n') if i.strip()]
        results = [fetch_bond_data(isin) for isin in isins]
        
        st.session_state['df_bond'] = pd.DataFrame(results)
        st.success("Query eseguita. Dati bloccati in RAM.")

    if st.session_state['df_bond'] is not None:
        st.dataframe(st.session_state['df_bond'].style.apply(lambda x: ['background: #450a0a; color: #fca5a5' if x['Status'] == 'N/A' else '' for i in x], axis=1), use_container_width=True)

with tab_portfolio:
    st.subheader("➤ CROSS-ASSET MATRIX")
    
    if st.session_state['df_stock'] is not None and st.session_state['df_bond'] is not None:
        df_s = st.session_state['df_stock'].copy()
        df_s['Asset Class'] = 'Equity'
        df_s['Identifier'] = df_s['Symbol']
        df_s['Yield/Growth'] = df_s['ROE %']
        
        df_b = st.session_state['df_bond'].copy()
        df_b['Asset Class'] = 'Fixed Income'
        df_b['Identifier'] = df_b['ISIN']
        df_b['Yield/Growth'] = df_b['YTM']
        
        portfolio = pd.concat([
            df_s[['Identifier', 'Asset Class', 'Price', 'Yield/Growth', 'Status']], 
            df_b[['Identifier', 'Asset Class', 'Price', 'Yield/Growth', 'Status']]
        ], ignore_index=True)
        
        portfolio_clean = portfolio[portfolio['Status'] == 'OK'].dropna(subset=['Price', 'Yield/Growth'])
        
        st.dataframe(portfolio, use_container_width=True)
        
        if not portfolio_clean.empty:
            fig = px.scatter(
                portfolio_clean, x='Price', y='Yield/Growth', color='Asset Class', hover_name='Identifier',
                color_discrete_sequence=['#3B82F6', '#10B981']
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#D1D5DB', family="Courier New"),
                xaxis=dict(showgrid=True, gridcolor='#1F2937', title="Prezzo di Mercato"),
                yaxis=dict(showgrid=True, gridcolor='#1F2937', title="Rendimento (YTM / ROE)")
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ SYSTEM HALT: Popolare entrambi i database (EQ e FI) dai rispettivi tab prima di generare la matrice.")

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.caption(f"SYSTEM STATUS: ONLINE | CACHE: ACTIVE | TOTAL API HITS (SESSION): {st.session_state['api_calls']}")