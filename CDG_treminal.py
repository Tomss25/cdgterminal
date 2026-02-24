import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import yfinance as yf
import streamlit.components.v1 as components
from datetime import datetime, timedelta

# =============================================================================
# 1. CORE CONFIG & UI INJECTION
# =============================================================================
st.set_page_config(page_title="Alpha Terminal V9.1 - X-Ray Edition", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    /* Sfondo globale e testo principale */
    .stApp { background-color: #0A0E17; color: #D1D5DB; font-family: 'Inter', -apple-system, sans-serif; }
    h1, h2, h3, h4 { color: #F8B400 !important; font-family: 'SF Pro Display', sans-serif; font-weight: 600; letter-spacing: -0.5px; }
    
    /* Stile dei pulsanti */
    .stButton>button { background-color: #1E293B; color: #60A5FA; border: 1px solid #3B82F6; border-radius: 4px; font-weight: bold; transition: all 0.2s; width: 100%; }
    .stButton>button:hover { background-color: #3B82F6; color: #FFFFFF; border-color: #60A5FA; }
    
    /* Input testuali base */
    .stTextArea>div>div>textarea, .stTextInput>div>div>input { background-color: #111827; color: #10B981; font-family: 'Courier New', Courier, monospace; border: 1px solid #374151; border-radius: 4px; }
    .stTextArea>div>div>textarea:focus, .stTextInput>div>div>input:focus { border-color: #F8B400; box-shadow: none; }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { background-color: #0F172A; border-bottom: 2px solid #1E293B; }
    .stTabs [data-baseweb="tab"] { color: #94A3B8; font-weight: 600; padding-top: 1rem; padding-bottom: 1rem; }
    .stTabs [aria-selected="true"] { color: #F8B400; border-bottom-color: #F8B400; }
    .stAlert { background-color: #111827; border: 1px solid #374151; color: #D1D5DB; }

    /* --- LE TUE MODIFICHE ESTETICHE --- */
    /* Forza il colore arancione per le etichette di inserimento testo */
    .stTextInput label p, .stTextArea label p { color: #F8B400 !important; font-weight: bold; }
    
    /* Forza il colore bianco puro per i valori e le etichette delle metriche (Deep Dive) */
    [data-testid="stMetricValue"] div { color: #FFFFFF !important; }
    [data-testid="stMetricLabel"] p { color: #FFFFFF !important; }
</style>
""", unsafe_allow_html=True)

st.title("ALPHA TERMINAL // PROPRIETARY ENGINE")
st.markdown("---")

# =============================================================================
# 2. SESSION STATE
# =============================================================================
if 'df_stock' not in st.session_state:
    st.session_state['df_stock'] = None
if 'df_bond' not in st.session_state:
    st.session_state['df_bond'] = None
if 'api_calls' not in st.session_state:
    st.session_state['api_calls'] = 0

# =============================================================================
# 3. MOTORI DI ESTRAZIONE DATI E VALUTAZIONE
# =============================================================================
@st.cache_data(ttl=600)
def fetch_stock_data(symbol):
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        price = info.get('currentPrice', np.nan)
        pe = info.get('trailingPE', np.nan)
        roe = info.get('returnOnEquity', np.nan)
        rev_growth = info.get('revenueGrowth', np.nan)
        debt_eq = info.get('debtToEquity', np.nan)
        eps = info.get('trailingEps', np.nan)
        
        g_raw = info.get('earningsGrowth', None)
        if g_raw is None: 
            g_raw = rev_growth if pd.notnull(rev_growth) else 0.05
            
        g_base = g_raw * 100 
        Y = 4.4 
        
        if pd.notnull(eps) and eps > 0:
            g_bear = g_base * 0.5
            g_bull = g_base * 1.2
            val_bear = (eps * (8.5 + 2 * g_bear) * 4.4) / Y
            val_base = (eps * (8.5 + 2 * g_base) * 4.4) / Y
            val_bull = (eps * (8.5 + 2 * g_bull) * 4.4) / Y
            mos = ((val_base - price) / val_base) * 100 if pd.notnull(price) and val_base > 0 else np.nan
        else:
            val_bear = val_base = val_bull = mos = np.nan

        return {
            'Symbol': symbol, 'Price': price, 'P/E': pe, 'ROE %': roe * 100 if pd.notnull(roe) else np.nan,
            'Rev Growth %': rev_growth * 100 if pd.notnull(rev_growth) else np.nan, 'Debt/Eq': debt_eq,
            'EPS': eps, 'Val. BEAR': val_bear, 'Val. BASE': val_base, 'Val. BULL': val_bull, 'MoS %': mos, 'Status': 'OK'
        }
    except Exception as e:
        return {'Symbol': symbol, 'Status': 'ERRORE', 'Price': np.nan, 'P/E': np.nan, 'ROE %': np.nan, 'Rev Growth %': np.nan, 'Debt/Eq': np.nan, 'EPS': np.nan, 'Val. BEAR': np.nan, 'Val. BASE': np.nan, 'Val. BULL': np.nan, 'MoS %': np.nan}

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

@st.cache_data(ttl=600)
def fetch_deep_dive(symbol, benchmark='SPY'):
    try:
        ticker = yf.Ticker(symbol)
        bench = yf.Ticker(benchmark)
        info = ticker.info
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        hist_stock = ticker.history(start=start_date, end=end_date)
        hist_bench = bench.history(start=start_date, end=end_date)
        
        ret_stock = ((hist_stock['Close'].iloc[-1] / hist_stock['Close'].iloc[0]) - 1) * 100 if not hist_stock.empty else np.nan
        ret_bench = ((hist_bench['Close'].iloc[-1] / hist_bench['Close'].iloc[0]) - 1) * 100 if not hist_bench.empty else np.nan
        alpha = ret_stock - ret_bench if pd.notnull(ret_stock) and pd.notnull(ret_bench) else np.nan

        target_mean = info.get('targetMeanPrice', np.nan)
        current_price = info.get('currentPrice', np.nan)
        upside_consensus = ((target_mean / current_price) - 1) * 100 if pd.notnull(target_mean) and pd.notnull(current_price) else np.nan
        recommendation = info.get('recommendationKey', 'N/A').replace('_', ' ').upper()
        
        sma50 = info.get('fiftyDayAverage', np.nan)
        sma200 = info.get('twoHundredDayAverage', np.nan)
        high52 = info.get('fiftyTwoWeekHigh', np.nan)
        dist_from_high = ((current_price / high52) - 1) * 100 if pd.notnull(high52) else np.nan

        return {
            'Symbol': symbol, 'Price': current_price, 'Ret 1Y': ret_stock, 'Bench Ret 1Y': ret_bench,
            'Alpha (vs Bench)': alpha, 'Analyst Target': target_mean, 'Consensus Upside %': upside_consensus,
            'Recommendation': recommendation, 'SMA 50': sma50, 'SMA 200': sma200, 'Drawdown (52w)': dist_from_high, 'Status': 'OK'
        }
    except Exception as e:
        return {'Symbol': symbol, 'Status': 'ERRORE API', 'Ret 1Y': np.nan, 'Bench Ret 1Y': np.nan, 'Alpha (vs Bench)': np.nan, 'Analyst Target': np.nan, 'Consensus Upside %': np.nan, 'Recommendation': 'ERRORE', 'SMA 50': np.nan, 'SMA 200': np.nan, 'Drawdown (52w)': np.nan}

# =============================================================================
# 4. INTERFACCIA A TAB
# =============================================================================
tab_home, tab_stock, tab_bond, tab_portfolio, tab_deepdive = st.tabs(["[🏠] MACRO", "[EQ] EQUITY", "[FI] FIXED INCOME", "[PRT] MATRIX", "[🔍] DEEP DIVE"])

with tab_home:
    st.subheader("➤ GLOBAL MACRO & SENTIMENT (EXTERNAL FEED)")
    st.warning("⚠️ RETAIL FEED: Dati puramente visivi e isolati. Non interagiscono con il motore quantitativo backend.")
    col_dx, col_sx = st.columns([1.5, 1])
    with col_dx:
        cal_html = """<iframe src="https://sslecal2.investing.com?ecoDayBackground=%23000000&defaultFont=%23050505&ecoDayFontColor=%23d47f00&columns=exc_flags,exc_currency,exc_importance,exc_actual,exc_forecast,exc_previous&features=datepicker,timezone&countries=25,32,6,37,72,22,17,39,14,10,35,43,56,36,110,11,26,12,4,5&calType=week&timeZone=16&lang=1" width="100%" height="467" frameborder="0" allowtransparency="true" marginwidth="0" marginheight="0"></iframe><div class="poweredBy" style="font-family: Arial, Helvetica, sans-serif;"><span style="font-size: 11px;color: #333333;text-decoration: none;">Real Time Economic Calendar provided by <a href="https://www.investing.com/" rel="nofollow" target="_blank" style="font-size: 11px;color: #06529D; font-weight: bold;" class="underline_link">Investing.com</a>.</span></div>"""
        components.html(cal_html, height=500)
    with col_sx:
        fx_html = """<iframe src="https://www.widgets.investing.com/live-currency-cross-rates?theme=darkTheme&roundedCorners=true&pairs=1,3,2,4,7,5,8,6,9,10,11" width="100%" height="220" frameborder="0" allowtransparency="true" marginwidth="0" marginheight="0"></iframe><div class="poweredBy" style="font-family: Arial, Helvetica, sans-serif;">Powered by <a href="https://www.investing.com?utm_source=WMT&amp;utm_medium=referral&amp;utm_campaign=LIVE_CURRENCY_X_RATES&amp;utm_content=Footer%20Link" target="_blank" rel="nofollow">Investing.com</a></div>"""
        components.html(fx_html, height=250)
        crypto_html = """<iframe src="https://www.widgets.investing.com/top-cryptocurrencies?theme=darkTheme&roundedCorners=true" width="100%" height="220" frameborder="0" allowtransparency="true" marginwidth="0" marginheight="0"></iframe><div class="poweredBy" style="font-family: Arial, Helvetica, sans-serif;">Powered by <a href="https://www.investing.com?utm_source=WMT&amp;utm_medium=referral&amp;utm_campaign=TOP_CRYPTOCURRENCIES&amp;utm_content=Footer%20Link" target="_blank" rel="nofollow">Investing.com</a></div>"""
        components.html(crypto_html, height=250)

with tab_stock:
    st.subheader("➤ EQUITY COMMAND LINE & VALUATION ENGINE")
    symbols_input = st.text_area("INSERISCI TICKER (Separati da a capo)", "AAPL\nMSFT\nENI.MI\nTSLA", height=100)
    if st.button("EXECUTE VALUATION SCENARIOS"):
        symbols = [s.strip().upper() for s in symbols_input.split('\n') if s.strip()]
        st.session_state['api_calls'] += len(symbols)
        results = [fetch_stock_data(sym) for sym in symbols]
        st.session_state['df_stock'] = pd.DataFrame(results)
        st.success("Modelli di valutazione calcolati e bloccati in RAM.")
        
    if st.session_state['df_stock'] is not None:
        df_show = st.session_state['df_stock'].copy()
        def color_mos(val):
            if pd.isna(val): return ''
            return 'color: #10B981; font-weight: bold' if val > 15 else ('color: #EF4444' if val < 0 else 'color: #F8B400')
        st.dataframe(df_show.style.format({
            'Price': '{:.2f}', 'P/E': '{:.2f}', 'ROE %': '{:.1f}%', 'Rev Growth %': '{:.1f}%', 'Debt/Eq': '{:.1f}', 'EPS': '{:.2f}', 
            'Val. BEAR': '{:.2f}', 'Val. BASE': '{:.2f}', 'Val. BULL': '{:.2f}', 'MoS %': '{:.1f}%'
        }, na_rep='N/A').applymap(color_mos, subset=['MoS %']), use_container_width=True)

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
        df_s['Asset Class'], df_s['Identifier'], df_s['Yield/Growth'] = 'Equity', df_s['Symbol'], df_s['ROE %']
        df_b = st.session_state['df_bond'].copy()
        df_b['Asset Class'], df_b['Identifier'], df_b['Yield/Growth'] = 'Fixed Income', df_b['ISIN'], df_b['YTM']
        portfolio = pd.concat([df_s[['Identifier', 'Asset Class', 'Price', 'Yield/Growth', 'Status']], df_b[['Identifier', 'Asset Class', 'Price', 'Yield/Growth', 'Status']]], ignore_index=True)
        portfolio_clean = portfolio[portfolio['Status'] == 'OK'].dropna(subset=['Price', 'Yield/Growth'])
        st.dataframe(portfolio, use_container_width=True)
        if not portfolio_clean.empty:
            fig = px.scatter(portfolio_clean, x='Price', y='Yield/Growth', color='Asset Class', hover_name='Identifier', color_discrete_sequence=['#3B82F6', '#10B981'])
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#FFFFFF', family="Courier New"), xaxis=dict(showgrid=True, gridcolor='#1F2937', title="Prezzo di Mercato"), yaxis=dict(showgrid=True, gridcolor='#1F2937', title="Rendimento (YTM / ROE %)"))
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ SYSTEM HALT: Popolare entrambi i database (EQ e FI) dai rispettivi tab prima di generare la matrice.")

with tab_deepdive:
    st.subheader("➤ X-RAY SINGLE ASSET (ISOLATED ANALYSIS)")
    st.write("Analisi incrociata: Motore Quantitativo vs Analyst Consensus vs Trend Tecnico")
    col_input, col_bench = st.columns(2)
    with col_input:
        target_ticker = st.text_input("Inserisci UN SINGOLO Ticker (es: MSFT, NVDA, RACE.MI):", "MSFT").upper()
    with col_bench:
        benchmark_ticker = st.text_input("Inserisci Benchmark di Riferimento (es: SPY, QQQ):", "SPY").upper()
        
    if st.button("EXECUTE X-RAY ANALYSIS"):
        st.session_state['api_calls'] += 2
        with st.spinner('Estrazione dati storici e consensus in corso...'):
            dd_data = fetch_deep_dive(target_ticker, benchmark_ticker)
            if dd_data.get('Status') == 'OK':
                st.markdown(f"### 🎯 RADIOGRAFIA: {dd_data['Symbol']}")
                
                st.markdown("#### 1. FORZA RELATIVA (1 YEAR PERFORMANCE)")
                c1, c2, c3 = st.columns(3)
                c1.metric(f"Rendimento {dd_data['Symbol']}", f"{dd_data['Ret 1Y']:.2f}%" if pd.notnull(dd_data['Ret 1Y']) else "N/A")
                c2.metric(f"Rendimento {benchmark_ticker}", f"{dd_data['Bench Ret 1Y']:.2f}%" if pd.notnull(dd_data['Bench Ret 1Y']) else "N/A")
                c3.metric("Alpha Generato", f"{dd_data['Alpha (vs Bench)']:.2f}%" if pd.notnull(dd_data['Alpha (vs Bench)']) else "N/A", delta="Sotto-performante" if pd.notnull(dd_data['Alpha (vs Bench)']) and dd_data['Alpha (vs Bench)'] < 0 else "Sovra-performante", delta_color="normal")
                
                st.markdown("#### 2. WALL STREET CONSENSUS (LE 'VOCI')")
                c4, c5, c6 = st.columns(3)
                c4.metric("Raccomandazione Analisti", dd_data['Recommendation'])
                c5.metric("Target Price Medio", f"{dd_data['Analyst Target']:.2f}" if pd.notnull(dd_data['Analyst Target']) else "N/A")
                c6.metric("Upside Stimato %", f"{dd_data['Consensus Upside %']:.2f}%" if pd.notnull(dd_data['Consensus Upside %']) else "N/A")
                
                st.markdown("#### 3. REGIME TECNICO E DRAWDOWN")
                c7, c8, c9 = st.columns(3)
                trend_status = "📉 BEAR (Sotto SMA200)" if pd.notnull(dd_data['Price']) and pd.notnull(dd_data['SMA 200']) and dd_data['Price'] < dd_data['SMA 200'] else ("📈 BULL (Sopra SMA200)" if pd.notnull(dd_data['Price']) and pd.notnull(dd_data['SMA 200']) else "N/A")
                c7.metric("Trend di Lungo Periodo", trend_status)
                sma_cross = "🟢 POSITIVO (SMA50 > SMA200)" if pd.notnull(dd_data['SMA 50']) and pd.notnull(dd_data['SMA 200']) and dd_data['SMA 50'] > dd_data['SMA 200'] else ("🔴 NEGATIVO (Death Cross)" if pd.notnull(dd_data['SMA 50']) and pd.notnull(dd_data['SMA 200']) else "N/A")
                c8.metric("Momentum Breve/Medio", sma_cross)
                c9.metric("Distanza da 52w High", f"{dd_data['Drawdown (52w)']:.2f}%" if pd.notnull(dd_data['Drawdown (52w)']) else "N/A")
            else:
                st.error("Errore API. Verifica che i ticker esistano su Yahoo Finance.")

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.caption(f"SYSTEM STATUS: ONLINE | CACHE: ACTIVE | TOTAL API HITS (SESSION): {st.session_state['api_calls']}")
