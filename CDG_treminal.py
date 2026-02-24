import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
import streamlit.components.v1 as components
from datetime import datetime, timedelta

# =============================================================================
# 1. CORE CONFIG & UI INJECTION
# =============================================================================
st.set_page_config(page_title="Alpha Terminal V10.2 - Final Edition", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    .stApp { background-color: #0A0E17; color: #D1D5DB; font-family: 'Inter', -apple-system, sans-serif; }
    h1, h2, h3, h4 { color: #F8B400 !important; font-family: 'SF Pro Display', sans-serif; font-weight: 600; letter-spacing: -0.5px; }
    .stButton>button { background-color: #1E293B; color: #60A5FA; border: 1px solid #3B82F6; border-radius: 4px; font-weight: bold; transition: all 0.2s; width: 100%; }
    .stButton>button:hover { background-color: #3B82F6; color: #FFFFFF; border-color: #60A5FA; }
    .stTextArea>div>div>textarea, .stTextInput>div>div>input { background-color: #111827; color: #10B981; font-family: 'Courier New', Courier, monospace; border: 1px solid #374151; border-radius: 4px; }
    .stTextArea>div>div>textarea:focus, .stTextInput>div>div>input:focus { border-color: #F8B400; box-shadow: none; }
    .stTabs [data-baseweb="tab-list"] { background-color: #0F172A; border-bottom: 2px solid #1E293B; }
    .stTabs [data-baseweb="tab"] { color: #94A3B8; font-weight: 600; padding-top: 1rem; padding-bottom: 1rem; }
    .stTabs [aria-selected="true"] { color: #F8B400; border-bottom-color: #F8B400; }
    .stAlert { background-color: #111827; border: 1px solid #374151; color: #D1D5DB; }
    
    .stTextInput label p, .stTextArea label p { color: #F8B400 !important; font-weight: bold; }
    [data-testid="stMetricValue"] div { color: #FFFFFF !important; }
    [data-testid="stMetricLabel"] p { color: #FFFFFF !important; }
</style>
""", unsafe_allow_html=True)

st.title("ALPHA TERMINAL // PROPRIETARY ENGINE")
st.markdown("---")

# =============================================================================
# 2. SESSION STATE
# =============================================================================
if 'df_stock' not in st.session_state: st.session_state['df_stock'] = None
if 'df_bond' not in st.session_state: st.session_state['df_bond'] = None
if 'api_calls' not in st.session_state: st.session_state['api_calls'] = 0

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
        if g_raw is None: g_raw = rev_growth if pd.notnull(rev_growth) else 0.05
        g_base = g_raw * 100 
        Y = 4.4 
        
        if pd.notnull(eps) and eps > 0:
            val_bear = (eps * (8.5 + 2 * (g_base * 0.5)) * 4.4) / Y
            val_base = (eps * (8.5 + 2 * g_base) * 4.4) / Y
            val_bull = (eps * (8.5 + 2 * (g_base * 1.2)) * 4.4) / Y
            mos = ((val_base - price) / val_base) * 100 if pd.notnull(price) and val_base > 0 else np.nan
        else:
            val_bear = val_base = val_bull = mos = np.nan

        return {
            'Symbol': symbol, 'Price': price, 'P/E': pe, 'ROE %': roe * 100 if pd.notnull(roe) else np.nan,
            'Rev Growth %': rev_growth * 100 if pd.notnull(rev_growth) else np.nan, 'Debt/Eq': debt_eq,
            'EPS': eps, 'Val. BEAR': val_bear, 'Val. BASE': val_base, 'Val. BULL': val_bull, 'MoS %': mos, 'Status': 'OK'
        }
    except Exception:
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
        
        current_price = info.get('currentPrice', np.nan)
        eps = info.get('trailingEps', np.nan)
        rev_growth = info.get('revenueGrowth', np.nan)

        g_raw = info.get('earningsGrowth', None)
        if g_raw is None: g_raw = rev_growth if pd.notnull(rev_growth) else 0.05
        g_base = g_raw * 100
        Y = 4.4
        
        if pd.notnull(eps) and eps > 0 and pd.notnull(current_price):
            implied_g = (((current_price * Y) / (eps * 4.4)) - 8.5) / 2
            graham_fv = (eps * (8.5 + 2 * g_base) * 4.4) / Y
        else:
            implied_g = np.nan
            graham_fv = np.nan

        if not hist_stock.empty and pd.notnull(current_price):
            mean_p = hist_stock['Close'].mean()
            std_p = hist_stock['Close'].std()
            z_score = (current_price - mean_p) / std_p if std_p > 0 else np.nan
            dates = hist_stock.index
            prices = hist_stock['Close']
        else:
            mean_p = std_p = z_score = dates = prices = np.nan

        target_mean = info.get('targetMeanPrice', np.nan)
        upside_consensus = ((target_mean / current_price) - 1) * 100 if pd.notnull(target_mean) and pd.notnull(current_price) else np.nan

        return {
            'Symbol': symbol, 'Price': current_price, 'Ret 1Y': ret_stock, 'Bench Ret 1Y': ret_bench,
            'Alpha (vs Bench)': alpha, 'Analyst Target': target_mean, 'Consensus Upside %': upside_consensus,
            'Implied Growth %': implied_g, 'Z-Score': z_score, 'Mean Price': mean_p, 'Std Price': std_p, 'Graham FV': graham_fv, 
            'Dates': dates, 'Prices': prices, 'Status': 'OK'
        }
    except Exception:
        return {'Symbol': symbol, 'Status': 'ERRORE API'}

# =============================================================================
# 4. INTERFACCIA A TAB
# =============================================================================
tab_home, tab_stock, tab_bond, tab_portfolio, tab_deepdive = st.tabs(["[🏠] MACRO", "[EQ] EQUITY", "[FI] FIXED INCOME", "[PRT] MATRIX", "[🔍] DEEP DIVE"])

with tab_home:
    st.subheader("➤ GLOBAL MACRO & SENTIMENT (EXTERNAL FEED)")
    st.warning("⚠️ RETAIL FEED: Dati puramente visivi e isolati. Non interagiscono con il motore quantitativo backend.")
    
    col_sx, col_cx, col_dx = st.columns([1, 1.5, 1])
    
    with col_sx:
        st.markdown("**TECHNICAL SUMMARY (RETAIL)**")
        tech_html = """<iframe src="https://ssltsw.investing.com?tabsLine=%23ff890a&chosenTab=%234a2804&notChosenTab=%23062545&buttonFont=%23000000&lang=1&forex=1,2,3,5,7,9,10&commodities=8830,8836,8831,8849,8833,8862,8832&indices=175,166,172,27,179,170,174&stocks=345,346,347,348,349,350,352&tabs=1,2,3,4" width="100%" height="467" frameborder="0" allowtransparency="true" marginwidth="0" marginheight="0"></iframe><div class="poweredBy" style="font-family:arial,helvetica,sans-serif; direction:ltr;"><span style="font-size: 11px;color: #333333;text-decoration: none;">Technical Summary Widget Powered by <a href="https://www.investing.com/" rel="nofollow" target="_blank" style="font-size: 11px;color: #06529D; font-weight: bold;" class="underline_link">Investing.com</a></span></div>"""
        components.html(tech_html, height=480)
        
    with col_cx:
        st.markdown("**CALENDARIO ECONOMICO**")
        cal_html = """<iframe src="https://sslecal2.investing.com?ecoDayBackground=%23000000&defaultFont=%23050505&ecoDayFontColor=%23d47f00&columns=exc_flags,exc_currency,exc_importance,exc_actual,exc_forecast,exc_previous&features=datepicker,timezone&countries=25,32,6,37,72,22,17,39,14,10,35,43,56,36,110,11,26,12,4,5&calType=week&timeZone=16&lang=1" width="100%" height="467" frameborder="0"></iframe>"""
        components.html(cal_html, height=480)
        
    with col_dx:
        st.markdown("**LIVE CURRENCIES & CRYPTO**")
        fx_html = """<iframe src="https://www.widgets.investing.com/live-currency-cross-rates?theme=darkTheme&roundedCorners=true&pairs=1,3,2,4,7,5,8,6,9,10,11" width="100%" height="220" frameborder="0"></iframe>"""
        components.html(fx_html, height=230)
        crypto_html = """<iframe src="https://www.widgets.investing.com/top-cryptocurrencies?theme=darkTheme&roundedCorners=true" width="100%" height="220" frameborder="0"></iframe>"""
        components.html(crypto_html, height=230)

with tab_stock:
    st.subheader("➤ EQUITY COMMAND LINE & VALUATION ENGINE")
    symbols_input = st.text_area("INSERISCI TICKER (Separati da a capo)", "AAPL\nMSFT\nENI.MI", height=100)
    if st.button("EXECUTE VALUATION SCENARIOS"):
        symbols = [s.strip().upper() for s in symbols_input.split('\n') if s.strip()]
        st.session_state['api_calls'] += len(symbols)
        st.session_state['df_stock'] = pd.DataFrame([fetch_stock_data(sym) for sym in symbols])
    if st.session_state['df_stock'] is not None:
        def color_mos(val):
            if pd.isna(val): return ''
            return 'color: #10B981; font-weight: bold' if val > 15 else ('color: #EF4444' if val < 0 else 'color: #F8B400')
        st.dataframe(st.session_state['df_stock'].style.format({'Price': '{:.2f}', 'P/E': '{:.2f}', 'ROE %': '{:.1f}%', 'Rev Growth %': '{:.1f}%', 'Debt/Eq': '{:.1f}', 'EPS': '{:.2f}', 'Val. BEAR': '{:.2f}', 'Val. BASE': '{:.2f}', 'Val. BULL': '{:.2f}', 'MoS %': '{:.1f}%'}, na_rep='N/A').applymap(color_mos, subset=['MoS %']), use_container_width=True)

with tab_bond:
    st.subheader("➤ FIXED INCOME COMMAND LINE")
    isins_input = st.text_area("INSERISCI ISIN (Separati da a capo)", "XS2486589596\nFR0013327988", height=100)
    if st.button("EXECUTE BOND QUERY"):
        isins = [i.strip().upper() for i in isins_input.split('\n') if i.strip()]
        st.session_state['df_bond'] = pd.DataFrame([fetch_bond_data(isin) for isin in isins])
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

with tab_deepdive:
    st.subheader("➤ X-RAY SINGLE ASSET (QUANTITATIVE ISOLATION)")
    col_input, col_bench = st.columns(2)
    with col_input: target_ticker = st.text_input("Inserisci UN SINGOLO Ticker:", "NVDA").upper()
    with col_bench: benchmark_ticker = st.text_input("Inserisci Benchmark di Riferimento:", "SPY").upper()
        
    if st.button("EXECUTE QUANTITATIVE X-RAY"):
        st.session_state['api_calls'] += 2
        with st.spinner('Calcolo modelli quantitativi e distribuzioni statistiche...'):
            dd_data = fetch_deep_dive(target_ticker, benchmark_ticker)
            
            if dd_data.get('Status') == 'OK':
                st.markdown(f"### 🎯 RADIOGRAFIA: {dd_data['Symbol']}")
                
                with st.expander("📖 SPIEGAZIONE METRICHE (LEGGERE PRIMA DI AGIRE)"):
                    st.markdown("""
                    * **Crescita Implicita (Rev DCF):** Se l'azienda fosse prezzata giustamente *oggi*, quanto dovrebbe crescere ogni anno per giustificare questo prezzo? Se il numero è > 20%, il mercato è in pura estasi irrazionale. Al primo trimestre sotto le attese, collasserà.
                    * **Z-Score (Elastico):** Calcola quante deviazioni standard il prezzo attuale dista dalla sua media annuale. Maggiore di +2? Stai comprando sui massimi assoluti spinto dalla massa. Minore di -2? Stai raccogliendo i cocci del panico altrui.
                    * **Alpha vs Benchmark:** La differenza tra il rendimento del tuo titolo e quello dell'indice (S&P 500). Se è rosso, detenere questo titolo invece dell'indice S&P 500 è stato finanziariamente distruttivo per il tuo portafoglio.
                    """)

                c1, c2, c3 = st.columns(3)
                implied_g = dd_data['Implied Growth %']
                c1.metric("Crescita Implicita (Rev DCF)", f"{implied_g:.2f}%" if pd.notnull(implied_g) else "N/A", delta="Delirio (>20%)" if pd.notnull(implied_g) and implied_g > 20 else "Razionale", delta_color="inverse")
                z = dd_data['Z-Score']
                c2.metric("Z-Score Storico (1Y)", f"{z:.2f} σ" if pd.notnull(z) else "N/A", delta="Bolla Statistica" if pd.notnull(z) and z > 2 else "In Media", delta_color="inverse" if pd.notnull(z) and z > 2 else "normal")
                c3.metric("Alpha vs Benchmark", f"{dd_data['Alpha (vs Bench)']:.2f}%" if pd.notnull(dd_data['Alpha (vs Bench)']) else "N/A", delta="Distruzione Valore" if pd.notnull(dd_data['Alpha (vs Bench)']) and dd_data['Alpha (vs Bench)'] < 0 else "Creazione Valore", delta_color="normal")
                
                st.markdown("---")
                st.markdown("#### 📉 DINAMICA DEI PREZZI VS STORICO (1Y Z-SCORE BANDS)")
                
                if type(dd_data['Dates']) is not float: 
                    fig_ts = go.Figure()
                    fig_ts.add_trace(go.Scatter(x=dd_data['Dates'], y=dd_data['Prices'], mode='lines', name='Prezzo Mercato', line=dict(color='#F8B400', width=2)))
                    fig_ts.add_trace(go.Scatter(x=dd_data['Dates'], y=[dd_data['Mean Price']]*len(dd_data['Dates']), mode='lines', name='Media 1Y', line=dict(color='#94A3B8', width=1, dash='dash')))
                    fig_ts.add_trace(go.Scatter(x=dd_data['Dates'], y=[dd_data['Mean Price'] + (2*dd_data['Std Price'])]*len(dd_data['Dates']), mode='lines', name='+2σ (Euforia Estrema)', line=dict(color='#EF4444', width=1, dash='dot')))
                    fig_ts.add_trace(go.Scatter(x=dd_data['Dates'], y=[dd_data['Mean Price'] - (2*dd_data['Std Price'])]*len(dd_data['Dates']), mode='lines', name='-2σ (Panico Estremo)', line=dict(color='#10B981', width=1, dash='dot')))
                    
                    if pd.notnull(dd_data['Graham FV']):
                        fig_ts.add_trace(go.Scatter(x=dd_data['Dates'], y=[dd_data['Graham FV']]*len(dd_data['Dates']), mode='lines', name='Graham Fair Value', line=dict(color='#3B82F6', width=2)))

                    fig_ts.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#FFFFFF', family="Courier New"),
                        xaxis_title="Data", yaxis_title="Prezzo del Titolo ($)",
                        hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    st.plotly_chart(fig_ts, use_container_width=True)
                else:
                    st.warning("Dati storici insufficienti per generare il grafico di deviazione standard.")
            else:
                st.error("Errore API. Verifica il ticker.")

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.caption(f"SYSTEM STATUS: ONLINE | CACHE: ACTIVE | TOTAL API HITS: {st.session_state['api_calls']}")
