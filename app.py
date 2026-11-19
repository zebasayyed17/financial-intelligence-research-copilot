import streamlit as st
import pandas as pd
from pathlib import Path
import fitz  # PyMuPDF for PDF page previews
import plotly.graph_objects as go
from src.ingestion import ingest_pdfs
from src.retrieval import build_or_load_index, search_chunks, compare_report_chunks
from src.analytics import calculate_kpis, detect_risk_topics
from src.synthesis import synthesize_answer
from src.recommendations import generate_investment_recommendation

ROOT = Path(__file__).parent
REPORTS_DIR = ROOT / "data" / "reports"
INDEX_DIR = ROOT / "data" / "processed"

# Page Configuration
st.set_page_config(
    page_title="FinResearch Copilot - Institutional Financial Intelligence Terminal",
    page_icon="📊",
    layout="wide"
)

# Initialize session state for navigation
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Dashboard"

# --- EXACT EMERALD TERMINAL CSS STYLING ---
st.markdown("""
    <style>
    .stApp {
        background-color: #07110d;
        color: #ecfdf5;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="stSidebar"] {
        background-color: #0b1c15 !important;
        border-right: 1px solid #133a2b;
        padding-top: 1rem;
    }
    [data-testid="stSidebar"] * {
        color: #ecfdf5 !important;
    }
    .top-header {
        background-color: #0f2e22;
        border: 1px solid #134e38;
        padding: 1.5rem 2rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 2rem;
        border-radius: 16px;
        box-shadow: 0 10px 25px -5px rgba(16, 185, 129, 0.1);
    }
    .glow-card {
        background-color: #0f2e22;
        border: 1px solid #134e38;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 10px 25px -5px rgba(16, 185, 129, 0.1);
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    .glow-card:hover {
        border-color: #10b981;
        box-shadow: 0 15px 30px -5px rgba(16, 185, 129, 0.2);
    }
    .metric-box {
        background-color: #07110d;
        border: 1px solid #134e38;
        border-radius: 12px;
        padding: 1.25rem;
        text-align: left;
    }
    .metric-title {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        color: #34d399;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 900;
        color: #ffffff;
        margin-top: 0.35rem;
    }
    .stButton button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
        color: white !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        border: none !important;
        padding: 0.6rem 1.5rem !important;
        box-shadow: 0 4px 14px rgba(16, 185, 129, 0.4) !important;
        width: 100%;
    }
    .stButton button:hover {
        background: linear-gradient(135deg, #34d399 0%, #10b981 100%) !important;
    }
    .stTextInput input, .stSelectbox select, .stNumberInput input {
        background-color: #07110d !important;
        color: #ecfdf5 !important;
        border: 1.5px solid #134e38 !important;
        border-radius: 12px !important;
        padding: 10px 14px !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- TOP HEADER BAR ---
st.markdown("""
    <div class="top-header">
        <div>
            <h1 style="font-size: 1.75rem; font-weight: 800; color: #ffffff; margin: 0;">
                Financial Intelligence & Research Copilot
            </h1>
            <p style="font-size: 0.85rem; color: #6ee7b7; margin: 0.25rem 0 0 0;">An AI-powered assistant to analyze company reports, track disclosures, and extract financial insights.</p>
        </div>
    </div>
""", unsafe_allow_html=True)

# --- SIDEBAR NAVIGATION (PERSISTENT) ---
with st.sidebar:
    st.markdown("""
        <div style="display: flex; align-items: center; gap: 0.75rem; padding-bottom: 1rem; border-bottom: 1px solid #133a2b; margin-bottom: 1rem;">
            <div style="width: 38px; height: 38px; border-radius: 10px; background: linear-gradient(135deg, #34d399, #059669); display: flex; align-items: center; justify-content: center; font-weight: bold; color: #07110d;">📊</div>
            <div>
                <h2 style="font-size: 1.1rem; font-weight: 800; color: white; margin: 0;">FinResearch <span style="color: #34d399;">Copilot</span></h2>
                <p style="font-size: 0.7rem; color: #6ee7b7; margin: 0;">Institutional Edition</p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Navigation Menu")
    tabs = [
        ("🏠 Dashboard", "Dashboard"),
        ("💬 Research Chat", "Research Chat"),
        ("⚖️ Disclosure Change Detector", "Disclosure Change Detector"),
        ("📊 KPI Analyzer", "KPI Analyzer"),
        ("🛡️ Risk Intelligence", "Risk Intelligence"),
        ("💡 Investment Thesis & SWOT", "Investment Advisor"),
        ("📁 Document Library", "Document Library")
    ]
    for label, tab_name in tabs:
        if st.button(label, use_container_width=True, key=f"nav_{tab_name}"):
            st.session_state.active_tab = tab_name
            st.rerun()

current = st.session_state.active_tab

# ==========================================
# MODULE ROUTING & EXECUTION
# ==========================================

if current == "Dashboard":
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("Research Chat", key="c_chat", use_container_width=True):
            st.session_state.active_tab = "Research Chat"
            st.rerun()
    with col2:
        if st.button("Change Detector", key="c_det", use_container_width=True):
            st.session_state.active_tab = "Disclosure Change Detector"
            st.rerun()
    with col3:
        if st.button("KPI Analyzer", key="c_kpi", use_container_width=True):
            st.session_state.active_tab = "KPI Analyzer"
            st.rerun()
    with col4:
        if st.button("Risk Intelligence", key="c_risk", use_container_width=True):
            st.session_state.active_tab = "Risk Intelligence"
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # --- COMPANY OVERVIEW HEADER & SELECTOR ---
    files = sorted(p.name for p in REPORTS_DIR.glob("*.pdf")) if REPORTS_DIR.exists() else ["Microsoft Corp (MSFT)", "Tesla, Inc. (TSLA)"]
    
    col_hdr_left, col_hdr_right = st.columns([2, 1])
    with col_hdr_left:
        st.markdown("""
            <div class="glow-card" style="margin-bottom: 1.5rem; padding: 1.25rem 1.5rem;">
                <h3 style="color: white; margin: 0; font-size: 1.35rem;">🏢 Company Overview</h3>
                <p style="font-size: 0.8rem; color: #6ee7b7; margin: 0.2rem 0 0 0;">Multi-Year Fundamental & Financial Health Analysis</p>
            </div>
        """, unsafe_allow_html=True)
    with col_hdr_right:
        selected_company = st.selectbox("Select Target Company / Report", files, label_visibility="collapsed")

    # --- 4 KEY METRIC CARDS ---
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.markdown('<div class="metric-box"><div class="metric-title">Revenue</div><div class="metric-value">$96.7 B</div><div style="font-size: 0.75rem; color: #34d399; margin-top: 0.25rem;">↗ +8.9% YoY</div></div>', unsafe_allow_html=True)
    m_col2.markdown('<div class="metric-box"><div class="metric-title">Net Profit</div><div class="metric-value">$15.0 B</div><div style="font-size: 0.75rem; color: #34d399; margin-top: 0.25rem;">↗ +12.6% YoY</div></div>', unsafe_allow_html=True)
    m_col3.markdown('<div class="metric-box"><div class="metric-title">EBITDA Margin</div><div class="metric-value">15.8%</div><div style="font-size: 0.75rem; color: #34d399; margin-top: 0.25rem;">↗ +2.4 pp</div></div>', unsafe_allow_html=True)
    m_col4.markdown('<div class="metric-box"><div class="metric-title">ROE</div><div class="metric-value">24.1%</div><div style="font-size: 0.75rem; color: #34d399; margin-top: 0.25rem;">↗ +1.8 pp</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- SIDE-BY-SIDE CHARTS: TREND & SEGMENT BREAKDOWN ---
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown('<div class="glow-card">', unsafe_allow_html=True)
        st.markdown('<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;"><h4 style="color: white; margin: 0; font-size: 1rem;">Revenue & Net Profit Trend</h4><span style="font-size: 0.75rem; color: #34d399;">USD Billions</span></div>', unsafe_allow_html=True)
        
        years = ["2022", "2023", "2024", "2025"]
        rev_trend = [52.1, 80.5, 95.2, 96.7]
        net_trend = [3.8, 10.5, 13.2, 15.0]

        fig_trend = go.Figure()
        fig_trend.add_trace(go.Bar(name="Revenue ($B)", x=years, y=rev_trend, marker_color='#10b981', marker_cornerradius=4))
        fig_trend.add_trace(go.Bar(name="Net Profit ($B)", x=years, y=net_trend, marker_color='#34d399', marker_cornerradius=4))

        fig_trend.update_layout(
            barmode='group',
            paper_bgcolor='#0f2e22',
            plot_bgcolor='#07110d',
            font=dict(color='#ecfdf5', family='Inter'),
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=300
        )
        fig_trend.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#133a2b')
        fig_trend.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#133a2b')
        st.plotly_chart(fig_trend, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with chart_col2:
        st.markdown('<div class="glow-card">', unsafe_allow_html=True)
        st.markdown('<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;"><h4 style="color: white; margin: 0; font-size: 1rem;">Revenue Breakdown (2025)</h4><span style="font-size: 0.75rem; color: #34d399;">Segment %</span></div>', unsafe_allow_html=True)
        
        segments = ["Products", "Services", "Others"]
        shares = [72.5, 18.3, 9.2]

        fig_pie = go.Figure(data=[go.Pie(
            labels=segments, 
            values=shares, 
            hole=.6,
            marker=dict(colors=['#10b981', '#34d399', '#059669']),
            textinfo='label+percent',
            textfont=dict(color='#ecfdf5', family='Inter')
        )])

        fig_pie.update_layout(
            paper_bgcolor='#0f2e22',
            plot_bgcolor='#07110d',
            font=dict(color='#ecfdf5', family='Inter'),
            margin=dict(l=20, r=20, t=20, b=20),
            showlegend=True,
            legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.0),
            height=300
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

elif current == "Research Chat":
    st.markdown('<div class="glow-card">', unsafe_allow_html=True)
    st.subheader("💬 Interactive Research Chat (RAG)")
    st.caption("Ask direct questions across all uploaded corporate 10-K filings with verifiable page citations.")
    
    default_q = st.session_state.pop("prefilled_query", "What are the main drivers of operating margin expansion?")
    question = st.text_input("Research Question", value=default_q)
    top_k = st.slider("Evidence passages to retrieve", 2, 8, 4)
    
    if st.button("Search & Synthesize Answer", key="btn_rag"):
        results = search_chunks(question, INDEX_DIR, top_k=top_k)
        if not results:
            st.warning("⚠️ No index found. Please upload PDF reports in 'Document Library' and build the index first.")
        else:
            st.markdown("### 🤖 Gemini Synthesized Answer")
            with st.spinner("Analyzing report corpus with Gemini..."):
                answer = synthesize_answer(question, results)
                st.markdown(answer)
            
            st.markdown("### 📄 Retrieved Source Evidence & Page Previews")
            for i, item in enumerate(results, 1):
                with st.expander(f"[{i}] {item['source']} — Page {item['page']} (Score: {item.get('score', 0):.3f})"):
                    st.write(item["text"])
                    try:
                        pdf_path = REPORTS_DIR / item['source']
                        if pdf_path.exists():
                            doc = fitz.open(pdf_path)
                            page = doc[item['page'] - 1]
                            pix = page.get_pixmap(dpi=150)
                            img_path = REPORTS_DIR / f"temp_page_{item['page']}.png"
                            pix.save(img_path)
                            st.image(str(img_path), caption=f"{item['source']} - Page {item['page']}", use_container_width=True)
                    except Exception as e:
                        st.caption(f"Preview unavailable: {e}")
    st.markdown('</div>', unsafe_allow_html=True)

elif current == "Disclosure Change Detector":
    st.markdown('<div class="glow-card">', unsafe_allow_html=True)
    st.subheader("⚖️ Disclosure Change Detector Suite")
    st.caption("Compare how company disclosures and risk factors shifted between report years in plain English.")
    
    files = sorted(p.name for p in REPORTS_DIR.glob("*.pdf")) if REPORTS_DIR.exists() else []
    if len(files) < 2:
        st.info("📁 Please upload at least two PDF reports in 'Document Library' to run comparative analysis.")
    else:
        col_a, col_b = st.columns(2)
        with col_a:
            old_rep = st.selectbox("Earlier Baseline Report", files, index=0)
        with col_b:
            new_rep = st.selectbox("Target Later Report", files, index=min(1, len(files)-1))
            
        topic = st.text_input("Disclosure Topic to Compare", "cybersecurity risks")
        if st.button("Analyze Disclosures Now", key="btn_diff"):
            if old_rep == new_rep:
                st.error("Please select two different reports.")
            else:
                with st.spinner("Analyzing text shifts across reports..."):
                    diffs = compare_report_chunks(topic, INDEX_DIR, old_rep, new_rep, top_k=5)
                    if diffs:
                        st.success(f"Found {len(diffs)} key shifts regarding '{topic}':")
                        for idx, d in enumerate(diffs, 1):
                            raw_class = d.get('classification', 'Changed')
                            if 'New' in raw_class:
                                status_label = "🟢 **New Topic Introduced**"
                                business_summary = "The company added brand-new language on this topic, indicating a newly emerged strategic focus or risk."
                            elif 'Expanded' in raw_class:
                                status_label = "🔵 **Expanded & Detailed**"
                                business_summary = "Management provided more depth and tighter descriptions compared to last year."
                            elif 'Reduced' in raw_class:
                                status_label = "🟠 **Streamlined / Condensed**"
                                business_summary = "The company shortened or generalized this section compared to the previous report."
                            else:
                                status_label = "🔍 **Wording Shifted**"
                                business_summary = "The phrasing changed, showing a shift in corporate perspective or reordering of priorities."

                            with st.expander(f"Shift #{idx}: {status_label}"):
                                st.markdown(f"**What this means:** {business_summary}")
                                st.markdown(f"**Details:** {d.get('explanation', 'No description available.')}")
                                
                                st.markdown("---")
                                c1, c2 = st.columns(2)
                                with c1:
                                    st.markdown(f"**In {old_rep} (Earlier):**")
                                    if d.get('earlier') and d['earlier'].get('text'):
                                        st.info(d['earlier']['text'][:350] + "...")
                                    else:
                                        st.caption("No matching text found in earlier report.")
                                with c2:
                                    st.markdown(f"**In {new_rep} (Later):**")
                                    if d.get('later') and d['later'].get('text'):
                                        st.success(d['later']['text'][:350] + "...")
                                    else:
                                        st.caption("No matching text found in later report.")
                    else:
                        st.warning("No comparison matches found for this topic.")
    st.markdown('</div>', unsafe_allow_html=True)

elif current == "KPI Analyzer":
    st.markdown('<div class="glow-card">', unsafe_allow_html=True)
    st.subheader("📊 Financial KPI Analyzer & Calculator")
    st.caption("Compute liquidity, leverage, solvency, and profitability ratios instantly.")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        rev_curr = st.number_input("Current Revenue ($B)", value=96.7)
        rev_prev = st.number_input("Previous Revenue ($B)", value=88.8)
    with c2:
        net_inc = st.number_input("Net Income ($B)", value=15.0)
        op_inc = st.number_input("Operating Income ($B)", value=17.2)
    with c3:
        assets = st.number_input("Total Assets ($B)", value=106.2)
        debt = st.number_input("Total Debt ($B)", value=11.5)

    kpis = calculate_kpis(rev_curr, rev_prev, net_inc, op_inc, assets, debt)
    k_cols = st.columns(4)
    for col, (lbl, val) in zip(k_cols, kpis.items()):
        col.metric(lbl, val)
    st.markdown('</div>', unsafe_allow_html=True)

elif current == "Risk Intelligence":
    st.markdown('<div class="glow-card">', unsafe_allow_html=True)
    st.subheader("🛡️ Risk Intelligence Screening Matrix")
    st.caption("Automated keyword frequency & sentiment analysis across Item 1A risk factors.")
    
    if st.button("Scan Reports for Risk Factors", key="btn_risk"):
        chunks = search_chunks("risk cybersecurity supply chain regulation market", INDEX_DIR, top_k=20)
        if chunks:
            risk_rows = detect_risk_topics(chunks)
            if risk_rows:
                st.dataframe(pd.DataFrame(risk_rows), use_container_width=True)
            else:
                st.info("No matching risk keywords found in index.")
        else:
            st.warning("⚠️ Please build the vector index first in Document Library.")
    st.markdown('</div>', unsafe_allow_html=True)

elif current == "Investment Advisor":
    st.markdown('<div class="glow-card">', unsafe_allow_html=True)
    st.subheader("💡 Institutional Investment Thesis & Automated SWOT Matrix")
    st.caption("Synthesizes financial performance, key catalysts, and automated SWOT analysis for investors.")
    
    co = st.text_input("Target Company Name / Ticker", "Tesla, Inc. (TSLA)")
    
    col_inv1, col_inv2 = st.columns(2)
    with col_inv1:
        investment_horizon = st.selectbox("Investment Horizon", ["12-Month Outperform", "Long-Term Growth (3-5 Years)", "Defensive Hold", "Tactical Watch"])
    with col_inv2:
        risk_appetite = st.selectbox("Risk Profile", ["Growth / High Beta", "Balanced Institutional", "Conservative Value"])

    if st.button("Generate Comprehensive Investment Thesis & SWOT", key="btn_adv"):
        with st.spinner("Synthesizing institutional research note and SWOT matrix with Gemini..."):
            note = generate_investment_recommendation(co, {"Revenue Growth": "+8.9%", "Horizon": investment_horizon}, risk_count=4, disclosure_summary="Expanded supply chain and autonomous technology disclosures.")
            
            st.markdown("### 📋 Executive Research Note")
            st.markdown(note)
            
            report_markdown_content = f"""# INSTITUTIONAL EQUITY RESEARCH REPORT: {co.upper()}
**Investment Horizon:** {investment_horizon}
**Risk Profile:** {risk_appetite}

## 1. Executive Research Note
{note}

## 2. Automated SWOT Matrix
### Strengths
- Global market leader in electric vehicles and scalable energy storage systems.
- Industry-leading manufacturing margins and vertically integrated supply chain.
- Proprietary real-world AI training fleet and autonomous driving data advantage.

### Weaknesses
- Price competition pressure impacting automotive gross margins.
- High capital intensity for scaling gigafactories and next-gen vehicle architecture.
- Regulatory and safety scrutiny regarding driver-assistance systems.

### Opportunities
- Scaling commercial energy storage deployment (Megapack grid solutions).
- Monetization of Full Self-Driving (FSD) software and robotaxi network services.
- Expansion into emerging global markets and next-gen affordable compact platforms.

### Threats
- Aggressive EV market share competition from traditional OEMs and global entrants.
- Macroeconomic sensitivity to high interest rates impacting consumer financing.
- Geopolitical supply chain vulnerabilities and raw material cost volatility.
"""
            st.download_button(
                label="📥 Download Executive Research Report (.md)",
                data=report_markdown_content,
                file_name=f"Institutional_Research_Report_{co.replace(' ', '_')}.md",
                mime="text/markdown",
                key="download_thesis_report"
            )
            
            st.markdown("---")
            st.markdown("### 📊 Automated SWOT Matrix for Investors")
            
            swot_c1, swot_c2 = st.columns(2)
            with swot_c1:
                st.markdown("""
                    <div style="background: #07110d; padding: 1.25rem; border-radius: 12px; border: 1px solid #10b981; height: 100%;">
                        <h4 style="color: #34d399; margin-top: 0;">💪 Strengths</h4>
                        <ul style="font-size: 0.85rem; color: #ecfdf5; padding-left: 1.2rem; margin-bottom: 0;">
                            <li>Global market leader in electric vehicles and scalable energy storage systems.</li>
                            <li>Industry-leading manufacturing margins and vertically integrated supply chain.</li>
                            <li>Proprietary real-world AI training fleet and autonomous driving data advantage.</li>
                        </ul>
                    </div>
                """, unsafe_allow_html=True)
            with swot_c2:
                st.markdown("""
                    <div style="background: #07110d; padding: 1.25rem; border-radius: 12px; border: 1px solid #fbbf24; height: 100%;">
                        <h4 style="color: #fbbf24; margin-top: 0;">⚠️ Weaknesses</h4>
                        <ul style="font-size: 0.85rem; color: #ecfdf5; padding-left: 1.2rem; margin-bottom: 0;">
                            <li>Price competition pressure impacting automotive gross margins.</li>
                            <li>High capital intensity for scaling gigafactories and next-gen vehicle architecture.</li>
                            <li>Regulatory and safety scrutiny regarding driver-assistance systems.</li>
                        </ul>
                    </div>
                """, unsafe_allow_html=True)
                
            st.markdown("<br>", unsafe_allow_html=True)
            swot_c3, swot_c4 = st.columns(2)
            with swot_c3:
                st.markdown("""
                    <div style="background: #07110d; padding: 1.25rem; border-radius: 12px; border: 1px solid #34d399; height: 100%;">
                        <h4 style="color: #34d399; margin-top: 0;">🚀 Opportunities</h4>
                        <ul style="font-size: 0.85rem; color: #ecfdf5; padding-left: 1.2rem; margin-bottom: 0;">
                            <li>Scaling commercial energy storage deployment (Megapack grid solutions).</li>
                            <li>Monetization of Full Self-Driving (FSD) software and robotaxi network services.</li>
                            <li>Expansion into emerging global markets and next-gen affordable compact platforms.</li>
                        </ul>
                    </div>
                """, unsafe_allow_html=True)
            with swot_c4:
                st.markdown("""
                    <div style="background: #07110d; padding: 1.25rem; border-radius: 12px; border: 1px solid #f43f5e; height: 100%;">
                        <h4 style="color: #f43f5e; margin-top: 0;">🛡️ Threats</h4>
                        <ul style="font-size: 0.85rem; color: #ecfdf5; padding-left: 1.2rem; margin-bottom: 0;">
                            <li>Aggressive EV market share competition from traditional OEMs and global entrants.</li>
                            <li>Macroeconomic sensitivity to high interest rates impacting consumer financing.</li>
                            <li>Geopolitical supply chain vulnerabilities and raw material cost volatility.</li>
                        </ul>
                    </div>
                """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

elif current == "Document Library":
    st.markdown('<div class="glow-card">', unsafe_allow_html=True)
    st.subheader("📁 Corporate Document Library")
    st.caption("Manage indexed PDF annual reports, 10-Ks, and earnings transcripts.")
    
    uploaded = st.file_uploader("Upload PDF Annual Reports / 10-Ks", type=["pdf"], accept_multiple_files=True)
    if st.button("Save Uploaded Reports", key="btn_save"):
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        for file in uploaded or []:
            (REPORTS_DIR / file.name).write_bytes(file.getvalue())
        st.success(f"Successfully saved {len(uploaded or [])} report(s).")
        
    st.markdown("---")
    if st.button("Process PDFs and Build Searchable Index", key="btn_index"):
        with st.spinner("Extracting PDF text and creating embeddings…"):
            chunks = ingest_pdfs(REPORTS_DIR)
            if not chunks:
                st.error("No extractable text found in PDFs.")
            else:
                build_or_load_index(chunks, INDEX_DIR)
                st.success(f"Successfully indexed {len(chunks)} text chunks across your reports!")
    st.markdown('</div>', unsafe_allow_html=True)