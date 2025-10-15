import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from ai_agent import (
    analyze_data, 
    quick_summary, 
    suggest_questions, 
    validate_dataframe,
    export_analysis_report,
    get_column_insights,
    COLOR_SCHEMES
)
import io
import time
from datetime import datetime
import numpy as np

# Page Setup
st.set_page_config(
    page_title="AI Data Insights Agent | Squid Game Edition",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Theme State
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'

if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

if 'current_question' not in st.session_state:
    st.session_state.current_question = ""

if 'last_result' not in st.session_state:
    st.session_state.last_result = None

# Theme CSS Function
def get_theme_css(theme='dark'):
    if theme == 'dark':
        return """<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
* { font-family: 'Inter', sans-serif; }
.main-header { background: linear-gradient(135deg, #8C205C 0%, #BF4B96 35%, #A6859D 65%, #0E7373 100%); padding: 3rem; border-radius: 20px; margin-bottom: 2rem; box-shadow: 0 15px 40px rgba(140, 32, 92, 0.4); position: relative; overflow: hidden; }
.main-header h1 { color: #FFFFFF !important; font-size: 3.2rem; font-weight: 800; text-shadow: 2px 2px 8px rgba(0, 0, 0, 0.8); z-index: 1; position: relative; }
.main-header p { color: #FFFFFF !important; font-size: 1.3rem; font-weight: 600; text-shadow: 1px 1px 4px rgba(0, 0, 0, 0.6); z-index: 1; position: relative; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #1A1A1A 0%, #0F0F0F 100%) !important; border-right: 3px solid #BF4B96 !important; }
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: #BF4B96 !important; font-weight: 700; text-shadow: 0 0 10px rgba(191, 75, 150, 0.5); }
[data-testid="stSidebar"] p, [data-testid="stSidebar"] label, [data-testid="stSidebar"] span { color: #E8E8E8 !important; }
[data-testid="stSidebar"] .stMarkdown ul li { color: #CCCCCC !important; }
.stButton > button { border-radius: 10px; font-weight: 700; padding: 0.7rem 2rem; box-shadow: 0 4px 15px rgba(0,0,0,0.15); }
.stButton > button[kind="primary"] { background: linear-gradient(135deg, #8C205C 0%, #BF4B96 100%) !important; color: #FFFFFF !important; }
[data-testid="stSidebar"] .stButton > button { background: linear-gradient(135deg, #8C205C 0%, #BF4B96 100%) !important; color: #FFFFFF !important; }
[data-testid="stFileUploader"] { border: 3px dashed #0E7373 !important; border-radius: 15px; background: linear-gradient(145deg, #2A2A2A 0%, #1A1A1A 100%) !important; }
[data-testid="stFileUploader"] label, [data-testid="stFileUploader"] span { color: #FFFFFF !important; }
[data-testid="stSidebar"] .stSelectbox > div > div { background-color: #2A2A2A !important; border: 2px solid #0E7373 !important; color: #FFFFFF !important; }
[data-testid="stSidebar"] .stSelectbox label, [data-testid="stSidebar"] .stSlider label, [data-testid="stSidebar"] .stCheckbox label { color: #FFFFFF !important; font-weight: 600; }
.stTabs [data-baseweb="tab-list"] { gap: 12px; background: linear-gradient(135deg, #2A2A2A 0%, #1A1A1A 100%); padding: 12px; border-radius: 15px; }
.stTabs [data-baseweb="tab"] { padding: 14px 28px; border-radius: 10px; font-weight: 700; background: linear-gradient(145deg, #3A3A3A 0%, #2A2A2A 100%); color: #FFFFFF !important; }
.stTabs [data-baseweb="tab"]:hover { background: linear-gradient(145deg, #0E7373 0%, #0A5555 100%); color: #ADD9D1 !important; }
.stTabs [aria-selected="true"] { background: linear-gradient(135deg, #8C205C 0%, #BF4B96 100%) !important; color: #FFFFFF !important; }
.streamlit-expanderHeader { background: linear-gradient(145deg, #2A2A2A 0%, #1A1A1A 100%); border-radius: 10px; color: #FFFFFF !important; border: 2px solid #3A3A3A; }
[data-testid="stSidebar"] .streamlit-expanderHeader { background: linear-gradient(145deg, #2A2A2A 0%, #1A1A1A 100%) !important; color: #FFFFFF !important; }
[data-testid="stSidebar"] .streamlit-expanderContent { background-color: #1A1A1A !important; color: #CCCCCC !important; }
.info-box { background: linear-gradient(135deg, #0E4F4F 0%, #0A3838 100%); padding: 2rem; border-radius: 15px; border-left: 6px solid #0E7373; color: #FFFFFF; }
.info-box h3 { color: #ADD9D1 !important; }
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3 { color: #BF4B96 !important; font-weight: 700; }
.stTextInput input, .stTextArea textarea { border: 2px solid #0E7373 !important; background-color: #2A2A2A !important; color: #FFFFFF !important; }
div[data-testid="stMetricValue"] { font-size: 2.2rem; font-weight: 800; background: linear-gradient(135deg, #8C205C 0%, #BF4B96 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
div[data-testid="stMetricLabel"] { color: #BF4B96 !important; font-weight: 600; }
::-webkit-scrollbar { width: 12px; }
::-webkit-scrollbar-track { background: #1A1A1A; }
::-webkit-scrollbar-thumb { background: linear-gradient(135deg, #8C205C 0%, #BF4B96 100%); border-radius: 10px; }
.stProgress > div > div { background: linear-gradient(90deg, #8C205C 0%, #BF4B96 50%, #0E7373 100%) !important; }
#MainMenu {visibility: hidden;} footer {visibility: hidden;}
hr { border-top: 2px solid #BF4B96; opacity: 0.3; }
</style>"""
    else:  # light mode
        return """<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
* { font-family: 'Inter', sans-serif; }
.main-header { background: linear-gradient(135deg, #8C205C 0%, #BF4B96 35%, #A6859D 65%, #0E7373 100%); padding: 3rem; border-radius: 20px; margin-bottom: 2rem; box-shadow: 0 15px 40px rgba(140, 32, 92, 0.4); }
.main-header h1 { color: #FFFFFF !important; font-size: 3.2rem; font-weight: 800; text-shadow: 2px 2px 8px rgba(0, 0, 0, 0.8); }
.main-header p { color: #FFFFFF !important; font-size: 1.3rem; font-weight: 600; text-shadow: 1px 1px 4px rgba(0, 0, 0, 0.6); }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #F8F9FA 0%, #E8E9EB 100%) !important; border-right: 3px solid #BF4B96 !important; }
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: #8C205C !important; font-weight: 700; }
[data-testid="stSidebar"] p, [data-testid="stSidebar"] label, [data-testid="stSidebar"] span { color: #2A2A2A !important; }
[data-testid="stSidebar"] .stMarkdown ul li { color: #333333 !important; }
.stButton > button { border-radius: 10px; font-weight: 700; padding: 0.7rem 2rem; }
.stButton > button[kind="primary"] { background: linear-gradient(135deg, #8C205C 0%, #BF4B96 100%) !important; color: #FFFFFF !important; }
[data-testid="stSidebar"] .stButton > button { background: linear-gradient(135deg, #8C205C 0%, #BF4B96 100%) !important; color: #FFFFFF !important; }
[data-testid="stFileUploader"] { border: 3px dashed #0E7373 !important; border-radius: 15px; background: linear-gradient(145deg, #FFFFFF 0%, #F5F5F5 100%) !important; }
[data-testid="stFileUploader"] label, [data-testid="stFileUploader"] span { color: #1A1A1A !important; }
[data-testid="stSidebar"] .stSelectbox > div > div { background-color: #FFFFFF !important; border: 2px solid #0E7373 !important; color: #1A1A1A !important; }
[data-testid="stSidebar"] .stSelectbox label, [data-testid="stSidebar"] .stSlider label, [data-testid="stSidebar"] .stCheckbox label { color: #1A1A1A !important; font-weight: 600; }
.stTabs [data-baseweb="tab-list"] { gap: 12px; background: linear-gradient(135deg, #F8F9FA 0%, #E8E9EB 100%); padding: 12px; border-radius: 15px; }
.stTabs [data-baseweb="tab"] { padding: 14px 28px; border-radius: 10px; font-weight: 700; background: linear-gradient(145deg, #FFFFFF 0%, #F5F5F5 100%); color: #8C205C !important; }
.stTabs [data-baseweb="tab"]:hover { background: linear-gradient(145deg, #0E7373 0%, #0A5555 100%); color: #FFFFFF !important; }
.stTabs [aria-selected="true"] { background: linear-gradient(135deg, #8C205C 0%, #BF4B96 100%) !important; color: #FFFFFF !important; }
.streamlit-expanderHeader { background: linear-gradient(145deg, #FFFFFF 0%, #F5F5F5 100%); border-radius: 10px; color: #8C205C !important; border: 2px solid #E8E9EB; }
[data-testid="stSidebar"] .streamlit-expanderHeader { background: linear-gradient(145deg, #FFFFFF 0%, #F5F5F5 100%) !important; color: #8C205C !important; }
[data-testid="stSidebar"] .streamlit-expanderContent { background-color: #F8F9FA !important; color: #333333 !important; }
.info-box { background: linear-gradient(135deg, #E8F5F3 0%, #D4EBE7 100%); padding: 2rem; border-radius: 15px; border-left: 6px solid #0E7373; color: #1A1A1A; }
.info-box h3 { color: #0E7373 !important; }
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3 { color: #8C205C !important; font-weight: 700; }
.stTextInput input, .stTextArea textarea { border: 2px solid #0E7373 !important; background-color: #FFFFFF !important; color: #1A1A1A !important; }
div[data-testid="stMetricValue"] { font-size: 2.2rem; font-weight: 800; background: linear-gradient(135deg, #8C205C 0%, #BF4B96 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
div[data-testid="stMetricLabel"] { color: #8C205C !important; font-weight: 600; }
::-webkit-scrollbar { width: 12px; }
::-webkit-scrollbar-track { background: #F8F9FA; }
::-webkit-scrollbar-thumb { background: linear-gradient(135deg, #8C205C 0%, #BF4B96 100%); border-radius: 10px; }
.stProgress > div > div { background: linear-gradient(90deg, #8C205C 0%, #BF4B96 50%, #0E7373 100%) !important; }
#MainMenu {visibility: hidden;} footer {visibility: hidden;}
hr { border-top: 2px solid #BF4B96; opacity: 0.3; }
</style>"""

# Apply theme
st.markdown(get_theme_css(st.session_state.theme), unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🎯 AI Data Insights Agent</h1>
    <p>💎 Squid Game Edition | ⚙️ Powered by Groq Llama 3.1 | 🚀 Transform data into stunning insights</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 🎨 Theme")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🌙 Dark", key="dark", use_container_width=True, type="primary" if st.session_state.theme == 'dark' else "secondary"):
            st.session_state.theme = 'dark'
            st.rerun()
    with col2:
        if st.button("☀️ Light", key="light", use_container_width=True, type="primary" if st.session_state.theme == 'light' else "secondary"):
            st.session_state.theme = 'light'
            st.rerun()
    
    st.divider()
    st.markdown("## ⚙️ Configuration")
    st.markdown("### 📁 Upload Dataset")
    uploaded_file = st.file_uploader("Choose CSV/Excel", type=["csv", "xlsx", "xls"], label_visibility="collapsed")
    
    st.divider()
    st.markdown("### 📊 Settings")
    chart_choice = st.selectbox("Chart Type", ["auto", "bar", "line", "pie", "scatter", "heatmap", "none"], index=0)
    color_scheme = st.selectbox("Color Theme", list(COLOR_SCHEMES.keys()), index=1, format_func=lambda x: x.replace('_', ' ').title())
    
    with st.expander("🎨 Color Preview"):
        cols = st.columns(5)
        for idx, color in enumerate(COLOR_SCHEMES[color_scheme]):
            with cols[idx % 5]:
                color_text = "#CCCCCC" if st.session_state.theme == 'dark' else "#666666"
                st.markdown(f"<div style='background:{color}; height:40px; border-radius:8px;'></div><p style='text-align:center; font-size:0.7rem; color:{color_text};'>{color}</p>", unsafe_allow_html=True)
    
    temperature = st.slider("🎯 AI Creativity", 0.0, 1.0, 0.1, 0.1)
    
    st.divider()
    st.markdown("### 🎛️ Display")
    col_o1, col_o2 = st.columns(2)
    with col_o1:
        show_code = st.checkbox("💻 Code", True)
        show_stats = st.checkbox("📊 Stats", True)
    with col_o2:
        show_raw_data = st.checkbox("📋 Data", True)
        show_insights = st.checkbox("💡 Insights", True)
    
    st.divider()
    st.markdown("### ⚡ Actions")
    if st.button("🔄 Clear Cache", use_container_width=True):
        st.cache_data.clear()
        st.toast("✅ Cache cleared!", icon="🔄")
    
    if st.button("📜 History", use_container_width=True):
        if st.session_state.analysis_history:
            with st.expander("📋 Recent", expanded=True):
                for i, item in enumerate(reversed(st.session_state.analysis_history[-5:])):
                    st.markdown(f"**{i+1}.** {item['question']}")
                    st.caption(f"🕒 {item['timestamp']}")
        else:
            st.info("📭 No history yet!")
    
    st.divider()
    
    with st.expander("💡 Examples"):
        st.markdown("""
**📊 Analysis:**
- Average revenue by region?
- Top 10 products by sales
- Show correlation heatmap

**📈 Trends:**
- Sales trend over time
- Monthly growth rate

**🔍 Distribution:**
- Customer age distribution
- Price range breakdown
""")
    
    with st.expander("ℹ️ About"):
        st.markdown(f"""
**Version:** 2.0 Squid Game Edition  
**Theme:** {st.session_state.theme.title()} Mode  
**AI Model:** Groq Llama 3.1-8B  

**Features:**
- 🎨 9 color palettes
- 🌙 Dark/Light toggle
- 🚀 Fast processing
- 📊 Auto charts
""")

# Main Content
if uploaded_file:
    try:
        # Load data
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        # Parse dates
        for col in df.columns:
            if "date" in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                except:
                    pass
        
        # Validate
        is_valid, msg = validate_dataframe(df)
        if not is_valid:
            st.error(msg)
            st.stop()
        
        st.success(f"✅ Loaded **{uploaded_file.name}** ({len(df):,} rows × {len(df.columns)} cols)")
        
        # Overview
        st.markdown("### 📊 Dataset Overview")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("📊 Rows", f"{len(df):,}")
        with col2:
            st.metric("📋 Columns", len(df.columns))
        with col3:
            st.metric("🔢 Numeric", len(df.select_dtypes(include='number').columns))
        with col4:
            st.metric("📝 Text", len(df.select_dtypes(include='object').columns))
        with col5:
            missing = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100)
            st.metric("⚠️ Missing", f"{missing:.1f}%")
        
        st.divider()
        
        # Tabs
        tab1, tab2, tab3, tab4 = st.tabs(["🔍 AI Analysis", "📊 Data Overview", "📈 Stats", "🎯 Insights"])
        
        with tab1:
            st.markdown("### 💬 Ask Your Data Anything")
            
            if show_insights:
                with st.expander("💡 Suggested Questions"):
                    suggestions = suggest_questions(df)
                    cols = st.columns(2)
                    for idx, suggestion in enumerate(suggestions):
                        with cols[idx % 2]:
                            if st.button(suggestion, key=f"suggest_{idx}"):
                                st.session_state.current_question = suggestion.split(" ", 1)[1] if " " in suggestion else suggestion
                                st.rerun()
            
            question = st.text_area("Question", st.session_state.current_question, 
                                   placeholder="e.g., What are the top 5 products?", height=100)
            
            col_b1, col_b2, col_b3, _ = st.columns([2, 1, 1, 4])
            with col_b1:
                analyze_btn = st.button("🚀 Analyze Now", type="primary")
            with col_b2:
                if st.button("🔄 Clear"):
                    st.session_state.current_question = ""
                    st.session_state.last_result = None
                    st.rerun()
            with col_b3:
                if st.session_state.last_result:
                    export_btn = st.button("📥 Export")
            
            if question and analyze_btn:
                start = time.time()
                with st.spinner("🤖 Analyzing..."):
                    try:
                        result, code = analyze_data(df, question, chart_choice, color_scheme, temperature=temperature)
                        elapsed = time.time() - start
                        
                        st.divider()
                        st.caption(f"⚡ Done in {elapsed:.2f}s with {color_scheme.replace('_', ' ').title()} in {st.session_state.theme} mode")
                        
                        if show_code and code:
                            with st.expander("🧩 Generated Code"):
                                st.code(code, language="python", line_numbers=True)
                                st.download_button("📋 Download Code", code, f"code_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py", "text/plain", key="dl_code")
                        
                        exec_locals = {"df": df.copy(), "pd": pd, "plt": plt, "np": np}
                        
                        try:
                            exec(code, {}, exec_locals)
                            result = exec_locals.get("result")
                            
                            fig = plt.gcf()
                            if fig.axes:
                                st.markdown("### 📊 Visualization")
                                st.pyplot(fig)
                                
                                buf = io.BytesIO()
                                fig.savefig(buf, format='png', dpi=300, bbox_inches='tight', facecolor='white')
                                buf.seek(0)
                                st.download_button("📥 Download Chart", buf, f"chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png", "image/png", key="dl_chart")
                                st.toast("📊 Chart generated!", icon="✅")
                                
                                plt.clf()
                                plt.close('all')
                            
                            if result is not None:
                                st.markdown("### 📋 Result")
                                if isinstance(result, pd.DataFrame):
                                    st.dataframe(result, use_container_width=True, height=400)
                                    csv = result.to_csv(index=True).encode('utf-8')
                                    st.download_button("📥 Download CSV", csv, f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", "text/csv", key="dl_csv")
                                    
                                    numeric_cols = result.select_dtypes(include='number').columns
                                    if len(numeric_cols) > 0:
                                        with st.expander("📊 Statistics"):
                                            st.dataframe(result[numeric_cols].describe())
                                
                                elif isinstance(result, pd.Series):
                                    st.dataframe(result.to_frame(), use_container_width=True)
                                elif isinstance(result, (list, dict)):
                                    st.json(result)
                                else:
                                    st.markdown(f"**Result:** `{result}`")
                                
                                st.toast("✅ Analysis complete!", icon="🎉")
                                
                                st.session_state.analysis_history.append({
                                    'question': question,
                                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                    'theme': st.session_state.theme,
                                    'success': True
                                })
                                
                                st.session_state.last_result = {'result': result, 'code': code, 'question': question}
                        
                        except Exception as e:
                            st.error(f"⚠️ **Execution Error:** {e}")
                            with st.expander("💡 Tips"):
                                st.markdown("""
**Common Issues:**
1. **Column not found:** Check names match exactly
2. **Type errors:** Verify data types
3. **Empty results:** Try broader query
4. **Syntax errors:** Rephrase question

**Try:**
- Simplifying your question
- Using exact column names
- Breaking into smaller parts
""")
                    
                    except Exception as e:
                        st.error(f"⚠️ **Error:** {e}")
                        st.toast("❌ Analysis failed", icon="⚠️")
        
        with tab2:
            st.markdown("### 🗂️ Dataset Preview")
            
            col_r1, col_r2 = st.columns([3, 1])
            with col_r1:
                num_rows = st.slider("Rows to display", 5, min(100, len(df)), min(20, len(df)), key="preview_rows")
            with col_r2:
                view_mode = st.selectbox("View", ["Head", "Tail", "Random"], key="view_mode")
            
            if view_mode == "Head":
                display_df = df.head(num_rows)
            elif view_mode == "Tail":
                display_df = df.tail(num_rows)
            else:
                display_df = df.sample(min(num_rows, len(df)))
            
            st.dataframe(display_df, use_container_width=True, height=400)
            
            csv_full = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Full Dataset", csv_full, f"dataset_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv", key="dl_dataset")
            
            st.divider()
            
            st.markdown("### 📋 Column Information")
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes.astype(str),
                'Non-Null': df.count(),
                'Null': df.isnull().sum(),
                'Null %': (df.isnull().sum() / len(df) * 100).round(2),
                'Unique': [df[col].nunique() for col in df.columns]
            })
            st.dataframe(col_info, use_container_width=True)
            
            st.divider()
            
            missing_data = df.isnull().sum()
            if missing_data.sum() > 0:
                st.markdown("### 🔍 Missing Values")
                missing_df = missing_data[missing_data > 0].sort_values(ascending=False)
                
                col_m1, col_m2 = st.columns([2, 1])
                with col_m1:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    colors = COLOR_SCHEMES[color_scheme]
                    from matplotlib.colors import LinearSegmentedColormap
                    color_map = LinearSegmentedColormap.from_list("custom", colors)
                    colors_mapped = [color_map(i/len(missing_df)) for i in range(len(missing_df))]
                    missing_df.plot(kind='barh', ax=ax, color=colors_mapped, edgecolor='white', linewidth=2)
                    ax.set_title('Missing Values', fontsize=14, fontweight='bold', color='#8C205C')
                    ax.set_xlabel('Count')
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                
                with col_m2:
                    st.markdown("**Summary:**")
                    for col, count in missing_df.items():
                        pct = (count / len(df) * 100)
                        st.metric(col, f"{count:,}", f"{pct:.1f}%")
            else:
                st.success("✅ No missing values!")
        
        with tab3:
            st.markdown("### 📈 Statistical Summary")
            summary = quick_summary(df)
            
            st.markdown("#### 📊 Dataset Stats")
            cols = st.columns(4)
            for idx, (key, val) in enumerate(summary.items()):
                with cols[idx % 4]:
                    st.metric(key, val)
            
            st.divider()
            
            numeric_cols = df.select_dtypes(include='number').columns
            if len(numeric_cols) > 0:
                st.markdown("#### 🔢 Numeric Columns")
                st.dataframe(df[numeric_cols].describe(), use_container_width=True)
                
                if len(numeric_cols) > 1:
                    st.divider()
                    st.markdown("#### 🔗 Correlation")
                    
                    corr = df[numeric_cols].corr()
                    fig, ax = plt.subplots(figsize=(12, 8))
                    
                    from matplotlib.colors import LinearSegmentedColormap
                    custom_cmap = LinearSegmentedColormap.from_list("custom", COLOR_SCHEMES[color_scheme])
                    im = ax.imshow(corr, cmap=custom_cmap, aspect='auto', vmin=-1, vmax=1)
                    
                    cbar = plt.colorbar(im, ax=ax)
                    cbar.set_label('Correlation', rotation=270, labelpad=20)
                    
                    ax.set_xticks(range(len(corr.columns)))
                    ax.set_yticks(range(len(corr.columns)))
                    ax.set_xticklabels(corr.columns, rotation=45, ha='right')
                    ax.set_yticklabels(corr.columns)
                    
                    for i in range(len(corr)):
                        for j in range(len(corr)):
                            text = ax.text(j, i, f'{corr.iloc[i, j]:.2f}', ha="center", va="center", 
                                         color="white" if abs(corr.iloc[i, j]) > 0.5 else "#1A1A1A", fontsize=9, fontweight='bold')
                    
                    ax.set_title('Correlation Heatmap', fontsize=14, fontweight='bold', color='#8C205C')
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
            else:
                st.info("ℹ️ No numeric columns")
            
            st.divider()
            
            cat_cols = df.select_dtypes(include='object').columns
            if len(cat_cols) > 0:
                st.markdown("#### 📝 Categorical Analysis")
                selected_cat = st.selectbox("Select column:", cat_cols, key="cat_analysis")
                
                if selected_cat:
                    col_c1, col_c2, col_c3, col_c4 = st.columns(4)
                    with col_c1:
                        st.metric("Unique", df[selected_cat].nunique())
                    with col_c2:
                        mode_val = df[selected_cat].mode()[0] if len(df[selected_cat].mode()) > 0 else "N/A"
                        st.metric("Most Common", mode_val)
                    with col_c3:
                        st.metric("Missing", df[selected_cat].isnull().sum())
                    with col_c4:
                        st.metric("Missing %", f"{(df[selected_cat].isnull().sum() / len(df) * 100):.1f}%")
                    
                    value_counts = df[selected_cat].value_counts().head(15)
                    
                    col_v1, col_v2 = st.columns([2, 1])
                    with col_v1:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        colors_list = COLOR_SCHEMES[color_scheme] * 3
                        value_counts.plot(kind='barh', ax=ax, color=colors_list[:len(value_counts)], edgecolor='white', linewidth=2)
                        ax.set_title(f'Top 15 Values in {selected_cat}', fontsize=14, fontweight='bold', color='#8C205C')
                        ax.set_xlabel('Count')
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                    
                    with col_v2:
                        st.markdown("**Frequency:**")
                        freq_df = pd.DataFrame({
                            'Value': value_counts.index,
                            'Count': value_counts.values,
                            '%': (value_counts.values / len(df) * 100).round(2)
                        })
                        st.dataframe(freq_df, height=400)
        
        with tab4:
            st.markdown("### 🎯 Column Insights")
            selected_col = st.selectbox("Select column:", df.columns, key="insight_col")
            
            if selected_col:
                insights = get_column_insights(df, selected_col)
                
                if "error" not in insights:
                    col_i1, col_i2, col_i3, col_i4 = st.columns(4)
                    with col_i1:
                        st.metric("Type", insights['dtype'])
                    with col_i2:
                        st.metric("Count", f"{insights['count']:,}")
                    with col_i3:
                        st.metric("Missing", f"{insights['null_count']:,}")
                    with col_i4:
                        st.metric("Missing %", insights['null_percentage'])
                    
                    st.divider()
                    
                    if pd.api.types.is_numeric_dtype(df[selected_col]):
                        st.markdown("#### 📊 Numeric Statistics")
                        col_n1, col_n2, col_n3, col_n4 = st.columns(4)
                        with col_n1:
                            st.metric("Mean", f"{insights['mean']:.2f}")
                            st.metric("Std", f"{insights['std']:.2f}")
                        with col_n2:
                            st.metric("Median", f"{insights['median']:.2f}")
                            st.metric("Min", f"{insights['min']:.2f}")
                        with col_n3:
                            st.metric("Q1", f"{insights['q25']:.2f}")
                            st.metric("Q3", f"{insights['q75']:.2f}")
                        with col_n4:
                            st.metric("Max", f"{insights['max']:.2f}")
                            iqr = insights['q75'] - insights['q25']
                            st.metric("IQR", f"{iqr:.2f}")
                        
                        st.divider()
                        st.markdown("#### 📈 Distribution")
                        
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                        df[selected_col].dropna().hist(bins=30, ax=ax1, color=COLOR_SCHEMES[color_scheme][1], edgecolor=COLOR_SCHEMES[color_scheme][0], linewidth=1.5)
                        ax1.set_title(f'Histogram of {selected_col}', fontweight='bold', color='#8C205C')
                        ax1.set_xlabel(selected_col)
                        ax1.set_ylabel('Frequency')
                        ax1.axvline(insights['mean'], color=COLOR_SCHEMES[color_scheme][3], linestyle='--', linewidth=2, label=f"Mean: {insights['mean']:.2f}")
                        ax1.axvline(insights['median'], color=COLOR_SCHEMES[color_scheme][4], linestyle='--', linewidth=2, label=f"Median: {insights['median']:.2f}")
                        ax1.legend()
                        
                        df[selected_col].dropna().plot(kind='box', ax=ax2)
                        ax2.set_title(f'Box Plot of {selected_col}', fontweight='bold', color='#8C205C')
                        ax2.set_ylabel(selected_col)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                    
                    elif pd.api.types.is_object_dtype(df[selected_col]):
                        st.markdown("#### 📝 Categorical Statistics")
                        col_cat1, col_cat2 = st.columns(2)
                        with col_cat1:
                            st.metric("Unique", f"{insights['unique_count']:,}")
                        with col_cat2:
                            st.metric("Most Frequent", insights['top_value'])
                        
                        st.divider()
                        st.markdown("#### 🏆 Top Values")
                        
                        top_values = df[selected_col].value_counts().head(10)
                        col_t1, col_t2 = st.columns([3, 2])
                        
                        with col_t1:
                            fig, ax = plt.subplots(figsize=(10, 6))
                            colors_list = COLOR_SCHEMES[color_scheme] * 2
                            top_values.plot(kind='bar', ax=ax, color=colors_list[:len(top_values)], edgecolor='white', linewidth=2)
                            ax.set_title(f'Top 10 Values in {selected_col}', fontweight='bold', color='#8C205C')
                            ax.set_xlabel(selected_col)
                            ax.set_ylabel('Count')
                            plt.xticks(rotation=45, ha='right')
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close()
                        
                        with col_t2:
                            top_df = pd.DataFrame({
                                'Value': top_values.index,
                                'Count': top_values.values,
                                '%': (top_values.values / len(df) * 100).round(2)
                            })
                            st.dataframe(top_df, height=400, use_container_width=True)
    
    except Exception as e:
        st.error(f"⚠️ **Error:** {e}")
        st.info("💡 Ensure your file is properly formatted (CSV/Excel)")

else:
    st.markdown("""
    <div class='info-box'>
        <h3>👋 Welcome to AI Data Insights Agent!</h3>
        <p>Upload your dataset from the sidebar to start analyzing with stunning Squid Game colors.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div style='text-align: center; padding: 2rem; background: rgba(140, 32, 92, 0.1); border-radius: 15px; margin: 2rem 0;'>
        <h3 style='color: {"#BF4B96" if st.session_state.theme == "dark" else "#8C205C"};'>
            🎨 Currently in {st.session_state.theme.title()} Mode
        </h3>
        <p style='color: {"#CCCCCC" if st.session_state.theme == "dark" else "#666666"}; margin-top: 1rem;'>
            Toggle between dark and light modes using the buttons in the sidebar!
        </p>
    </div>
    """, unsafe_allow_html=True)
