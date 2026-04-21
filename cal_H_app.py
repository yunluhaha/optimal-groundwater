import streamlit as st
import pandas as pd
import numpy as np
import math
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image

# ====== 1. 核心计算函数 ======
def calc_D_star(E0, SI, Sg, dc, Ky, b, EC_prime):
    theta = 0.23; rho_s = 1475.0
    I = 0.3216; delta = 0.29; f = 0.5
    alpha = 2.156; beta = 0.616; lam = 1.7256; mu = 1.3603
    eps = 0.5; omega = 0.405; q = 0.1; P = 0.0

    M = - (alpha * theta * b * Sg) / (100.0 * Ky * rho_s * f * delta * I)
    term_SI = SI * (1.0 - (1.0 - f) * delta) / (f * delta)
    N = 1.0 - (b / (Ky * 100.0)) * ((alpha * theta / rho_s) * term_SI + beta - EC_prime)

    ET_gm = lam * E0
    denom = omega * I * (1.0 - delta) + P + eps * ET_gm
    Phi = eps / denom
    Psi = (omega * I * (1.0 - delta) / (1.0 - q) + P) / denom

    inside_ln = (-1.0 / (2.0 * lam * E0)) * (N / M + Psi / Phi)

    if inside_ln <= 0:
        return None, None

    D_unc = (-1.0 / mu) * math.log(inside_ln)
    D_opt = min(dc, D_unc)
    return D_unc, D_opt

# ====== 2. 页面配置 ======
st.set_page_config(
    page_title="最优地下水埋深计算器 | Optimal Groundwater Depth",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====== 3. 全局 CSS 样式 ======
st.markdown("""
<style>
/* ---- 全局字体与背景 ---- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ---- 主背景 ---- */
.stApp {
    background: linear-gradient(135deg, #f0f7ff 0%, #e8f4f8 50%, #f0faf5 100%);
}

/* ---- Hero 标题区 ---- */
.hero-container {
    background: linear-gradient(135deg, #0a3d62 0%, #1a6b8a 40%, #1e8449 100%);
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    box-shadow: 0 8px 32px rgba(10, 61, 98, 0.25);
    position: relative;
    overflow: hidden;
}
.hero-container::before {
    content: '';
    position: absolute;
    top: -50%; right: -20%;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(255,255,255,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    color: #ffffff;
    font-size: 1.85rem;
    font-weight: 700;
    margin: 0 0 0.6rem 0;
    letter-spacing: -0.5px;
    line-height: 1.3;
}
.hero-subtitle {
    color: rgba(255,255,255,0.82);
    font-size: 0.95rem;
    font-weight: 300;
    margin: 0;
    line-height: 1.6;
}
.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.18);
    border: 1px solid rgba(255,255,255,0.3);
    color: white;
    font-size: 0.75rem;
    font-weight: 500;
    padding: 3px 10px;
    border-radius: 20px;
    margin-bottom: 0.8rem;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

/* ---- 指标卡片 ---- */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1.2rem;
    margin: 1.2rem 0;
}
.metric-card {
    background: white;
    border-radius: 14px;
    padding: 1.5rem 1.6rem;
    box-shadow: 0 2px 16px rgba(0,0,0,0.07);
    border-left: 5px solid #1a6b8a;
    transition: transform 0.2s, box-shadow 0.2s;
    position: relative;
    overflow: hidden;
}
.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 6px 24px rgba(0,0,0,0.12);
}
.metric-card.primary {
    border-left-color: #0a3d62;
    background: linear-gradient(135deg, #0a3d62 0%, #1a6b8a 100%);
    color: white;
}
.metric-card.success {
    border-left-color: #1e8449;
}
.metric-card.warning {
    border-left-color: #f39c12;
}
.metric-card.triggered {
    border-left-color: #e74c3c;
}
.metric-icon {
    font-size: 1.6rem;
    margin-bottom: 0.4rem;
    display: block;
}
.metric-label {
    font-size: 0.78rem;
    font-weight: 500;
    color: #6c757d;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 0.3rem;
}
.metric-card.primary .metric-label {
    color: rgba(255,255,255,0.75);
}
.metric-value {
    font-size: 2.1rem;
    font-weight: 700;
    color: #0a3d62;
    line-height: 1;
}
.metric-card.primary .metric-value {
    color: white;
}
.metric-status {
    display: inline-block;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 20px;
    margin-top: 0.5rem;
}
.status-triggered {
    background: #fdecea;
    color: #c0392b;
}
.status-ok {
    background: #eafaf1;
    color: #1e8449;
}
.metric-card.primary .metric-status {
    background: rgba(255,255,255,0.2);
    color: white;
}

/* ---- Section 标题 ---- */
.section-header {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin: 2rem 0 1rem 0;
    padding-bottom: 0.6rem;
    border-bottom: 2px solid #e3eaf2;
}
.section-header h3 {
    color: #0a3d62;
    font-size: 1.1rem;
    font-weight: 600;
    margin: 0;
}

/* ---- 公式卡片 ---- */
.formula-card {
    background: #f8fbff;
    border: 1px solid #d0e4f0;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin: 1rem 0;
    font-family: 'Courier New', monospace;
    color: #2c3e50;
    font-size: 0.88rem;
    line-height: 1.8;
}

/* ---- 侧边栏样式 ---- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a3d62 0%, #1a6b8a 100%);
}
[data-testid="stSidebar"] * {
    color: white !important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label {
    color: rgba(255,255,255,0.9) !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: rgba(255,255,255,0.85) !important;
}
[data-testid="stSidebar"] .stInfo {
    background: rgba(255,255,255,0.12) !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    border-radius: 10px !important;
}
[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.2) !important;
}

/* ---- 标签页 ---- */
.stTabs [data-baseweb="tab-list"] {
    background: white;
    border-radius: 10px 10px 0 0;
    padding: 0.3rem;
    gap: 0.2rem;
    border-bottom: none;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #6c757d;
    font-weight: 500;
    font-size: 0.88rem;
}
.stTabs [aria-selected="true"] {
    background: #0a3d62 !important;
    color: white !important;
}

/* ---- 错误提示 ---- */
.error-card {
    background: #fdecea;
    border: 1px solid #f5c6cb;
    border-left: 5px solid #e74c3c;
    border-radius: 10px;
    padding: 1rem 1.4rem;
    color: #721c24;
    font-weight: 500;
}

/* ---- 参数说明小标签 ---- */
.param-chip {
    display: inline-block;
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.25);
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 2px 2px;
    color: #e8f4ff;
}

/* ---- 隐藏 Streamlit 默认元素 ---- */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ====== 4. 语言选择 ======
lang_choice = st.sidebar.radio("Language / 语言", ["中文", "English"])
lang = "cn" if lang_choice == "中文" else "en"

# ====== 5. 多语言文本 ======
t = {
    "title": {
        "cn": "浅埋干旱灌区最优地下水埋深可视化系统",
        "en": "Optimal Groundwater Depth — Arid Shallow-Aquifer Irrigation"
    },
    "badge": {"cn": "水文经济解析解模型", "en": "Hydroeconomic Closed-Form Model"},
    "desc": {
        "cn": "基于水文经济解析解公式构建，支持动态参数模拟与全球典型植被生态约束参考。实时计算最优地下水埋深并可视化多参数敏感性分析。",
        "en": "Built on a hydroeconomic closed-form solution. Supports dynamic parameter simulation, ecological constraint references, and real-time sensitivity visualization."
    },
    "sidebar_crop_header": {"cn": "🌾 作物类型", "en": "🌾 Crop Type"},
    "crop_label": {"cn": "选择代表性作物", "en": "Select Representative Crop"},
    "crop_info": {
        "cn": "产量响应因子 **Ky** = `{Ky}` · 盐分减产斜率 **b** = `{b}` %/(dS/m) · 耐盐阈值 **EC'** = `{EC}` dS/m",
        "en": "Yield response **Ky** = `{Ky}` · Salinity slope **b** = `{b}` %/(dS/m) · Threshold **EC'** = `{EC}` dS/m"
    },
    "sidebar_header1": {"cn": "🎛️ 水文驱动参数", "en": "🎛️ Hydrological Parameters"},
    "E0_label": {"cn": "潜在蒸发量 E₀ (m)", "en": "Potential Evaporation E₀ (m)"},
    "SI_label": {"cn": "灌溉水矿化度 SI (mg/L)", "en": "Irrigation Water Salinity SI (mg/L)"},
    "Sg_label": {"cn": "地下水矿化度 Sg (mg/L)", "en": "Groundwater Salinity Sg (mg/L)"},
    "sidebar_header2": {"cn": "🌿 生态约束", "en": "🌿 Ecological Constraints"},
    "hv_label": {"cn": "植被根系厚度 hv (m)", "en": "Vegetation Root Thickness hv (m)"},
    "dc_info": {
        "cn": "生态红线 **dc** = `{dc:.2f} m`",
        "en": "Ecological Redline **dc** = `{dc:.2f} m`"
    },
    "expander_title": {"cn": "📖 全球潜水植被参考指南", "en": "📖 Global Phreatophyte Reference"},
    "guide_md": {
        "cn": """
| 典型植被 | 地点 | 地下水深度 (m) |
| :--- | :--- | :---: |
| 多利柽柳 (*Tamarix ramosissima*) | 美国 亚利桑那 | 1.08 ~ 3.89 |
| 骆驼刺 (*Alhagi*) | 中亚干旱区 | 2.0 ~ 2.5 |
| 牧豆树 (*Prosopis velutina*) | 美国 亚利桑那 | 1.8 ~ 4.3 |
| 赤桉 (*Eucalyptus camaldulensis*) | 澳大利亚南部 | 1.3 ~ 4.0 |
| 白杨 (*Populus fremontii*) | 美国 加州/犹他 | 0.1 ~ 3.89 |
| 无叶柽柳 (*Tamarix aphylla*) | 南亚/中东 | 2.5 ~ 3.0 |
        """,
        "en": """
| Vegetation (Species) | Location | Depth (m) |
| :--- | :--- | :---: |
| Saltcedar (*Tamarix ramosissima*) | Arizona, USA | 1.08 ~ 3.89 |
| Camel Thorn (*Alhagi*) | Arid Central Asia | 2.0 ~ 2.5 |
| Velvet Mesquite (*Prosopis velutina*) | Arizona, USA | 1.8 ~ 4.3 |
| River Red Gum (*Eucalyptus camaldulensis*) | South Australia | 1.3 ~ 4.0 |
| Fremont Cottonwood (*Populus fremontii*) | California/Utah, USA | 0.1 ~ 3.89 |
| Athel Tamarisk (*Tamarix aphylla*) | South Asia/Middle East | 2.5 ~ 3.0 |
        """
    },
    "img_caption": {
        "cn": "表2：利用稳定同位素识别地下水为植被水分来源的研究汇总",
        "en": "Table 2: Studies identifying groundwater as a plant water source via stable isotopes"
    },
    "guide_implication": {
        "cn": "> **管理启示**：柽柳属和牧豆树在埋深超过 4m 时仍能维持较高地下水利用率，设定 hv 时应考虑当地主要保护物种的生物学特性。",
        "en": "> **Management note**: Tamarix and Prosopis maintain high groundwater utilization beyond 4 m depth. Calibrate hv to local conservation species."
    },
    "res_header": {"cn": "计算结果", "en": "Calculation Results"},
    "status_trigger": {"cn": "触发生态红线约束", "en": "Ecological Redline Active"},
    "status_unconstrained": {"cn": "无约束经济最优", "en": "Unconstrained Economic Optimum"},
    "metric_D_opt": {"cn": "最终最优埋深 D*", "en": "Final Optimal Depth D*"},
    "metric_D_unc": {"cn": "理论无约束埋深", "en": "Unconstrained Depth"},
    "metric_dc": {"cn": "生态红线上限 dc", "en": "Ecological Redline dc"},
    "err_msg": {
        "cn": "⚠️ 当前参数组合下最优埋深趋于无限深（对数内部值 ≤ 0，无解析解）。请调整参数后重试。",
        "en": "⚠️ No analytical solution under current parameters (logarithm argument ≤ 0). Please adjust inputs."
    },
    "sens_header": {"cn": "敏感性分析", "en": "Sensitivity Analysis"},
    "tab_sg": {"cn": "地下水矿化度 Sg", "en": "Groundwater Salinity Sg"},
    "tab_e0": {"cn": "潜在蒸发量 E₀", "en": "Potential Evaporation E₀"},
    "tab_si": {"cn": "灌溉水矿化度 SI", "en": "Irrigation Water Salinity SI"},
    "formula_header": {"cn": "核心公式", "en": "Core Formula"},
    "formula_caption": {
        "cn": "最优地下水埋深解析解（受生态红线 dc 约束）",
        "en": "Closed-form solution for optimal groundwater depth (constrained by ecological redline dc)"
    },
}


# ====== 6. 作物数据 ======
crop_data = {
    "wheat":     {"Ky": 1.0,  "b": 7.1,  "EC_prime": 6.0},
    "maize":     {"Ky": 1.25, "b": 12.0, "EC_prime": 1.7},
    "sunflower": {"Ky": 0.95, "b": 12.0, "EC_prime": 1.7},
}
crop_keys = list(crop_data.keys())
crop_display_names = {
    "wheat":     {"cn": "小麦 (Wheat)",       "en": "Wheat"},
    "maize":     {"cn": "玉米 (Maize)",        "en": "Maize"},
    "sunflower": {"cn": "向日葵 (Sunflower)", "en": "Sunflower"},
}

# ====== 7. 侧边栏 ======
st.sidebar.markdown(f"<div style='padding:0.8rem 0 0.2rem 0'><span style='font-size:1.4rem'>💧</span> <span style='font-weight:700;font-size:1.05rem;letter-spacing:-0.3px'>GW Depth Calc</span></div>", unsafe_allow_html=True)
st.sidebar.markdown("---")

st.sidebar.header(t["sidebar_crop_header"][lang])
crop_options = [crop_display_names[k][lang] for k in crop_keys]
selected_crop_str = st.sidebar.selectbox(t["crop_label"][lang], crop_options)
selected_key = crop_keys[crop_options.index(selected_crop_str)]
Ky = crop_data[selected_key]["Ky"]
b = crop_data[selected_key]["b"]
EC_prime = crop_data[selected_key]["EC_prime"]
st.sidebar.info(t["crop_info"][lang].format(Ky=Ky, b=b, EC=EC_prime))

st.sidebar.markdown("---")
st.sidebar.header(t["sidebar_header1"][lang])
E0 = st.sidebar.slider(t["E0_label"][lang],  min_value=1.00, max_value=2.50, value=1.237, step=0.01)
SI = st.sidebar.slider(t["SI_label"][lang],  min_value=100,  max_value=3000, value=586,   step=10)
Sg = st.sidebar.slider(t["Sg_label"][lang],  min_value=1000, max_value=8000, value=4000,  step=50)

st.sidebar.markdown("---")
st.sidebar.header(t["sidebar_header2"][lang])
hv = st.sidebar.slider(t["hv_label"][lang],  min_value=0.1,  max_value=10.0, value=1.8,   step=0.1)
Dp = 1.48
dc = hv + Dp
st.sidebar.info(t["dc_info"][lang].format(dc=dc))

with st.sidebar.expander(t["expander_title"][lang], expanded=False):
    st.markdown(t["guide_md"][lang])
    image_path = "Table2_Isotopes.png"
    try:
        img = Image.open(image_path)
        st.image(img, caption=t["img_caption"][lang], use_column_width=True)
    except FileNotFoundError:
        pass
    st.markdown(t["guide_implication"][lang])


# ====== 8. 计算 ======
D_unc, D_opt = calc_D_star(E0, SI, Sg, dc, Ky, b, EC_prime)


# ====== 9. Hero 区域 ======
st.markdown(f"""
<div class="hero-container">
  <div class="hero-badge">{'🔬 ' + t['badge'][lang]}</div>
  <h1 class="hero-title">💧 {t['title'][lang]}</h1>
  <p class="hero-subtitle">{t['desc'][lang]}</p>
</div>
""", unsafe_allow_html=True)


# ====== 10. 结果指标卡 ======
_sec_res = "📊 " + t["res_header"][lang]
st.markdown(f"""<div class="section-header"><h3>{_sec_res}</h3></div>""", unsafe_allow_html=True)

if D_opt is not None:
    triggered = (D_opt == dc)
    status_label = t["status_trigger"][lang] if triggered else t["status_unconstrained"][lang]
    status_cls = "status-triggered" if triggered else "status-ok"
    card1_cls = "metric-card primary triggered" if triggered else "metric-card primary"

    st.markdown(f"""
    <div class="metric-grid">
      <div class="{card1_cls}">
        <span class="metric-icon">🎯</span>
        <div class="metric-label">{t['metric_D_opt'][lang]}</div>
        <div class="metric-value">{D_opt:.2f} m</div>
        <span class="metric-status {status_cls}">{status_label}</span>
      </div>
      <div class="metric-card success">
        <span class="metric-icon">📐</span>
        <div class="metric-label">{t['metric_D_unc'][lang]}</div>
        <div class="metric-value" style="color:#1e8449">{D_unc:.2f} m</div>
      </div>
      <div class="metric-card warning">
        <span class="metric-icon">🌿</span>
        <div class="metric-label">{t['metric_dc'][lang]}</div>
        <div class="metric-value" style="color:#f39c12">{dc:.2f} m</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # 深度对比迷你图（横向条形）
    fig_bar = go.Figure()
    colors = {"D*": "#0a3d62", "D_unc": "#1e8449", "dc": "#f39c12"}
    labels = {
        "D*": f"D* = {D_opt:.2f} m",
        "D_unc": ("D_unc" if lang == "en" else "无约束") + f" = {D_unc:.2f} m",
        "dc": f"dc = {dc:.2f} m"
    }
    for key, val in [("dc", dc), ("D_unc", D_unc), ("D*", D_opt)]:
        fig_bar.add_trace(go.Bar(
            x=[val], y=[labels[key]], orientation='h',
            marker_color=colors[key],
            marker_line_width=0,
            text=[f"{val:.2f} m"], textposition='inside',
            textfont=dict(color='white', size=13, family='Inter'),
            name=labels[key],
            hovertemplate=f"<b>{labels[key]}</b><extra></extra>"
        ))
    fig_bar.update_layout(
        height=160, margin=dict(l=10, r=20, t=10, b=10),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False, barmode='overlay',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=True,
                   tickfont=dict(size=11), title="深度 (m)" if lang == "cn" else "Depth (m)"),
        yaxis=dict(showgrid=False, tickfont=dict(size=11)),
    )
    st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

else:
    st.markdown(f'<div class="error-card">{t["err_msg"][lang]}</div>', unsafe_allow_html=True)


# ====== 11. 核心公式展示 ======
with st.expander(("📐 " + t["formula_header"][lang]) + " — " + t["formula_caption"][lang], expanded=False):
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        st.latex(r"""
        D^* = \min\!\left(d_c,\; D_{\text{unc}}\right)
        """)
        st.latex(r"""
        D_{\text{unc}} = -\frac{1}{\mu}\ln\!\left(
          -\frac{1}{2\lambda E_0}\cdot\frac{N/M + \Psi/\Phi}{1}
        \right)
        """)
    with col_f2:
        st.latex(r"""
        M = -\frac{\alpha\theta b S_g}{100\, K_y \rho_s f \delta I}, \quad
        N = 1 - \frac{b}{100 K_y}\!\left(\frac{\alpha\theta}{\rho_s}\cdot\frac{S_I(1-(1-f)\delta)}{f\delta}+\beta - EC'\right)
        """)


# ====== 12. 敏感性分析（标签页） ======
st.markdown("---")
_sec_sens = "📈 " + t["sens_header"][lang]
st.markdown(f"""<div class="section-header"><h3>{_sec_sens}</h3></div>""", unsafe_allow_html=True)

tab_sg, tab_e0, tab_si = st.tabs([
    t["tab_sg"][lang], t["tab_e0"][lang], t["tab_si"][lang]
])

# --- 通用绘图函数 ---
CHART_COLOR = "#0a3d62"
DC_COLOR    = "#e74c3c"
UNC_COLOR   = "#7fb3d3"

def make_sens_chart(x_arr, y_opt, y_unc, x_label, y_label, current_x, current_y, dc_val):
    fig = go.Figure()
    # 无约束区域填充
    fig.add_trace(go.Scatter(
        x=x_arr, y=y_unc, name="D_unc" if lang == "en" else "无约束埋深",
        line=dict(color=UNC_COLOR, width=1.5, dash="dot"),
        hovertemplate=f"{x_label}: %{{x:.1f}}<br>D_unc: %{{y:.2f}} m<extra></extra>"
    ))
    # 生态红线
    fig.add_hline(y=dc_val, line=dict(color=DC_COLOR, width=1.5, dash="dash"),
                  annotation_text=f"dc = {dc_val:.2f} m",
                  annotation_font=dict(color=DC_COLOR, size=11))
    # 最优埋深主线
    fig.add_trace(go.Scatter(
        x=x_arr, y=y_opt, name="D*",
        line=dict(color=CHART_COLOR, width=2.5),
        fill='tozeroy', fillcolor='rgba(10,61,98,0.06)',
        hovertemplate=f"{x_label}: %{{x:.1f}}<br>D*: %{{y:.2f}} m<extra></extra>"
    ))
    # 当前参数标注点
    if current_y is not None:
        fig.add_trace(go.Scatter(
            x=[current_x], y=[current_y], mode='markers',
            marker=dict(color=CHART_COLOR, size=10, symbol='circle',
                        line=dict(color='white', width=2)),
            name="当前值" if lang == "cn" else "Current",
            hovertemplate=f"当前: {x_label}=%{{x:.2f}}, D*=%{{y:.2f}} m<extra></extra>"
        ))
    fig.update_layout(
        height=340, margin=dict(l=10, r=20, t=20, b=10),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='white',
        legend=dict(orientation='h', yanchor='bottom', y=1.01, xanchor='right', x=1,
                    font=dict(size=11)),
        xaxis=dict(title=x_label, showgrid=True, gridcolor='#eef2f7',
                   tickfont=dict(size=11), title_font=dict(size=12)),
        yaxis=dict(title=y_label, showgrid=True, gridcolor='#eef2f7',
                   tickfont=dict(size=11), title_font=dict(size=12)),
        hovermode='x unified',
    )
    return fig

_y_label = "D* (m)"

with tab_sg:
    sg_arr = np.linspace(1000, 8000, 120)
    d_sg_opt = [calc_D_star(E0, SI, sg, dc, Ky, b, EC_prime)[1] for sg in sg_arr]
    d_sg_unc = [calc_D_star(E0, SI, sg, dc, Ky, b, EC_prime)[0] for sg in sg_arr]
    fig = make_sens_chart(sg_arr, d_sg_opt, d_sg_unc,
                          "Sg (mg/L)", _y_label, Sg, D_opt, dc)
    st.plotly_chart(fig, use_container_width=True)

with tab_e0:
    e0_arr = np.linspace(1.0, 2.5, 120)
    d_e0_opt = [calc_D_star(e, SI, Sg, dc, Ky, b, EC_prime)[1] for e in e0_arr]
    d_e0_unc = [calc_D_star(e, SI, Sg, dc, Ky, b, EC_prime)[0] for e in e0_arr]
    fig = make_sens_chart(e0_arr, d_e0_opt, d_e0_unc,
                          "E₀ (m)", _y_label, E0, D_opt, dc)
    st.plotly_chart(fig, use_container_width=True)

with tab_si:
    si_arr = np.linspace(100, 3000, 120)
    d_si_opt = [calc_D_star(E0, si, Sg, dc, Ky, b, EC_prime)[1] for si in si_arr]
    d_si_unc = [calc_D_star(E0, si, Sg, dc, Ky, b, EC_prime)[0] for si in si_arr]
    fig = make_sens_chart(si_arr, d_si_opt, d_si_unc,
                          "SI (mg/L)", _y_label, SI, D_opt, dc)
    st.plotly_chart(fig, use_container_width=True)
