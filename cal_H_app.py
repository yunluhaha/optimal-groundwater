import streamlit as st
import numpy as np
import math
import plotly.graph_objects as go
from PIL import Image

# ====== 1. 核心计算函数 ======
@st.cache_data(max_entries=512)
def calc_D_star(E0, SI, Sg, dc, Ky, b, EC_prime, I=0.3216, mu=1.3603):
    theta = 0.23; rho_s = 1475.0
    delta = 0.29; f = 0.5
    alpha = 2.156; beta = 0.616; lam = 1.7256
    eps = 0.5; omega = 0.405; q = 0.1; P = 0.0
    M = -(alpha * theta * b * Sg) / (100.0 * Ky * rho_s * f * delta * I)
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
    return D_unc, min(dc, D_unc)

# ====== 2. 全球典型灌区数据 ======
REGIONS = {
    "hetao": {
        "name_cn": "河套平原（中国）",       "name_en": "Hetao Plain (China)",
        "lat": 40.5, "lon": 107.5,
        "E0_range": (1.5, 2.5), "E0_default": 1.80,
        "SI_range": (200, 800), "SI_default": 400,
        "Sg_range": (2000, 7000), "Sg_default": 4000,
        "hv_range": (0.8, 1.5), "hv_default": 1.2,
        "I_range": (0.30, 0.60), "I_default": 0.45,
        "mu_range": (1.1, 1.6), "mu_default": 1.36,
        "crop_default": "sunflower",
        "desc_cn": "黄河中游干旱灌区，浅层地下水埋深 1–3 m，土壤盐碱化严重，是本模型主要研究区之一。典型作物：向日葵、小麦、玉米。",
        "desc_en": "Mid-Yellow River arid irrigation district; shallow GW 1–3 m; severe secondary salinization. Primary study region of this model. Crops: sunflower, wheat, maize.",
    },
    "ningxia": {
        "name_cn": "宁夏平原（中国）",        "name_en": "Ningxia Plain (China)",
        "lat": 37.5, "lon": 106.0,
        "E0_range": (1.2, 1.7), "E0_default": 1.40,
        "SI_range": (200, 800),  "SI_default": 400,
        "Sg_range": (2000, 6000),"Sg_default": 4000,
        "hv_range": (0.8, 1.4), "hv_default": 1.0,
        "I_range": (0.25, 0.55), "I_default": 0.38,
        "mu_range": (1.1, 1.7), "mu_default": 1.40,
        "crop_default": "wheat",
        "desc_cn": "黄河上游引黄灌区，温带大陆性气候，蒸发强烈，盐分随毛管水上升积累。典型作物：小麦、水稻、苜蓿。",
        "desc_en": "Upper Yellow River irrigation; continental arid climate; strong evaporation drives capillary salt rise. Crops: wheat, rice, alfalfa.",
    },
    "nile_delta": {
        "name_cn": "尼罗河三角洲（埃及）",   "name_en": "Nile Delta (Egypt)",
        "lat": 30.8, "lon": 31.0,
        "E0_range": (1.5, 2.0), "E0_default": 1.75,
        "SI_range": (650, 2860), "SI_default": 1500,
        "Sg_range": (5000, 8000),"Sg_default": 8000,
        "hv_range": (0.6, 1.2), "hv_default": 0.9,
        "I_range": (0.40, 0.80), "I_default": 0.60,
        "mu_range": (0.8, 1.3), "mu_default": 1.05,
        "crop_default": "wheat",
        "desc_cn": "93% 浅层地下水不适合灌溉，Na-Cl 型高矿化水；古代灌溉文明与现代盐化危机并存。典型作物：小麦、水稻、蔬菜。",
        "desc_en": "93% of shallow GW unfit for irrigation; Na-Cl brine; ancient irrigation heritage faces modern salinity crisis. Crops: wheat, rice, vegetables.",
    },
    "indus": {
        "name_cn": "印度河平原（巴基斯坦）", "name_en": "Indus Basin (Pakistan)",
        "lat": 30.0, "lon": 70.5,
        "E0_range": (1.6, 2.2), "E0_default": 1.90,
        "SI_range": (400, 1000), "SI_default": 700,
        "Sg_range": (1000, 8000),"Sg_default": 3000,
        "hv_range": (1.0, 1.6), "hv_default": 1.2,
        "I_range": (0.35, 0.70), "I_default": 0.52,
        "mu_range": (1.0, 1.5), "mu_default": 1.28,
        "crop_default": "wheat",
        "desc_cn": "南亚最大灌溉系统（16 Mha），35–40% 区域浅层含水层含盐；渍涝与盐化并发。典型作物：小麦、水稻、棉花。",
        "desc_en": "S. Asia's largest irrigation system (16 Mha); 35–40% underlain by saline aquifer; endemic waterlogging. Crops: wheat, rice, cotton.",
    },
    "murray_darling": {
        "name_cn": "墨累–达令盆地（澳大利亚）","name_en": "Murray-Darling (Australia)",
        "lat": -33.5, "lon": 143.0,
        "E0_range": (1.0, 1.8), "E0_default": 1.30,
        "SI_range": (300, 1500), "SI_default": 800,
        "Sg_range": (1500, 5000),"Sg_default": 2500,
        "hv_range": (0.8, 1.8), "hv_default": 1.2,
        "I_range": (0.20, 0.50), "I_default": 0.32,
        "mu_range": (1.3, 2.0), "mu_default": 1.62,
        "crop_default": "wheat",
        "desc_cn": "澳大利亚最重要农业区；灌溉引发地下水位上升（最高达 25 m），旱地盐化造成大面积土地退化。典型作物：小麦、棉花、水稻。",
        "desc_en": "Australia's most productive farming area; irrigation-induced water table rise (up to 25 m) drives dryland salinization. Crops: wheat, cotton, rice.",
    },
    "aral_sea": {
        "name_cn": "咸海流域（中亚）",        "name_en": "Aral Sea Basin (C. Asia)",
        "lat": 41.5, "lon": 61.0,
        "E0_range": (1.3, 1.6), "E0_default": 1.50,
        "SI_range": (600, 1500), "SI_default": 1000,
        "Sg_range": (1750, 5000),"Sg_default": 2500,
        "hv_range": (0.8, 1.2), "hv_default": 1.0,
        "I_range": (0.40, 0.75), "I_default": 0.55,
        "mu_range": (1.0, 1.5), "mu_default": 1.25,
        "crop_default": "wheat",
        "desc_cn": "咸海消亡引发的全球最大生态灾难之一；45% 灌区盐化，棉田地下水平均埋深仅 1.5 m。典型作物：棉花、苜蓿。",
        "desc_en": "One of the world's worst eco-disasters; 45% irrigated area saline; cotton fields average 1.5 m GW depth. Crops: cotton, alfalfa.",
    },
    "san_joaquin": {
        "name_cn": "圣华金河谷（美国）",       "name_en": "San Joaquin Valley (USA)",
        "lat": 36.5, "lon": -120.0,
        "E0_range": (1.2, 1.8), "E0_default": 1.50,
        "SI_range": (500, 2000), "SI_default": 1200,
        "Sg_range": (1000, 8000),"Sg_default": 4000,
        "hv_range": (1.0, 1.6), "hv_default": 1.3,
        "I_range": (0.25, 0.55), "I_default": 0.38,
        "mu_range": (1.1, 1.7), "mu_default": 1.42,
        "crop_default": "maize",
        "desc_cn": "美国最高产农业区之一；硒污染与地下水盐化并存，每年 2 Mt 盐随灌溉水输入。典型作物：棉花、苜蓿、玉米。",
        "desc_en": "Among US's most productive regions; selenium contamination co-exists with 2 Mt/yr salt import via irrigation. Crops: cotton, alfalfa, maize.",
    },
    "mesopotamia": {
        "name_cn": "美索不达米亚（伊拉克）", "name_en": "Mesopotamia (Iraq)",
        "lat": 32.0, "lon": 45.0,
        "E0_range": (1.8, 2.4), "E0_default": 2.10,
        "SI_range": (800, 3000), "SI_default": 1400,
        "Sg_range": (3000, 8000),"Sg_default": 6500,
        "hv_range": (0.8, 1.4), "hv_default": 1.1,
        "I_range": (0.45, 0.80), "I_default": 0.62,
        "mu_range": (0.8, 1.3), "mu_default": 1.08,
        "crop_default": "wheat",
        "desc_cn": "人类文明起源地；60–70% 土地已盐化；Na-Cl/Ca-Cl 型地下水毛管上升是主因。典型作物：小麦、大麦、椰枣。",
        "desc_en": "Cradle of civilization; 60–70% of land salinized; Na-Cl/Ca-Cl GW capillary rise is the primary mechanism. Crops: wheat, barley, date palms.",
    },
}
REGION_KEYS = list(REGIONS.keys())

# ====== 3. Session state 初始化 ======
_defaults = {
    "E0_s": 1.237, "SI_s": 586, "Sg_s": 4000, "hv_s": 1.8,
    "I_s": 0.3216, "mu_s": 1.3603,
    "crop_sel": "wheat", "sel_region": None, "_do_apply": False,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# 在 widgets 渲染前应用灌区参数
if st.session_state._do_apply and st.session_state.sel_region:
    r = REGIONS[st.session_state.sel_region]
    st.session_state.E0_s   = float(r["E0_default"])
    st.session_state.SI_s   = int(r["SI_default"])
    st.session_state.Sg_s   = int(r["Sg_default"])
    st.session_state.hv_s   = float(r["hv_default"])
    st.session_state.I_s    = float(r["I_default"])
    st.session_state.mu_s   = float(r["mu_default"])
    st.session_state.crop_sel = r["crop_default"]
    st.session_state._do_apply = False

# ====== 4. 页面配置 ======
st.set_page_config(
    page_title="最优地下水埋深计算器 | Optimal Groundwater Depth",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ====== 5. CSS ======
# 配色：深炭灰#2d2a24 · 沙尘蓝灰#5c8a96 · 赤陶棕#b87840 · 深朱红#a8212e · 琥珀金#c4984a · 暖羊皮纸#f5f0e8
st.markdown("""
<style>
/* ── Base ───────────────────────────────────────── */
.stApp { background-color: #f5f0e8 !important; color: #2d2a24 !important; }
.main .block-container { color: #2d2a24 !important; padding-top: 1rem !important; }
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li  { color: #3d3530 !important; font-size: 1.02rem; }
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3  { color: #2d2a24 !important; }
.stSelectbox label, .stSlider label, .stRadio label { color: #4a3f34 !important; font-size: 0.97rem !important; }

/* ── Sidebar ────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #c8d4d8 0%, #b8c8ce 60%, #a8bcc4 100%) !important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stRadio label  { color: #1a2e36 !important; font-size: 0.95rem !important; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #0f1e24 !important; font-size: 1.08rem !important; letter-spacing: 0.2px; }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.40) !important; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] li,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] strong { color: #1a2e36 !important; font-size: 0.95rem !important; }
[data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"] p { color: #1a2e36 !important; }
[data-testid="stSidebar"] .stInfo  { background: rgba(255,255,255,0.25) !important; border: 1px solid rgba(255,255,255,0.45) !important; border-radius: 8px !important; }
[data-testid="stSidebar"] .stInfo p { color: #0f1e24 !important; }
[data-testid="stSidebar"] .stExpander { background: rgba(255,255,255,0.15) !important; border: 1px solid rgba(255,255,255,0.30) !important; border-radius: 8px !important; }
[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] { background: rgba(255,255,255,0.90) !important; border-radius: 6px !important; }
[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] * { color: #1a2e36 !important; }

/* ── Tabs ───────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: #faf7f0; border-radius: 10px;
    padding: 4px 6px; gap: 4px;
    box-shadow: 0 1px 4px rgba(45,42,36,0.10);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 7px; padding: 6px 18px;
    font-size: 1.0rem; font-weight: 500; color: #7a6a58;
}
.stTabs [aria-selected="true"] {
    background: #5c8a96 !important; color: #ffffff !important;
    box-shadow: 0 2px 8px rgba(92,138,150,0.40);
}

/* ── Hero ───────────────────────────────────────── */
.hero-container {
    background: linear-gradient(135deg, #c8d4d8 0%, #b8c8ce 55%, #a8bcc4 100%);
    border-radius: 3px; padding: 0.14rem 0.45rem;
    margin-bottom: 0.5rem;
    box-shadow: 0 1px 3px rgba(168,188,196,0.20);
    display: flex; align-items: center; gap: 0.25rem;
    border-left: 1px solid #ffffff;
}
.hero-icon { font-size: 0.42rem; line-height: 1; flex-shrink: 0; }
.hero-content { flex: 1; min-width: 0; }
.hero-badge { display: none; }
.hero-title {
    color: #0f1e24; font-size: 0.2rem; font-weight: 700;
    margin: 0; line-height: 1.3;
}
.hero-subtitle { color: #1a3040; font-size: 0.32rem; margin: 0; line-height: 1.35; }

/* ── Section header ─────────────────────────────── */
.section-header {
    display: flex; align-items: center; gap: 0.5rem;
    margin: 1.5rem 0 0.8rem; padding-bottom: 0.5rem;
    border-bottom: 2px solid #b8d0d8;
    color: #7aaab8; font-size: 1.1rem; font-weight: 600;
}
.section-header::before {
    content: ''; display: inline-block;
    width: 4px; height: 18px; border-radius: 3px;
    background: #7aaab8; flex-shrink: 0;
}

/* ── Metric cards ───────────────────────────────── */
.metric-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: 0.9rem; margin: 0.9rem 0 0.5rem; }
.metric-card {
    background: #faf7f0; border-radius: 12px; padding: 1.1rem 1.3rem;
    box-shadow: 0 2px 10px rgba(45,42,36,0.08); border-left: 4px solid #5c8a96;
    transition: box-shadow 0.2s;
}
.metric-card:hover { box-shadow: 0 4px 18px rgba(45,42,36,0.13); }
.metric-card.success { border-left-color: #5c8a96; }
.metric-card.warning { border-left-color: #c4984a; }
.metric-card.danger  { border-left-color: #a8212e; }
.metric-icon  { font-size: 1.3rem; margin-bottom: 0.2rem; display: block; opacity: 0.85; }
.metric-label {
    font-size: 0.82rem; font-weight: 700; color: #7a6a58;
    text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 0.3rem;
}
.metric-value { font-size: 2.1rem; font-weight: 700; color: #5c8a96; line-height: 1; }
.metric-card.success .metric-value { color: #5c8a96; }
.metric-card.warning .metric-value { color: #b87840; }
.metric-card.danger  .metric-value { color: #a8212e; }
.metric-status {
    display: inline-block; font-size: 0.65rem; font-weight: 600;
    padding: 2px 8px; border-radius: 20px; margin-top: 0.4rem;
    background: #edf4f6; color: #3d6e7a; border: 1px solid #b0cdd4;
}
.metric-card.danger .metric-status {
    background: #fdf0f0; color: #a8212e; border-color: #e8b0b0;
}

/* ── Region card ────────────────────────────────── */
.region-card {
    background: #faf7f0; border-radius: 12px; padding: 1.3rem 1.5rem;
    box-shadow: 0 2px 12px rgba(45,42,36,0.08);
    border-top: 3px solid #b87840; margin-top: 0.8rem;
}
.region-name {
    font-size: 1.0rem; font-weight: 700; color: #2d2a24;
    margin-bottom: 0.45rem; display: flex; align-items: center; gap: 0.4rem;
}
.region-dot { width: 10px; height: 10px; border-radius: 50%; background: #a8212e; flex-shrink: 0; display: inline-block; }
.region-desc { font-size: 0.82rem; color: #5c4a38; line-height: 1.65; margin-bottom: 0.9rem; }
.param-table { width: 100%; border-collapse: collapse; font-size: 0.81rem; border-radius: 8px; overflow: hidden; }
.param-table thead tr { background: #2d2a24; }
.param-table th { color: #e8d8b8; font-weight: 600; padding: 7px 12px; text-align: left; font-size: 0.75rem; letter-spacing: 0.3px; }
.param-table td { padding: 7px 12px; color: #3d3530; border-bottom: 1px solid #ede5d8; }
.param-table tr:nth-child(even) td { background: #f5f0e8; }
.param-table tr:last-child td { border-bottom: none; }
.param-table td b { color: #b87840; }

/* ── Error card ─────────────────────────────────── */
.error-card {
    background: #fdf0f0; border-left: 4px solid #a8212e; border-radius: 10px;
    padding: 0.9rem 1.3rem; color: #a8212e; font-size: 0.88rem;
}

/* ── Map hint caption ───────────────────────────── */
.map-hint {
    font-size: 0.8rem; color: #7a6a58; margin-bottom: 0.5rem;
    display: flex; align-items: center; gap: 0.4rem;
}

/* ── Primary button ─────────────────────────────── */
.stButton > button[kind="primary"] {
    background: #9abfc8 !important;
    border: none !important;
    color: #1a2e36 !important;
}
.stButton > button[kind="primary"]:hover {
    background: #7aaab8 !important;
}

#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# ====== 6. 多语言文本 ======
t = {
    "title":   {"cn": "浅埋干旱灌区最优地下水埋深可视化系统",
                "en": "Optimal Groundwater Depth — Arid Shallow-Aquifer Irrigation"},
    "badge":   {"cn": "水文经济解析解模型", "en": "Hydroeconomic Closed-Form Model"},
    "desc":    {"cn": "基于水文经济解析解公式，支持动态参数模拟、全球典型灌区参考与敏感性分析。",
                "en": "Based on a hydroeconomic closed-form solution. Supports parameter simulation, global region references, and sensitivity analysis."},
    "tab_map": {"cn": "🗺️ 全球灌区地图", "en": "🗺️ Global Irrigation Map"},
    "tab_calc":{"cn": "📊 计算结果与分析", "en": "📊 Results & Analysis"},
    "map_title":{"cn": "全球主要干旱浅埋地下水灌区", "en": "Major Global Arid Shallow-Groundwater Irrigation Districts"},
    "map_hint": {"cn": "点击地图上的标记选择灌区，或在下方下拉框选择", "en": "Click a marker on the map to select a region, or use the dropdown below"},
    "sel_label":{"cn": "选择灌区", "en": "Select Region"},
    "sel_custom":{"cn":"— 自定义（不使用预设）","en":"— Custom (no preset)"},
    "apply_btn":{"cn": "✅ 应用该灌区参数到计算器", "en": "✅ Apply Region Parameters to Calculator"},
    "apply_ok": {"cn": "✅ 已应用参数，切换到「计算结果」标签查看结果。", "en": "✅ Parameters applied. Switch to the Results & Analysis tab."},
    "region_params_cn": ["参数", "范围", "本次默认值"],
    "region_params_en": ["Parameter", "Range", "Default"],
    "sidebar_crop_header": {"cn": "🌾 作物类型", "en": "🌾 Crop Type"},
    "crop_label": {"cn": "选择代表性作物", "en": "Select Representative Crop"},
    "crop_info": {"cn": "Ky = **{Ky}** · b = **{b}** · EC' = **{EC}** dS/m",
                  "en": "Ky = **{Ky}** · b = **{b}** · EC' = **{EC}** dS/m"},
    "sidebar_header1": {"cn": "🎛️ 水文驱动参数", "en": "🎛️ Hydrological Parameters"},
    "E0_label": {"cn": "潜在蒸发量 E₀ (m/yr)", "en": "Potential Evaporation E₀ (m/yr)"},
    "SI_label": {"cn": "灌溉水矿化度 SI (mg/L)", "en": "Irrigation Water Salinity SI (mg/L)"},
    "Sg_label": {"cn": "地下水矿化度 Sg (mg/L)", "en": "Groundwater Salinity Sg (mg/L)"},
    "sidebar_header3": {"cn": "🪨 土壤与灌溉参数", "en": "🪨 Soil & Irrigation Parameters"},
    "I_label":  {"cn": "灌溉定额 I (m)",        "en": "Irrigation Quota I (m)"},
    "mu_label": {"cn": "土壤毛管系数 μ（粗粒↑细粒↓）", "en": "Soil Capillary Coeff. μ (coarse↑ fine↓)"},
    "sidebar_header2": {"cn": "🌿 生态约束", "en": "🌿 Ecological Constraints"},
    "hv_label": {"cn": "植被根系厚度 hv (m)", "en": "Vegetation Root Thickness hv (m)"},
    "dc_info":  {"cn": "生态红线 **dc** = `{dc:.2f} m`", "en": "Ecological Redline **dc** = `{dc:.2f} m`"},
    "expander_title": {"cn": "📖 全球潜水植被参考", "en": "📖 Global Phreatophyte Reference"},
    "guide_md": {
        "cn": """| 典型植被 | 地点 | 地下水深度 (m) |
|:---|:---|:---:|
| 多利柽柳 | 美国亚利桑那 | 1.08–3.89 |
| 骆驼刺 | 中亚干旱区 | 2.0–2.5 |
| 牧豆树 | 美国亚利桑那 | 1.8–4.3 |
| 赤桉 | 澳大利亚南部 | 1.3–4.0 |
| 白杨 | 美国加州/犹他 | 0.1–3.89 |""",
        "en": """| Vegetation | Location | Depth (m) |
|:---|:---|:---:|
| Saltcedar | Arizona, USA | 1.08–3.89 |
| Camel Thorn | Arid C. Asia | 2.0–2.5 |
| Mesquite | Arizona, USA | 1.8–4.3 |
| River Red Gum | S. Australia | 1.3–4.0 |
| Cottonwood | CA/Utah, USA | 0.1–3.89 |""",
    },
    "res_header":  {"cn": "计算结果", "en": "Calculation Results"},
    "status_trigger":    {"cn": "触发生态红线约束", "en": "Ecological Redline Active"},
    "status_unconstrained": {"cn": "无约束经济最优", "en": "Unconstrained Economic Optimum"},
    "metric_D_opt": {"cn": "最终最优埋深 D*", "en": "Final Optimal Depth D*"},
    "metric_D_unc": {"cn": "理论无约束埋深",   "en": "Unconstrained Depth"},
    "metric_dc":    {"cn": "生态红线上限 dc",   "en": "Ecological Redline dc"},
    "err_msg":      {"cn": "⚠️ 当前参数组合无解析解（对数内部值 ≤ 0），请调整参数。",
                     "en": "⚠️ No analytical solution under current parameters. Please adjust inputs."},
    "sens_header": {"cn": "敏感性分析", "en": "Sensitivity Analysis"},
    "formula_header":  {"cn": "核心公式", "en": "Core Formula"},
    "formula_caption": {"cn": "最优地下水埋深解析解（受生态红线 dc 约束）",
                        "en": "Closed-form solution (constrained by ecological redline dc)"},
    "tab_sg": {"cn": "地下水矿化度 Sg", "en": "Groundwater Salinity Sg"},
    "tab_e0": {"cn": "潜在蒸发量 E₀",   "en": "Potential Evaporation E₀"},
    "tab_si": {"cn": "灌溉水矿化度 SI",  "en": "Irrigation Water Salinity SI"},
}

# ====== 7. 作物数据 ======
crop_data = {
    "wheat":     {"Ky": 1.00, "b": 7.1,  "EC_prime": 6.0},
    "maize":     {"Ky": 1.25, "b": 12.0, "EC_prime": 1.7},
    "sunflower": {"Ky": 0.95, "b": 12.0, "EC_prime": 1.7},
}
crop_keys = list(crop_data.keys())
crop_display_names = {
    "wheat":     {"cn": "小麦 (Wheat)",       "en": "Wheat"},
    "maize":     {"cn": "玉米 (Maize)",        "en": "Maize"},
    "sunflower": {"cn": "向日葵 (Sunflower)", "en": "Sunflower"},
}

# ====== 8. 侧边栏 ======
lang_choice = st.sidebar.radio("Language / 语言", ["中文", "English"])
lang = "cn" if lang_choice == "中文" else "en"

st.sidebar.markdown("---")
st.sidebar.header(t["sidebar_crop_header"][lang])
crop_options = [crop_display_names[k][lang] for k in crop_keys]
crop_default_idx = crop_keys.index(st.session_state.crop_sel) if st.session_state.crop_sel in crop_keys else 0
selected_crop_str = st.sidebar.selectbox(t["crop_label"][lang], crop_options, index=crop_default_idx)
selected_key = crop_keys[crop_options.index(selected_crop_str)]
st.session_state.crop_sel = selected_key
Ky = crop_data[selected_key]["Ky"]
b  = crop_data[selected_key]["b"]
EC_prime = crop_data[selected_key]["EC_prime"]
st.sidebar.info(t["crop_info"][lang].format(Ky=Ky, b=b, EC=EC_prime))

st.sidebar.markdown("---")
st.sidebar.header(t["sidebar_header1"][lang])
E0 = st.sidebar.slider(t["E0_label"][lang], min_value=1.0,  max_value=3.0,   step=0.01, key="E0_s")
SI = st.sidebar.slider(t["SI_label"][lang], min_value=100,  max_value=3000,  step=10,   key="SI_s")
Sg = st.sidebar.slider(t["Sg_label"][lang], min_value=1000, max_value=8000,  step=50,   key="Sg_s")

st.sidebar.markdown("---")
st.sidebar.header(t["sidebar_header3"][lang])
I_irr = st.sidebar.slider(t["I_label"][lang],  min_value=0.05, max_value=0.80, step=0.01, key="I_s")
mu    = st.sidebar.slider(t["mu_label"][lang], min_value=0.5,  max_value=2.5,  step=0.05, key="mu_s")

st.sidebar.markdown("---")
st.sidebar.header(t["sidebar_header2"][lang])
hv = st.sidebar.slider(t["hv_label"][lang], min_value=0.1,  max_value=10.0,  step=0.1,  key="hv_s")
Dp = 1.48
dc = hv + Dp
st.sidebar.info(t["dc_info"][lang].format(dc=dc))

with st.sidebar.expander(t["expander_title"][lang], expanded=False):
    st.markdown(t["guide_md"][lang])
    try:
        img = Image.open("Table2_Isotopes.png")
        st.image(img, use_column_width=True)
    except FileNotFoundError:
        pass

# ====== 9. 计算 ======
D_unc, D_opt = calc_D_star(E0, SI, Sg, dc, Ky, b, EC_prime, I_irr, mu)

# ====== 10. Hero ======
st.markdown(f"""
<div class="hero-container">
  <div class="hero-icon">💧</div>
  <div class="hero-content">
    <div class="hero-badge">🔬 {t['badge'][lang]}</div>
    <h1 class="hero-title">{t['title'][lang]}</h1>
    <p class="hero-subtitle">{t['desc'][lang]}</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ====== 11. 主 Tab ======
main_tab1, main_tab2 = st.tabs([t["tab_map"][lang], t["tab_calc"][lang]])

# ─────────────────────────────────────────────
# TAB 1：全球灌区地图（fragment 隔离，点击不触发全页重跑）
# ─────────────────────────────────────────────
@st.fragment
def map_tab_fragment(lang):
    st.markdown(f'<div class="section-header">🌍 {t["map_title"][lang]}</div>', unsafe_allow_html=True)
    st.caption(t["map_hint"][lang])

    # 构建地图数据
    sel_r = st.session_state.sel_region
    lats = [REGIONS[k]["lat"] for k in REGION_KEYS]
    lons = [REGIONS[k]["lon"] for k in REGION_KEYS]
    names = [REGIONS[k]["name_" + lang] for k in REGION_KEYS]
    hover_texts = []
    for k in REGION_KEYS:
        r = REGIONS[k]
        n  = r["name_" + lang]
        d  = r["desc_" + lang]
        e0 = r["E0_default"]
        si = r["SI_default"]
        sg = r["Sg_default"]
        hv_ = r["hv_default"]
        hover_texts.append(
            f"<b>{n}</b><br>"
            f"E₀={e0} m · SI={si} mg/L<br>"
            f"Sg={sg} mg/L · hv={hv_} m<br>"
            f"<i>{d[:55]}…</i>"
        )
    marker_colors = [
        "#a8212e" if k == sel_r else "#b87840" for k in REGION_KEYS
    ]
    marker_sizes = [18 if k == sel_r else 12 for k in REGION_KEYS]

    fig_map = go.Figure(go.Scattergeo(
        lat=lats, lon=lons,
        text=names,
        hoverinfo="none",
        mode="markers",
        marker=dict(
            size=marker_sizes,
            color=marker_colors,
            symbol="circle",
            line=dict(color="#f5f0e8", width=1.5),
            opacity=0.90,
        ),
        customdata=REGION_KEYS,
    ))
    fig_map.update_layout(
        height=400, margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        geo=dict(
            showland=True, landcolor="#e8dcc8",
            showocean=True, oceancolor="#c8d4d8",
            showlakes=True, lakecolor="#b8ccd0",
            showcoastlines=True, coastlinecolor="#a09080",
            coastlinewidth=0.8,
            showframe=False,
            showcountries=True, countrycolor="#c8b898",
            countrywidth=0.5,
            projection_type="natural earth",
            bgcolor="rgba(0,0,0,0)",
        ),
    )

    # 地图点击事件（fragment 内只重跑 fragment）
    map_event = st.plotly_chart(
        fig_map,
        use_container_width=True,
        on_select="rerun",
        selection_mode="points",
        key="map_chart",
        config={"displayModeBar": False, "scrollZoom": False},
    )

    # 处理点击 → 更新选中灌区
    custom_label = t["sel_custom"][lang]
    dd_options = [custom_label] + [REGIONS[k]["name_" + lang] for k in REGION_KEYS]

    if map_event and map_event.selection and map_event.selection.point_indices:
        clicked_idx = map_event.selection.point_indices[0]
        if 0 <= clicked_idx < len(REGION_KEYS):
            st.session_state.sel_region = REGION_KEYS[clicked_idx]
            sel_r = st.session_state.sel_region

    # 下拉框：用 index= 驱动，始终与 sel_r 对齐，无 key 干扰
    current_sel_name = REGIONS[sel_r]["name_" + lang] if sel_r else custom_label
    dd_idx = dd_options.index(current_sel_name) if current_sel_name in dd_options else 0
    chosen = st.selectbox(t["sel_label"][lang], dd_options, index=dd_idx)
    if chosen != custom_label:
        for k in REGION_KEYS:
            if REGIONS[k]["name_" + lang] == chosen:
                if st.session_state.sel_region != k:
                    st.session_state.sel_region = k
                    sel_r = k
    else:
        if st.session_state.sel_region is not None:
            st.session_state.sel_region = None
            sel_r = None

    # 选中灌区信息卡 + 应用按钮
    if sel_r:
        r = REGIONS[sel_r]
        name  = r["name_" + lang]
        desc  = r["desc_" + lang]
        e0r   = r["E0_range"]; e0d = r["E0_default"]
        sir   = r["SI_range"]; sid = r["SI_default"]
        sgr   = r["Sg_range"]; sgd = r["Sg_default"]
        hvr   = r["hv_range"]; hvd = r["hv_default"]
        ir    = r["I_range"];  id_ = r["I_default"]
        mur   = r["mu_range"]; mud = r["mu_default"]
        h  = t["region_params_" + lang]
        cr = crop_display_names[r["crop_default"]][lang]

        if lang == "cn":
            rows = [
                ("潜在蒸发量 E₀ (m/yr)",   f"{e0r[0]}–{e0r[1]}", f"{e0d}"),
                ("灌溉水矿化度 SI (mg/L)",  f"{sir[0]}–{sir[1]}", f"{sid}"),
                ("地下水矿化度 Sg (mg/L)",  f"{sgr[0]}–{sgr[1]}", f"{sgd}"),
                ("灌溉定额 I (m)",           f"{ir[0]}–{ir[1]}",   f"{id_}"),
                ("土壤毛管系数 μ",           f"{mur[0]}–{mur[1]}", f"{mud}"),
                ("植被根系厚度 hv (m)",      f"{hvr[0]}–{hvr[1]}", f"{hvd}"),
                ("典型作物", "—", cr),
            ]
        else:
            rows = [
                ("Potential Evap. E₀ (m/yr)",      f"{e0r[0]}–{e0r[1]}", f"{e0d}"),
                ("Irrigation Salinity SI (mg/L)",   f"{sir[0]}–{sir[1]}", f"{sid}"),
                ("Groundwater Salinity Sg (mg/L)",  f"{sgr[0]}–{sgr[1]}", f"{sgd}"),
                ("Irrigation Quota I (m)",           f"{ir[0]}–{ir[1]}",   f"{id_}"),
                ("Soil Capillary Coeff. μ",          f"{mur[0]}–{mur[1]}", f"{mud}"),
                ("Root Thickness hv (m)",            f"{hvr[0]}–{hvr[1]}", f"{hvd}"),
                ("Typical Crop", "—", cr),
            ]
        TH = 'style="background:#2d2a24;color:#e8d8b8;font-weight:600;padding:7px 12px;text-align:left;font-size:0.75rem;letter-spacing:0.3px;"'
        TD = 'style="padding:7px 12px;color:#3d3530;border-bottom:1px solid #ede5d8;"'
        TD_ALT = 'style="padding:7px 12px;color:#3d3530;border-bottom:1px solid #ede5d8;background:#f5f0e8;"'
        TD_B = 'style="padding:7px 12px;color:#3d3530;border-bottom:1px solid #ede5d8;"'
        rows_html = "".join(
            f"<tr><td {TD if i%2==0 else TD_ALT}>{p}</td>"
            f"<td {TD if i%2==0 else TD_ALT}>{rng}</td>"
            f"<td {TD_B if i%2==0 else TD_ALT}><b style='color:#b87840'>{dv}</b></td></tr>"
            for i, (p, rng, dv) in enumerate(rows)
        )
        st.markdown(f"""
        <div class="region-card">
          <div class="region-name"><span class="region-dot"></span>{name}</div>
          <div class="region-desc">{desc}</div>
          <table class="param-table" style="width:100%;border-collapse:collapse;font-size:0.81rem;border-radius:8px;overflow:hidden;">
            <tr><th {TH}>{h[0]}</th><th {TH}>{h[1]}</th><th {TH}>{h[2]}</th></tr>
            {rows_html}
          </table>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")
        if st.button(t["apply_btn"][lang], type="primary", use_container_width=True):
            st.session_state._do_apply = True
            st.rerun()  # 全页重跑以更新侧边栏滑块

        if not st.session_state._do_apply and st.session_state.get("_just_applied"):
            st.success(t["apply_ok"][lang])
            st.session_state._just_applied = False

with main_tab1:
    map_tab_fragment(lang)


# ─────────────────────────────────────────────
# TAB 2：计算结果与敏感性分析
# ─────────────────────────────────────────────
with main_tab2:

    # 当前灌区提示
    if st.session_state.sel_region:
        rn = REGIONS[st.session_state.sel_region]["name_" + lang]
        st.info(f"{'当前参考灌区' if lang == 'cn' else 'Current Reference Region'}: **{rn}**")

    # ---- 结果指标卡 ----
    st.markdown(f'<div class="section-header">📊 {t["res_header"][lang]}</div>', unsafe_allow_html=True)

    if D_opt is not None:
        triggered = (abs(D_opt - dc) < 1e-9)
        card1_cls = "metric-card danger" if triggered else "metric-card"
        status_label = t["status_trigger"][lang] if triggered else t["status_unconstrained"][lang]
        st.markdown(f"""
        <div class="metric-grid">
          <div class="{card1_cls}">
            <span class="metric-icon">🎯</span>
            <div class="metric-label">{t['metric_D_opt'][lang]}</div>
            <div class="metric-value">{D_opt:.2f} m</div>
            <span class="metric-status">{status_label}</span>
          </div>
          <div class="metric-card success">
            <span class="metric-icon">📐</span>
            <div class="metric-label">{t['metric_D_unc'][lang]}</div>
            <div class="metric-value">{D_unc:.2f} m</div>
          </div>
          <div class="metric-card warning">
            <span class="metric-icon">🌿</span>
            <div class="metric-label">{t['metric_dc'][lang]}</div>
            <div class="metric-value">{dc:.2f} m</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # 横向对比图
        BLUE = "#5c8a96"; GREEN = "#5c8a96"; ORANGE = "#b87840"
        fig_bar = go.Figure()
        bar_items = [
            ("dc",    dc,    ORANGE, f"dc = {dc:.2f} m"),
            ("D_unc", D_unc, GREEN,  ("无约束" if lang=="cn" else "Unc.") + f" = {D_unc:.2f} m"),
            ("D*",    D_opt, BLUE,   f"D* = {D_opt:.2f} m"),
        ]
        for _, val, col, lbl in bar_items:
            fig_bar.add_trace(go.Bar(
                x=[val], y=[lbl], orientation="h",
                marker_color=col, marker_line_width=0,
                text=[f"{val:.2f} m"], textposition="inside",
                textfont=dict(color="white", size=13),
                hovertemplate=f"<b>{lbl}</b><extra></extra>",
                showlegend=False,
            ))
        x_ax = "深度 (m)" if lang == "cn" else "Depth (m)"
        fig_bar.update_layout(
            height=155, margin=dict(l=10, r=20, t=8, b=8),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#faf7f0", barmode="overlay",
            xaxis=dict(title=x_ax, showgrid=False, zeroline=False, tickfont=dict(size=11)),
            yaxis=dict(showgrid=False, tickfont=dict(size=11)),
        )
        st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

    else:
        st.markdown(f'<div class="error-card">{t["err_msg"][lang]}</div>', unsafe_allow_html=True)

    # ---- 公式 ----
    with st.expander(f"📐 {t['formula_header'][lang]} — {t['formula_caption'][lang]}", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.latex(r"D^* = \min\!\left(d_c,\; D_{\text{unc}}\right)")
            st.latex(r"D_{\text{unc}} = -\frac{1}{\mu}\ln\!\left(-\frac{N/M + \Psi/\Phi}{2\lambda E_0}\right)")
        with c2:
            st.latex(r"M = -\frac{\alpha\theta b S_g}{100\,K_y\rho_s f\delta I}")
            st.latex(r"N = 1 - \frac{b}{100K_y}\!\left(\frac{\alpha\theta S_I(1-(1-f)\delta)}{\rho_s f\delta}+\beta-EC'\right)")

    # ---- 敏感性分析 ----
    st.markdown("---")
    st.markdown(f'<div class="section-header">📈 {t["sens_header"][lang]}</div>', unsafe_allow_html=True)

    LCOL = "#5c8a96"; GCOL = "#a0c0c8"; RCOL = "#a8212e"

    def sens_chart(x_arr, y_opt, y_unc, x_label, dc_val, cur_x, cur_y):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_arr, y=y_unc, name="D_unc" if lang=="en" else "无约束埋深",
            line=dict(color=GCOL, width=1.5, dash="dot"),
            hovertemplate=f"{x_label}: %{{x:.1f}}<br>D_unc=%{{y:.2f}} m<extra></extra>",
        ))
        fig.add_hline(y=dc_val, line=dict(color=RCOL, width=1.5, dash="dash"),
                      annotation_text=f"dc={dc_val:.2f} m",
                      annotation_font=dict(color=RCOL, size=10))
        fig.add_trace(go.Scatter(
            x=x_arr, y=y_opt, name="D*",
            line=dict(color=LCOL, width=2.5),
            fill="tozeroy", fillcolor="rgba(92,138,150,0.08)",
            hovertemplate=f"{x_label}: %{{x:.1f}}<br>D*=%{{y:.2f}} m<extra></extra>",
        ))
        if cur_y is not None:
            fig.add_trace(go.Scatter(
                x=[cur_x], y=[cur_y], mode="markers", showlegend=False,
                marker=dict(color=LCOL, size=10, line=dict(color="white", width=2)),
                hovertemplate=f"{x_label}=%{{x:.2f}}, D*=%{{y:.2f}} m<extra></extra>",
            ))
        fig.update_layout(
            height=320, margin=dict(l=10, r=20, t=18, b=8),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#faf7f0",
            legend=dict(orientation="h", y=1.05, x=1, xanchor="right", font=dict(size=11)),
            xaxis=dict(title=x_label, showgrid=True, gridcolor="#ede5d8", tickfont=dict(size=11)),
            yaxis=dict(title="D* (m)", showgrid=True, gridcolor="#ede5d8", tickfont=dict(size=11)),
            hovermode="x unified",
        )
        return fig

    ts1, ts2, ts3 = st.tabs([t["tab_sg"][lang], t["tab_e0"][lang], t["tab_si"][lang]])

    with ts1:
        sg_arr = np.linspace(1000, 8000, 80)
        fig = sens_chart(
            sg_arr,
            [calc_D_star(E0, SI, sg, dc, Ky, b, EC_prime, I_irr, mu)[1] for sg in sg_arr],
            [calc_D_star(E0, SI, sg, dc, Ky, b, EC_prime, I_irr, mu)[0] for sg in sg_arr],
            "Sg (mg/L)", dc, Sg, D_opt,
        )
        st.plotly_chart(fig, use_container_width=True)

    with ts2:
        e0_arr = np.linspace(1.0, 3.0, 80)
        fig = sens_chart(
            e0_arr,
            [calc_D_star(e, SI, Sg, dc, Ky, b, EC_prime, I_irr, mu)[1] for e in e0_arr],
            [calc_D_star(e, SI, Sg, dc, Ky, b, EC_prime, I_irr, mu)[0] for e in e0_arr],
            "E₀ (m)", dc, E0, D_opt,
        )
        st.plotly_chart(fig, use_container_width=True)

    with ts3:
        si_arr = np.linspace(100, 3000, 80)
        fig = sens_chart(
            si_arr,
            [calc_D_star(E0, si, Sg, dc, Ky, b, EC_prime, I_irr, mu)[1] for si in si_arr],
            [calc_D_star(E0, si, Sg, dc, Ky, b, EC_prime, I_irr, mu)[0] for si in si_arr],
            "SI (mg/L)", dc, SI, D_opt,
        )
        st.plotly_chart(fig, use_container_width=True)
