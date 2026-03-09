import streamlit as st
import pandas as pd
import numpy as np
import math
from PIL import Image  # 导入用于处理图片的库

# ====== 1. 核心计算函数 ======
def calc_D_star(E0, SI, Sg, dc):
    # 基础物理与农艺参数 (河套灌区基准)
    Ky = 1.0; b = 7.1; theta = 0.23; rho_s = 1475.0
    I = 0.3216; delta = 0.29; f = 0.5; EC_prime = 6.0
    alpha = 2.156; beta = 0.616; lam = 1.7256; mu = 1.3603
    eps = 0.5; omega = 0.405; q = 0.1; P = 0.0

    # 计算盐分胁迫参数 M 和 N
    M = - (alpha * theta * b * Sg) / (100.0 * Ky * rho_s * f * delta * I)
    term_SI = SI * (1.0 - (1.0 - f) * delta) / (f * delta)
    N = 1.0 - (b / (Ky * 100.0)) * ((alpha * theta / rho_s) * term_SI + beta - EC_prime)

    # 计算水分胁迫参数 Phi 和 Psi
    ET_gm = lam * E0
    denom = omega * I * (1.0 - delta) + P + eps * ET_gm
    Phi = eps / denom
    Psi = (omega * I * (1.0 - delta) / (1.0 - q) + P) / denom

    # 计算最优埋深
    inside_ln = (-1.0 / (2.0 * lam * E0)) * (N / M + Psi / Phi)

    if inside_ln <= 0:
        return None, None
        
    D_unc = (-1.0 / mu) * math.log(inside_ln)
    D_opt = min(dc, D_unc) # 应用生态红线约束
    return D_unc, D_opt

# ====== 2. 页面与侧边栏设置 ======
st.set_page_config(page_title="最优地下水埋深计算器", layout="wide")
st.title("💧 浅埋干旱灌区最优地下水埋深可视化系统")
st.markdown("基于水文经济解析解公式构建，支持动态参数模拟与全球典型植被生态约束参考。")

st.sidebar.header("🎛️ 驱动参数输入区")
E0 = st.sidebar.slider("潜在蒸发量 E0 (m)", min_value=1.00, max_value=2.50, value=1.237, step=0.01)
SI = st.sidebar.slider("灌溉水矿化度 SI (mg/L)", min_value=100, max_value=3000, value=586, step=10)
Sg = st.sidebar.slider("地下水矿化度 Sg (mg/L)", min_value=1000, max_value=8000, value=4000, step=50)

# ====== 生态约束参数输入 ======
st.sidebar.markdown("---")
st.sidebar.header("🌿 生态约束设置")
hv = st.sidebar.slider("植被根系厚度 hv (m)", min_value=0.1, max_value=10.0, value=1.8, step=0.1)

# 计算生态临界深度 dc
Dp = 1.48 
dc = hv + Dp
st.sidebar.info(f"**当前计算生态红线 (dc):** `{dc:.2f} m`")

# ====== 📚 全球典型植被地下水利用参考表 ======
with st.sidebar.expander("📖 全球地下水依赖型植被参考指南", expanded=False):
    st.markdown("""
    以下数据提取自稳定同位素文献汇总表，展示了不同地区植被对地下水的利用情况：
    
    | 典型植被 (物种名称) | 研究地点 (Location) | 地下水深度 (Depth, m) | 
    | :--- | :--- | :--- |
    | **多利柽柳 (*Tamarix ramosissima*)** | 美国 亚利桑那州 | **1.08 ~ 3.89** |
    | **骆驼刺属 (*Alhagi*)** | 中亚等干旱区 | **2.0 ~ 2.5** |
    | **牧豆树 (*Prosopis velutina*)** | 美国 亚利桑那州 | **1.8, 2.6, 4.3** |
    | **赤桉 (*Eucalyptus camaldulensis*)** | 澳大利亚南部 | **1.3 ~ 4.0** |
    | **白杨 (*Populus fremontii*)** | 美国 加州/犹他州 | **0.1 ~ 3.89** |
    | **无叶柽柳 (*Tamarix aphylla*)** | 南亚/中东 | **2.5 ~ 3.0** |
    """)
    
    # 🌟 在这里插入文献原图 🌟
    # 请确保图片文件放在你运行代码的相同文件夹（E:\ResearchCode\Optimal_groundwater_depth\py\）下
    # 如果图片名字不一样，请修改下面括号里的字符串
    try:
        image_path = "Table2_Isotopes.png"  # <-- 请把这行改成你实际保存的图片文件名
        img = Image.open(image_path)
        st.image(img, caption="Table 2: Selection of Studies in Which Groundwater Has Been Identified as a Source of Water for Vegetation Using Stable Isotopes", use_column_width=True)
    except FileNotFoundError:
        st.warning(f"无法加载图片: 找不到 '{image_path}'。请确保图片已放在代码同级目录下。")

    st.markdown("""
    > **管理启示**：根据同位素研究显示，柽柳属和牧豆树等典型潜水植被在埋深超过 4m 时仍能维持较高的地下水利用率。在设定 **$h_v$** 时，应考虑当地主要保护物种的生物学特性。
    """)

# ====== 3. 实时计算与核心结果显示 ======
D_unc, D_opt = calc_D_star(E0, SI, Sg, dc)

st.subheader("📊 实时计算结果")
if D_opt is not None:
    col1, col2, col3 = st.columns(3)
    status_text = "触发生态红线 (dc)" if D_opt == dc else "无约束经济最优"
    col1.metric("🌟 最终最优地下水埋深 D*", f"{D_opt:.2f} m", status_text, delta_color="off")
    col2.metric("理论无约束埋深", f"{D_unc:.2f} m")
    col3.metric("当前生态红线界限 (dc)", f"{dc:.2f} m")
else:
    st.error("⚠️ 当前参数组合下最优埋深趋于无限深（无解析解）。")

# ====== 4. 敏感性分析图表 ======
st.markdown("---")
st.subheader("📈 敏感性分析")
col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    st.markdown("**1. 地下水矿化度 ($S_g$) 对 $D^*$ 的影响**")
    sg_arr = np.linspace(1000, 8000, 100)
    d_sg = [calc_D_star(E0, SI, sg, dc)[1] for sg in sg_arr]
    df_sg = pd.DataFrame({"Sg (mg/L)": sg_arr, "D* (m)": d_sg}).set_index("Sg (mg/L)")
    st.line_chart(df_sg)

with col_chart2:
    st.markdown("**2. 潜在蒸发量 ($E_0$) 对 $D^*$ 的影响**")
    e0_arr = np.linspace(1.0, 2.5, 100)
    d_e0 = [calc_D_star(e, SI, Sg, dc)[1] for e in e0_arr]
    df_e0 = pd.DataFrame({"E0 (m)": e0_arr, "D* (m)": d_e0}).set_index("E0 (m)")
    st.line_chart(df_e0)