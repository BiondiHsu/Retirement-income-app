import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

font_path = "fonts/NotoSansCJK-Regular.otf"
font_prop = fm.FontProperties(fname=font_path)

plt.xlabel("決策準確度 (%)", fontproperties=font_prop)
plt.ylabel("EVII 成長值", fontproperties=font_prop)

st.set_page_config(layout="wide")
st.title("🎯 EVPI / EVII 決策分析互動遊戲（完整教學版）")

# =====================================================
# 一、產品資訊
# =====================================================
st.header("① 產品資訊")

colP = st.columns(3)
price = colP[0].number_input("售價 Price", value=100.0, step=1.0)
var_cost = colP[1].number_input("單位變動成本 Variable Cost", value=60.0, step=1.0)
fix_cost = colP[2].number_input("固定成本 Fixed Cost", value=0.0, step=500.0)

if price <= var_cost:
    st.warning("⚠️ 售價 ≤ 變動成本，所有決策利潤將無意義")

# =====================================================
# 二、行動：訂購量 A1–A3
# =====================================================
st.header("② 行動設定：訂購量（A1–A3）")

colA = st.columns(3)
order_qty = np.array([
    colA[0].number_input("A1（保守）", value=200.0, step=50.0),
    colA[1].number_input("A2（中等）", value=400.0, step=50.0),
    colA[2].number_input("A3（積極）", value=600.0, step=50.0),
])

# =====================================================
# 三、狀態：需求量 X1–X3
# =====================================================
st.header("③ 狀態設定：需求量（X1–X3）")

colX = st.columns(3)
demand = np.array([
    colX[0].number_input("X1 低需求", value=250.0, step=50.0),
    colX[1].number_input("X2 中需求", value=450.0, step=50.0),
    colX[2].number_input("X3 高需求", value=650.0, step=50.0),
])

# =====================================================
# 四、4×3 機率矩陣
# =====================================================
st.header("④ 機率設定（4×3）")

states = ["X1", "X2", "X3"]
signals = ["悲觀", "中等", "樂觀"]

# ---- 事前機率 ----
st.subheader("📌 事前機率 P(X)")
cols_px = st.columns(3)
p_x = np.array([
    cols_px[i].number_input(f"P({states[i]})", value=1/3, step=0.05)
    for i in range(3)
])

# ---- 條件機率 ----
st.subheader("📌 條件機率 P(Y | X)")
p_y_given_x = np.zeros((3, 3))

for i, x in enumerate(states):
    cols = st.columns(3)
    for j, y in enumerate(signals):
        p_y_given_x[i, j] = cols[j].number_input(
            f"P({y} | {x})", value=1/3, step=0.05
        )

# =====================================================
# 五、Payoff 矩陣
# =====================================================
payoff = np.zeros((3, 3))

for i in range(3):
    for j in range(3):
        sold = min(order_qty[i], demand[j])
        payoff[i, j] = sold * price - order_qty[i] * var_cost - fix_cost

# =====================================================
# 六、EVPI / EVII 計算
# =====================================================
emv = payoff @ p_x
best_emv = emv.max()

max_per_state = payoff.max(axis=0)
ev_wpi = (max_per_state * p_x).sum()
evpi = ev_wpi - best_emv

# ----- EVII -----
p_y = (p_x.reshape(-1, 1) * p_y_given_x).sum(axis=0)
p_x_given_y = (p_y_given_x * p_x.reshape(-1, 1)) / p_y.reshape(1, -1)

ev_y = payoff @ p_x_given_y
best_ev_y = ev_y.max(axis=0)

ev_wii = (best_ev_y * p_y).sum()
evii = ev_wii - best_emv

# =====================================================
# 七、數值結果
# =====================================================
st.header("⑤ 計算結果")

colR = st.columns(2)
colR[0].metric("EVPI（完美資訊價值）", f"{evpi:,.2f}")
colR[1].metric("EVII（不完美資訊價值）", f"{evii:,.2f}")

# =====================================================
# 八、資訊準確度 → EVII 曲線（重頭戲）
# =====================================================
st.header("⑥ 資訊準確度 → EVII 成長曲線")

st.markdown("""
<div style="margin-top:20px;">

<!-- 燈泡保持原色 -->
<div style="font-size:80px;">💡

<!-- 決策洞見：更大、更粗、更綠 -->
<div style="font-size:48px; font-weight:900; color:#2E7D32; margin-top:-10px;">
    Decision Insight
</div>

<!-- 英文深藍（可放可不放） -->
<div style="font-size:30px; color:#0D47A1; margin-top:10px;">
    Information itself does not create value.<br>
    Only information that changes decisions is valuable.
</div>

</div>
""", unsafe_allow_html=True)

lambdas = np.linspace(0, 1, 21)
evii_curve = []

for lam in lambdas:
    # 線性插值：從「沒資訊」走向「目前條件機率」
    p_y_given_x_lam = lam * p_y_given_x + (1 - lam) * np.ones((3, 3)) / 3

    p_y_lam = (p_x.reshape(-1, 1) * p_y_given_x_lam).sum(axis=0)
    p_x_given_y_lam = (p_y_given_x_lam * p_x.reshape(-1, 1)) / p_y_lam.reshape(1, -1)

    ev_y_lam = payoff @ p_x_given_y_lam
    best_ev_y_lam = ev_y_lam.max(axis=0)

    ev_wii_lam = (best_ev_y_lam * p_y_lam).sum()
    evii_curve.append(ev_wii_lam - best_emv)

# Plot
# ---- 中文字型設定（永不再變麻將白板） ----
font_path = "fonts/NotoSansCJK-Regular.otf"
font_prop = fm.FontProperties(fname=font_path)

# ---- Plot with Chinese font ----
fig, ax = plt.subplots()

ax.plot(lambdas, evii_curve, marker="o")

# 套用中文字型
ax.set_xlabel("資訊準確度 λ", fontproperties=font_prop)
ax.set_ylabel("EVII（資訊價值）", fontproperties=font_prop)
ax.set_title("資訊越準 → 不完美資訊價值（EVII）成長", fontproperties=font_prop)

# 讓刻度也用中文字型（很重要！）
for label in ax.get_xticklabels():
    label.set_fontproperties(font_prop)
for label in ax.get_yticklabels():
    label.set_fontproperties(font_prop)

ax.grid(True)

st.pyplot(fig)


st.markdown("""
<!-- 1️⃣ 燈泡＋標題同一行（使用 flex） -->
<div style="display:flex; align-items:center; gap:15px;">

<!-- 綠色燈泡 -->
<div style="font-size:100px;">💡</div>

<!-- 決策洞見標題（大／粗／綠） -->
<div style="font-size:42px; font-weight:900; color:#2E7D32;">
    決策洞見
</div>

<!-- 中文深藍 -->
<div style="font-size:28px; color:#0D47A1; margin-top:20px;">
    資訊本身不創造價值，<br>
    能改變決策的資訊，才有價值。
</div>

<!-- 英文深藍 + 與後續表格留兩行 -->
<div style="font-size:22px; color:#0D47A1; margin-top:10px; margin-bottom:40px;">
     · Decision Insight · 
    'Information itself does not create value.<br>
    Only information that changes decisions is valuable.'
</div>

""", unsafe_allow_html=True)

# =====================================================
# 九、教學用展開表
# =====================================================
with st.expander("📊 教學用表格（Payoff / 機率）"):
    st.write("Payoff 矩陣")
    st.table(pd.DataFrame(payoff, index=["A1", "A2", "A3"], columns=states))

    st.write("P(X)")
    st.table(pd.DataFrame([p_x], columns=states))

    st.write("P(Y | X)")
    st.table(pd.DataFrame(p_y_given_x, index=states, columns=signals))
