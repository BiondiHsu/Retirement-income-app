import streamlit as st
import pandas as pd
import numpy as np
import random
import io
from io import BytesIO
import re
import qrcode
from PIL import Image
import base64

# =========================
# 你的 APP 專屬連結
# =========================
APP_URL = "https://your-pps-app.streamlit.app"   # ← 請改成你的網址

# =========================
# 產生 QR Code as PNG bytes
# =========================
def generate_qr(url: str) -> bytes:
    qr = qrcode.QRCode(
        version=2,
        box_size=10,
        border=2,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
    )
    qr.add_data(url)
    qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

qr_png = generate_qr(APP_URL)

# =========================
# 美式卡片風格 CSS
# =========================
page_style = """
<style>
.hero {
    text-align: center;
    padding: 40px 10px 20px 10px;
}
.card {
    background: #ffffff;
    padding: 25px;
    border-radius: 18px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    width: 330px;
    margin: auto;
    text-align: center;
}
.qr-img {
    width: 230px;
    height: 230px;
    margin-top: 10px;
}
.btn-container {
    margin-top: 18px;
    display: flex;
    justify-content: center;
    gap: 10px;
}
.big-btn {
    background-color: #0047AB;
    color: white;
    border-radius: 10px;
    padding: 10px 22px;
    text-decoration: none;
    font-size: 15px;
}
.big-btn:hover {
    background-color: #003a89;
}
</style>
"""
st.markdown(page_style, unsafe_allow_html=True)

# =========================
# 頁面內容
# =========================
st.markdown(
    "<div class='hero'>"
    "<h1 style='color:#0047AB; font-size:36px;'>Green Audit — PPS Sampling</h1>"
    "<p style='font-size:18px; color:#444;'>Scan the QR Code below to open the mobile web app</p>"
    "</div>",
    unsafe_allow_html=True
)

# ---- QR Card ----
st.markdown("<div class='card'>", unsafe_allow_html=True)

st.image(qr_png, use_column_width=False, width=230)

st.markdown(
    f"<p style='margin-top:10px; font-size:16px;'><b>{APP_URL}</b></p>",
    unsafe_allow_html=True
)

# ---- Buttons ----
col1, col2 = st.columns(2)
with col1:
    if st.button("📋 Copy URL"):
        st.write("已複製網址！請貼到瀏覽器或分享給學生。")

with col2:
    st.markdown(
        f"<a href='{APP_URL}' target='_blank' class='big-btn'>Open App</a>",
        unsafe_allow_html=True
    )

st.markdown("</div>", unsafe_allow_html=True)  # end card

# =========================
# 結尾版權區（可省略）
# =========================
st.markdown(
    "<p style='margin-top:35px; text-align:center; color:#777;'>"
    "Designed for PPS Testing · Streamlit Edition</p>",
    unsafe_allow_html=True
)

st.set_page_config(page_title="PPS Sampling & Testing", layout="wide")
st.title("📊 PPS 系統抽樣與查核平台")
# 1204 PPS Testing.py
# =====================================================
# 🟩 第一段：PPS Sampling（📌 PPS 抽樣使用工作表：PPS1 格式)
# =====================================================
# st.markdown("<h2>① PPS Sampling（PPS1, 📌 PPS 抽樣使用工作表：PPS1 格式）</h2>", unsafe_allow_html=True)
st.set_page_config(page_title="PPS Testing", layout="wide")
st.title("① PPS Sampling（📌 PPS 抽樣使用工作表：PPS1 格式）")

uploaded_pps = st.file_uploader(
    "上傳 PPS Excel（含 PPS1）",
    type=["xlsx"],
    key="pps_sampling"
)

col1, col2 = st.columns(2)
with col1:
    n = st.number_input("樣本量 n", min_value=1, value=4, step=1)
with col2:
    start_point = st.number_input("起始點（0 = 隨機產生）", value=0.0)

if st.button("▶ 分析抽樣（PPS Sampling）"):

    if uploaded_pps is None:
        st.error("❌ 請先上傳 Excel 檔案")
        st.stop()

    df_raw = pd.read_excel(uploaded_pps, sheet_name="PPS1")

    # ---------- 找金額欄（語意優先） ----------
    keyword_priority = ["amount", "金額", "book", "record", "value"]
    amount_col = None
    for c in df_raw.columns:
        cname = str(c).lower()
        if any(k in cname for k in keyword_priority):
            amount_col = c
            break

    # fallback：數值最多的欄
    if amount_col is None:
        numeric_info = []
        for c in df_raw.columns:
            s = pd.to_numeric(df_raw[c], errors="coerce")
            numeric_info.append((s.notna().sum(), c))
        numeric_info.sort(reverse=True)
        amount_col = numeric_info[0][1]

    df_samp = df_raw[[amount_col]].copy()
    df_samp[amount_col] = pd.to_numeric(df_samp[amount_col], errors="coerce")
    df_samp = df_samp.dropna().reset_index(drop=True)

    total = df_samp[amount_col].sum()
    interval = total / n

    start = start_point if 0 < start_point <= interval else random.uniform(0, interval)

    df_samp["Cumulative"] = df_samp[amount_col].cumsum()

    max_cum = df_samp["Cumulative"].max()
    sampling_points = [start + i * interval for i in range(n)]

    result = []
    for p in sampling_points:
        cand = df_samp[df_samp["Cumulative"] >= p]
        if cand.empty:
            row = df_samp.iloc[-1]
            idx = df_samp.index[-1] + 1
        else:
            row = cand.iloc[0]
            idx = cand.index[0] + 1

        result.append({
            "Sampling Point": round(p, 2),
            "Selected Index": idx
        })

    result_df = pd.DataFrame(result)
    st.session_state["pps_sampling_result"] = result_df

    st.success("✅ PPS 抽樣完成（PPS1）")
    st.dataframe(result_df)

# ---------- 匯出按鈕：輸出為 Excel ----------
if st.button("📤 匯出 PPS 抽樣結果（Excel）"):

    if "pps_sampling_result" not in st.session_state:
        st.warning("⚠️ 尚未執行抽樣，請先按『分析抽樣』。")
    else:
        out_df = st.session_state["pps_sampling_result"]

        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            out_df.to_excel(writer, index=False, sheet_name="PPS_Sampling")
        buffer.seek(0)

        st.download_button(
            label="⬇ 下載 PPS_Sampling_Result.xlsx",
            data=buffer,
            file_name="PPS_Sampling_Result.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

# ============================================================
#  🟩 第二段：PPS Testing（PPS2~~） 工具函式（直接沿用你原始邏輯）
# ============================================================
# st.markdown("<h2>② PPS Testing（PPS2–PPS7）</h2>", unsafe_allow_html=True)
import re

def calc_incremental_allowance(df, record_col, risk_df, risk_col, sampling_interval):
    """
    只對 Recorded < sampling_interval 且有 PM 的樣本做 IA
    完全對應你原本的 calc_incremental_allowance。
    """
    rank_mask = (df[record_col] < sampling_interval) & df["PM"].notna()

    rankings = (
        df.loc[rank_mask, "PM"]
        .rank(ascending=False, method="first")
        .astype(int)
    )

    df.loc[rank_mask, "Ranking"] = rankings
    df["IA"] = 0.0

    max_rank_in_table = risk_df.index.max()

    for idx, r in rankings.items():
        r_use = int(min(r, max_rank_in_table))
        prev_r = max(r_use - 1, 0)

        CF_curr = risk_df.at[r_use, risk_col]
        CF_prev = risk_df.at[prev_r, risk_col]
        delta_CF = CF_curr - CF_prev

        IA = abs(df.at[idx, "PM"]) * delta_CF - df.at[idx, "PM"]
        df.at[idx, "IA"] = IA

    return df


def read_pps_testing_sheet(xls_file, sheet_name):
    """
    對應你原程式：
    - 先讀 raw（header=None）
    - 找到包含 Account / Recorded / Audited 的那一列當表頭
    - 刪掉空欄、空列
    - 只保留 Total / 合計 / 總計 / 小計 之前的樣本列
    """
    raw = pd.read_excel(xls_file, sheet_name=sheet_name, header=None)

    # 找 header 列
    header_row = None
    for i, row in raw.iterrows():
        s = "".join(row.astype(str).tolist()).lower()
        if "account" in s and "record" in s and "audit" in s:
            header_row = i
            break

    if header_row is None:
        raise ValueError("❌ 找不到包含 Account / Recorded / Audited 的欄位列。")

    df = pd.read_excel(xls_file, sheet_name=sheet_name, header=header_row)
    df = df.dropna(axis=1, how="all")
    df = df[df.notna().any(axis=1)].reset_index(drop=True)

    # 自動偵測欄位名稱
    def find_col(df, keywords):
        for col in df.columns:
            name = str(col).lower().replace(" ", "")
            if any(k in name for k in keywords):
                return col
        return None

    col_acc = find_col(df, ["account"])
    col_record = find_col(df, ["record"])
    col_audit = find_col(df, ["audit"])

    if col_acc is None or col_record is None or col_audit is None:
        raise ValueError(
            f"❌ 無法自動辨識欄位，取得結果："
            f"Account={col_acc}, Recorded={col_record}, Audited={col_audit}"
        )

    # 找 Total / 合計 列，只取之前的樣本
    total_index = None
    for i, v in enumerate(df[col_acc].astype(str)):
        if re.search(r"(?i)total|合計|總計|小計", v):
            total_index = i
            break

    if total_index is None:
        raise ValueError("❌ 無法找到 Total / 合計 列。")

    df = df.loc[:total_index - 1].copy()

    # 數字欄清理
    for c in [col_record, col_audit]:
        df[c] = pd.to_numeric(
            df[c].astype(str).str.replace(r"[^\d\.-]", "", regex=True),
            errors="coerce",
        )

    mask_data = df[[col_record, col_audit]].notna().any(axis=1)

    return df, mask_data, col_record, col_audit


def load_risk_factor_table(xls_file, risk_num):
    """
    對應你原程式的 Risk Factor 讀取邏輯：
    - sheet_name='Risk Factor'
    - 第一欄當 ranking index
    - 用「欄名去掉 % 轉成數字」比對 5/10/15/20
    """
    risk_df = pd.read_excel(xls_file, sheet_name="Risk Factor", header=0)
    first_col = risk_df.columns[0]
    risk_df = risk_df.rename(columns={first_col: "ranking"})
    risk_df = risk_df.set_index("ranking")

    risk_col = None
    for col in risk_df.columns:
        try:
            col_num = int(str(col).strip().replace("％", "").replace("%", ""))
            if col_num == risk_num:
                risk_col = col
                break
        except ValueError:
            continue

    if risk_col is None:
        raise ValueError(
            f"❌ Risk Factor 找不到對應誤受險 {risk_num}% 欄位，"
            f"目前欄名：{list(risk_df.columns)}"
        )

    return risk_df, risk_col

# ======================
# Streamlit UI 開始
# ======================
st.set_page_config(page_title="PPS Testing", layout="wide")
st.title("② PPS Testing（ ✅ PPS 測試使用工作表：PPS2+ 格式）")

# -------- 檔案與工作表選擇 --------
uploaded_file = st.file_uploader("上傳 112-1 Test.xlsx", type=["xlsx"])

pps_sheet = st.selectbox(
    "選擇 PPS 測試工作表",
    ["PPS2", "PPS3", "PPS4", "PPS5", "PPS6", "PPS7", "PPS8", "PPS9", "PPS10"],
)

# -------- 參數輸入（含樣本量）--------
c1, c2, c3, c4 = st.columns(4)
population_value = c1.number_input("Book Value（帳面價值 BV）", min_value=0.0, value=0.0)
TM = c2.number_input("Tolerable Misstatement（允收錯誤 TM）", min_value=0.0, value=0.0)
sample_size = c3.number_input("Sample Size（樣本量 n）", min_value=1, value=4, step=1)
risk_num = c4.selectbox("Acceptable Risk (%)（誤受險）", [5, 10, 15, 20])

st.markdown("---")

# ======================
# ① 分析按鈕：執行 PPS Testing
# ======================
def fmt(x):
    return float(f"{x:.2f}")

if st.button("▶ 執行 PPS Testing"):

    if uploaded_file is None:
        st.error("❌ 請先上傳 112-1 Test.xlsx")
        st.stop()

    if population_value <= 0:
        st.error("❌ Book Value 必須大於 0")
        st.stop()

    # 7) Sampling Interval
    sampling_interval = population_value / sample_size

    try:
        # 1) 讀取 PPSX 工作表，偵測欄位與樣本區
        df, mask_data, col_record, col_audit = read_pps_testing_sheet(
            uploaded_file, pps_sheet
        )

        # 9) 計算 FM / t% / PM（完全照你原本的寫法）
        df.loc[mask_data, "FM"] = (
            df.loc[mask_data, col_record] - df.loc[mask_data, col_audit]
        )

        df.loc[mask_data, "t%"] = np.where(
            df.loc[mask_data, col_record] != 0,
            df.loc[mask_data, "FM"] / df.loc[mask_data, col_record],
            0,
        )

        df.loc[mask_data, "PM"] = np.where(
            df.loc[mask_data, "FM"] < 0,
            df.loc[mask_data, "FM"],
            np.where(
                df.loc[mask_data, col_record] < sampling_interval,
                df.loc[mask_data, "t%"] * sampling_interval,
                df.loc[mask_data, "FM"],
            ),
        )

        # Risk Factor
        risk_df, risk_col = load_risk_factor_table(uploaded_file, risk_num)

        # IA（增額風險）
        df = calc_incremental_allowance(
            df, col_record, risk_df, risk_col, sampling_interval
        )

        # 12) 匯總：PM total, IA total, BP, ASR, UML
        PM_total = df.loc[mask_data, "PM"].sum()
        IA_total = df.loc[mask_data, "IA"].sum()
        BP = sampling_interval * risk_df.at[0, risk_col]
        ASR = BP + IA_total
        UML = PM_total + ASR

        decision = "接受 Accept ✅" if UML <= TM else "拒絕 Reject ❌"

        # 存進 session_state 方便輸出用
        st.session_state["pps_testing_detail"] = df.loc[mask_data].copy()
        st.session_state["pps_testing_summary"] = {
            "Book Value": fmt(population_value),
            "Sample Size": sample_size,
            "Sampling Interval": fmt(sampling_interval),
            "Tolerable Misstatement (TM)": fmt(TM),
            "Acceptable Risk": f"{risk_num}%",
            "PM Total": fmt(PM_total),
            "IA Total": fmt(IA_total),
            "Basic Precision (BP)": fmt(BP),
            "Audit Risk Premium (ASR)": fmt(ASR),
            "Upper Misstatement Limit (UML)": fmt(UML),
            "Decision": decision,
        }

        # ========= 畫面輸出 =========
        st.success("✅ PPS Testing 計算完成")

        cA, cB, cC = st.columns(3)
        cA.metric("UML", f"{UML:,.2f}")
        cB.metric("TM", f"{TM:,.2f}")
        cC.metric("審查結論", decision)

        st.subheader("📄 Summary")
        st.json(st.session_state["pps_testing_summary"])

        st.subheader("📑 Detail（FM / t% / PM / IA）")
        st.dataframe(st.session_state["pps_testing_detail"])

    except Exception as e:
        st.error(f"💥 執行 PPS Testing 發生錯誤：{e}")

# ==========================
# ② Summary / Detail（顯示區）
# ==========================

# --- Summary ---
if "show_summary" not in st.session_state:
    st.session_state["show_summary"] = False

if st.button("📄 顯示 / 隱藏 Summary"):
    st.session_state["show_summary"] = not st.session_state["show_summary"]

if st.session_state["show_summary"]:
    st.subheader("📄 Summary")
    st.json(st.session_state["pps_testing_summary"])

# --- Detail ---
if "show_detail" not in st.session_state:
    st.session_state["show_detail"] = False

if st.button("📑 顯示 / 隱藏 Detail（FM / t% / PM / IA）"):
    st.session_state["show_detail"] = not st.session_state["show_detail"]

if st.session_state["show_detail"]:
    st.subheader("📑 Detail（FM / t% / PM / IA）")
    st.dataframe(st.session_state["pps_testing_detail"])


# ======================
# ② 輸出按鈕（可選）
# ======================
def fmt(x):
    return float(f"{x:.2f}")

# ======================
# ③ 匯出按鈕（教學 / 比賽用）
# ======================

if st.button("📥 匯出 Summary + Detail（教學用）"):
    if "pps_testing_summary" not in st.session_state:
        st.warning("⚠️ 尚未執行 PPS Testing。")
    else:
        summary_df = pd.DataFrame([st.session_state["pps_testing_summary"]])
        detail_df = st.session_state["pps_testing_detail"]

        with pd.ExcelWriter("pps_testing_output.xlsx", engine="openpyxl") as writer:
            detail_df.to_excel(writer, sheet_name="Detail", index=False)
            summary_df.to_excel(writer, sheet_name="Summary", index=False)

        st.success("✅ 已在程式所在資料夾產生 pps_testing_output.xlsx")

# ======================
# ③ 教學示範區（第三段）
# ======================
# st.markdown("<h3>③ 教學示範（說明用，不參與計算）</h3>", unsafe_allow_html=True)
st.set_page_config(page_title="PPS Testing", layout="wide")
st.title("③ 教學示範（僅作計算之說明，不參與計算）")

if "show_teaching" not in st.session_state:
    st.session_state["show_teaching"] = False

if st.button("🎓 顯示 / 隱藏 教學示範（Teaching Notes）"):
    st.session_state["show_teaching"] = not st.session_state["show_teaching"]

if st.session_state["show_teaching"]:
    st.markdown("""
### 📘 PPS Testing 教學重點（中英對照）

**1️⃣ 抽樣區間（Sampling Interval）**  
Sampling Interval = Book Value ÷ Sample Size  

**2️⃣ 事實錯誤（Factual Misstatement, FM）**  
FM = Recorded Amount − Audited Amount  

**3️⃣ 汙染率（Tainting Percentage, t%）**  
t% = FM ÷ Recorded Amount  

**4️⃣ 推計誤差（Projected Misstatement, PM）**  
- 若 FM < 0  
  → PM = FM  
- 若 FM ≥ 0 且 Recorded < Sampling Interval  
  → PM = t% × Sampling Interval  
- 其他情況  
  → PM = FM  

**5️⃣ 增額風險（Incremental Allowance, IA）**  
依 PM 大小排序後，  
套用 **Risk Factor Table（誤受險查表）** 計算每一筆 IA  

**6️⃣ 基本精確度（Basic Precision, BP）**  
BP = Sampling Interval × Risk Factor（Ranking = 0）  

**7️⃣ 審計風險補貼（Audit Risk Premium, ASR）**  
ASR = BP + IA Total  

**8️⃣ 上限誤差（Upper Misstatement Limit, UML）**  
UML = PM Total + ASR  

**9️⃣ 審計決策（Audit Decision）**  
- 若 UML ≤ TM → **Accept（可接受）**  
- 若 UML > TM → **Reject（拒絕，需擴大查核）**
""")
