# app.py - AdPulse Studio (full, final)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO


st.set_page_config(page_title="AdPulse Studio", layout="wide", page_icon="🎧")


plt.rcParams.update({
    "font.size": 7,
    "axes.titlesize": 9,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 7
})


st.markdown(
    """
    <style>
    :root {
      --bg: #0c0f14;
      --card: #0f1720;
      --muted: #9aa4b2;
      --neon1: #7dd3fc;
      --neon2: #c7b3ff;
      --neon3: #86efac;
    }
    body { background: var(--bg); color: #e6eef6; }
    .header-card {
      background: rgba(255,255,255,0.02);
      border-radius: 14px;
      padding: 12px;
      box-shadow: 0 8px 30px rgba(2,6,23,0.6);
      display:flex;
      gap:14px;
      align-items:center;
      margin-bottom:12px;
    }
    .logo-wrap { width:96px; height:96px; flex: 0 0 96px; }
    .app-title {
      font-family: 'Trebuchet MS', sans-serif;
      font-size: 36px;
      font-weight: 800;
      margin: 0;
      color: #ffffff;
      text-shadow: 0 0 10px #6b21a8, 0 0 20px #06b6d4;
    }
    .app-slogan { color: var(--muted); margin-top:6px; font-size:13px; }
    .slogan-pill {
      display:inline-block;
      padding:6px 10px;
      border-radius: 10px;
      background: linear-gradient(90deg, rgba(125,211,252,0.06), rgba(199,179,255,0.04));
      box-shadow: 0 6px 18px rgba(124,58,237,0.04);
      margin-right:10px;
    }

    .graph-card {
      background: rgba(255,255,255,0.02);
      border-radius: 10px;
      padding: 8px;
      border: 1px solid rgba(125,211,252,0.12);
      box-shadow: 0 8px 30px rgba(2,6,23,0.6);
      margin-bottom: 10px;
    }

    /* small heading neon border */
    .graph-title {
      color: var(--neon1);
      font-weight:700;
      font-size:12px;
      margin: 0 0 6px 0;
      text-shadow: 0 0 8px rgba(125,211,252,0.12);
    }

    /* tiny caption */
    .muted { color: var(--muted); font-size:12px; }

    </style>
    """,
    unsafe_allow_html=True,
)

svg_logo = """
<svg viewBox="0 0 240 240" xmlns="http://www.w3.org/2000/svg" width="96" height="96">
  <defs>
    <radialGradient id="g" cx="50%" cy="40%">
      <stop offset="0%" stop-color="#c7b3ff" stop-opacity="1"/>
      <stop offset="60%" stop-color="#7dd3fc" stop-opacity="0.95"/>
      <stop offset="100%" stop-color="#86efac" stop-opacity="0.7"/>
    </radialGradient>
  </defs>
  <rect width="240" height="240" rx="36" fill="url(#g)"/>
  <g transform="translate(120,120)">
    <circle r="30" fill="white" opacity="0.97"/>
    <rect x="-5" y="-14" width="10" height="22" rx="2" fill="#7c3aed"/>
    <rect x="-10" y="10" width="20" height="5" rx="2" fill="#4ade80" opacity="0.95"/>
    <circle r="46" fill="none" stroke="#7c3aed" stroke-opacity="0.14" stroke-width="6"/>
    <circle r="68" fill="none" stroke="#7dd3fc" stroke-opacity="0.10" stroke-width="5"/>
  </g>
</svg>
"""


st.markdown(
    f"""
    <div class="header-card">
      <div class="logo-wrap">{svg_logo}</div>
      <div style="flex:1;">
        <div style="display:flex; align-items:center; gap:10px;">
          <div>
            <h1 class="app-title">AdPulse Studio</h1>
            <div style="display:flex; align-items:center; gap:8px; margin-top:4px;">
              <div class="slogan-pill">Feel the Beat of Your Revenue</div>
              <div class="muted">• Greedy allocation • DP comparison • Visual analytics</div>
            </div>
          </div>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


st.markdown('<div style="display:flex; gap:12px; align-items:center;">', unsafe_allow_html=True)
st.subheader("Upload Ad Dataset (CSV)")
sample_csv = """ad_id,duration,revenue
AD01,30,300
AD02,45,420
AD03,25,180
AD04,50,550
AD05,20,150
AD06,40,360
AD07,35,310
AD08,55,600
AD09,30,250
AD10,15,120
"""
st.download_button("Download sample CSV", sample_csv, file_name="sample_ads.csv")
st.markdown('</div>', unsafe_allow_html=True)

uploaded = st.file_uploader("Upload CSV (columns: ad_id,duration,revenue)", type=["csv"])

if not uploaded:
    st.info("Upload a CSV to run optimization (or download sample CSV).")
    st.stop()

df = pd.read_csv(uploaded)
if not {"ad_id", "duration", "revenue"}.issubset(set(df.columns)):
    st.error("CSV must contain columns: ad_id,duration,revenue")
    st.stop()


df["duration"] = df["duration"].astype(float)
df["revenue"] = df["revenue"].astype(float)
df["rps"] = df["revenue"] / df["duration"]


st.markdown('<div class="graph-card">', unsafe_allow_html=True)
st.write("### Uploaded Data")
st.dataframe(df, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)


c1, c2 = st.columns([2,1])
with c1:
    total_time = st.number_input("Total available ad time (seconds)", min_value=1, value=120, step=1)
with c2:
    run = st.button("Run Optimization ▶")


def greedy_fractional(df_in, max_time):
    temp = df_in.sort_values("rps", ascending=False).reset_index(drop=True)
    sel = []
    used = 0.0
    rev_total = 0.0
    for _, r in temp.iterrows():
        if used >= max_time - 1e-9:
            break
        if used + r["duration"] <= max_time + 1e-9:
            sel.append((r["ad_id"], float(r["duration"]), float(r["revenue"]), 1.0))
            used += r["duration"]
            rev_total += r["revenue"]
        else:
            rem = max_time - used
            if rem <= 1e-9:
                break
            frac = rem / r["duration"]
            rev_taken = r["revenue"] * frac
            sel.append((r["ad_id"], float(rem), float(rev_taken), float(frac)))
            rev_total += rev_taken
            used += rem
            break
    return rev_total, used, sel

def dp_knapsack(df_in, max_time):
    durations = df_in["duration"].round().astype(int).tolist()
    revenues = df_in["revenue"].tolist()
    items = df_in["ad_id"].tolist()
    n = len(durations)
    W = int(max_time)
    dp = np.zeros((n+1, W+1))
    for i in range(1, n+1):
        for w in range(1, W+1):
            if durations[i-1] <= w:
                dp[i,w] = max(dp[i-1,w], dp[i-1,w-durations[i-1]] + revenues[i-1])
            else:
                dp[i,w] = dp[i-1,w]
    total_rev = float(dp[n,W])
    chosen = []
    w = W
    for i in range(n, 0, -1):
        if dp[i,w] != dp[i-1,w]:
            chosen.append(items[i-1])
            w -= durations[i-1]
    chosen.reverse()
    return total_rev, chosen


if run:
    with st.spinner("Optimizing..."):
        g_rev, g_used, g_sel = greedy_fractional(df, total_time)
        d_rev, d_sel = dp_knapsack(df, total_time)

    greedy_df = pd.DataFrame(g_sel, columns=["ad_id", "time_taken", "rev_taken", "fraction"])
    dp_df = df[df["ad_id"].isin(d_sel)].copy() if d_sel else pd.DataFrame(columns=df.columns)

    st.markdown('<div class="graph-card" style="display:flex; gap:10px;">', unsafe_allow_html=True)
    colA, colB, colC = st.columns(3)
    colA.metric("Greedy Revenue", f"₹{g_rev:.2f}")
    colB.metric("DP Revenue (0/1)", f"₹{d_rev:.2f}")
    colC.metric("Greedy Time Used", f"{int(g_used)} sec")
    st.markdown("</div>", unsafe_allow_html=True)

   
    row1c1, row1c2 = st.columns(2)
    row2c1, row2c2 = st.columns(2)

    theme_colors = ["#c7b3ff", "#bfe9ff", "#bbf7d0", "#a78bfa", "#7dd3fc"]

    with row1c1:
        st.markdown('<div class="graph-card">', unsafe_allow_html=True)
        st.markdown('<div class="graph-title">🥧 Ad Time Usage (Greedy)</div>', unsafe_allow_html=True)
        if len(greedy_df) > 0:
            labels = greedy_df["ad_id"]
            sizes = greedy_df["time_taken"]
            fig1, ax1 = plt.subplots(figsize=(3.6,1.8))
            ax1.pie(sizes, labels=labels, autopct="%1.0f%%", textprops={"fontsize":6}, colors=theme_colors[:len(labels)])
            ax1.set_title("", fontsize=8)
            st.pyplot(fig1)
            plt.close(fig1)
        else:
            st.info("No greedy selection to show")
        st.markdown('</div>', unsafe_allow_html=True)

    with row1c2:
        st.markdown('<div class="graph-card">', unsafe_allow_html=True)
        st.markdown('<div class="graph-title">📊 Revenue per Ad</div>', unsafe_allow_html=True)
        fig2, ax2 = plt.subplots(figsize=(3.6,1.8))
        ax2.bar(df["ad_id"], df["revenue"], color=theme_colors[:len(df["ad_id"])])
        ax2.tick_params(labelsize=6)
        ax2.set_xlabel("", fontsize=7)
        ax2.set_ylabel("", fontsize=7)
        st.pyplot(fig2)
        plt.close(fig2)
        st.markdown('</div>', unsafe_allow_html=True)

   
    with row2c1:
        st.markdown('<div class="graph-card">', unsafe_allow_html=True)
        st.markdown('<div class="graph-title">⚔️ Greedy vs DP — Total Revenue</div>', unsafe_allow_html=True)
        fig3, ax3 = plt.subplots(figsize=(3.6,1.8))
        ax3.bar(["Greedy", "DP"], [g_rev, d_rev], color=["#7dd3fc", "#c7b3ff"])
        ax3.tick_params(labelsize=6)
        ax3.set_ylabel("", fontsize=7)
        st.pyplot(fig3)
        plt.close(fig3)
        st.markdown('</div>', unsafe_allow_html=True)

   
    with row2c2:
        st.markdown('<div class="graph-card">', unsafe_allow_html=True)
        st.markdown('<div class="graph-title">📈 Cumulative Revenue (Greedy)</div>', unsafe_allow_html=True)
        if len(greedy_df) > 0:
            greedy_df["cum_time"] = greedy_df["time_taken"].cumsum()
            greedy_df["cum_rev"] = greedy_df["rev_taken"].cumsum()
            fig4, ax4 = plt.subplots(figsize=(3.6,1.8))
            ax4.plot(greedy_df["cum_time"], greedy_df["cum_rev"], marker="o", markersize=3, linewidth=1, color="#34d399")
            ax4.tick_params(labelsize=6)
            ax4.set_xlabel("", fontsize=7)
            ax4.set_ylabel("", fontsize=7)
            ax4.grid(alpha=0.2)
            st.pyplot(fig4)
            plt.close(fig4)
        else:
            st.info("No greedy cumulative data")
        st.markdown('</div>', unsafe_allow_html=True)

   
    st.markdown('<div class="graph-card">', unsafe_allow_html=True)
    two1, two2 = st.columns(2)
    with two1:
        st.write("### Selected Ads — Greedy (fractional)")
        if len(greedy_df) > 0:
            st.dataframe(greedy_df, use_container_width=True)
        else:
            st.info("No greedy selection.")
    with two2:
        st.write("### Selected Ads — DP (0/1)")
        if len(dp_df) > 0:
            st.dataframe(dp_df[["ad_id", "duration", "revenue"]], use_container_width=True)
        else:
            st.info("No DP selection.")

 
    st.markdown("---")
    report = []
    report.append("AdPulse Studio — Optimization Report")
    report.append(f"Total allowed time: {total_time} sec")
    report.append(f"Greedy Revenue: ₹{g_rev:.2f}")
    report.append(f"Greedy Time used: {int(g_used)} sec")
    report.append(f"DP Revenue: ₹{d_rev:.2f}")
    report.append("")
    report.append("Greedy selection (ad_id, time_taken, rev_taken, fraction):")
    for s in g_sel:
        report.append(str(s))
    report_text = "\n".join(report)
    st.download_button("Download Report (.txt)", report_text, file_name="adpulse_report.txt")

    if len(greedy_df) > 0:
        st.download_button("Download Selected Ads (.csv)", data=greedy_df.to_csv(index=False), file_name="selected_ads.csv")


st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)
st.caption("AdPulse Studio • Feel the Beat of Your Revenue • Built with care ❤️")

