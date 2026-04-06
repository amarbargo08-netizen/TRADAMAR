# ============================================================
# TRADAMAR - APPLICATION STREAMLIT
# ============================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import linregress
from scipy.signal import argrelextrema

# --- CONFIGURATION PAGE ---
st.set_page_config(
    page_title="TRADAMAR",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- THÈME ---
if "theme" not in st.session_state:
    st.session_state.theme = "sombre"

DARK = st.session_state.theme == "sombre"

# --- COULEURS ---
BG_MAIN  = "#0a0f2c" if DARK else "#f0f4ff"
BG_SEC   = "#0d1b3e" if DARK else "#ffffff"
ACCENT   = "#00ff88"
BLUE_NEO = "#00aaff"
TEXT     = "#ffffff"  if DARK else "#0a0f2c"
TEXT_SEC = "#00aaff"

# --- STYLE CSS ---
st.markdown(f"""
<style>
    .stApp {{ background-color: {BG_MAIN}; }}
    .block-container {{ padding: 1rem; }}
    h1, h2, h3 {{ color: {ACCENT} !important; }}
    p, label, div {{ color: {TEXT} !important; }}
    .stTabs [data-baseweb="tab"] {{
        color: {TEXT_SEC};
        border-bottom: 2px solid {ACCENT};
    }}
    .stTabs [aria-selected="true"] {{
        color: {ACCENT} !important;
        border-bottom: 3px solid {ACCENT} !important;
    }}
    .stSelectbox label {{ color: {TEXT} !important; }}
    .stMetric label {{ color: {TEXT_SEC} !important; }}
    .stMetric div {{ color: {ACCENT} !important; font-size: 1.2rem !important; }}
    .stButton button {{
        background-color: {ACCENT};
        color: {BG_MAIN};
        border: none;
        font-weight: bold;
        border-radius: 8px;
    }}
    .stButton button:hover {{
        background-color: {BLUE_NEO};
        color: white;
    }}
    .info-box {{
        background-color: {BG_SEC};
        border: 1px solid {BLUE_NEO};
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }}
    .warning-box {{
        background-color: {BG_SEC};
        border: 1px solid #ff8800;
        border-radius: 10px;
        padding: 10px;
        margin: 10px 0;
        color: #ff8800 !important;
    }}
</style>
""", unsafe_allow_html=True)

# --- LISTE DES ACTIFS ---
ACTIFS = {
    "BTC-USD"  : "Bitcoin",
    "ETH-USD"  : "Ethereum",
    "GBPJPY=X" : "GBP/JPY",
    "AUDCAD=X" : "AUD/CAD",
    "AUDUSD=X" : "AUD/USD",
    "SI=F"     : "Silver",
    "GBPUSD=X" : "GBP/USD",
    "USDJPY=X" : "USD/JPY",
    "EURUSD=X" : "EUR/USD",
    "GC=F"     : "Gold",
}

INTERVAL     = "1h"
PERIOD       = "180d"
WINDOW       = 10
ZOOM_BOUGIES = 100

# ============================================================
# FONCTIONS DU PIPELINE
# ============================================================

@st.cache_data(ttl=3600)
def load_data(symbol, interval, period):
    df = yf.download(symbol, interval=interval, period=period)
    df.columns = df.columns.droplevel(1)
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.dropna(inplace=True)
    return df

def detect_points(df, window=10):
    high_idx = argrelextrema(df["High"].values, np.greater_equal, order=window)[0]
    low_idx  = argrelextrema(df["Low"].values,  np.less_equal,    order=window)[0]
    return high_idx, low_idx

def calc_regression(indices, values, outlier_threshold=1.5):
    indices = np.array(indices)
    values  = np.array(values)
    slope, intercept, r, p, se = linregress(indices, values)
    predicted = slope * indices + intercept
    distances = np.abs(values - predicted)
    mean_dist = np.mean(distances)
    mask = distances <= outlier_threshold * mean_dist
    if mask.sum() >= 2:
        slope, intercept, r, p, se = linregress(indices[mask], values[mask])
    return slope, intercept, r**2

def line_value(slope, intercept, idx):
    return slope * idx + intercept

def lines_cross_in_past(s_high, i_high, s_low, i_low, start, end):
    for idx in range(start, end):
        if line_value(s_low, i_low, idx) >= line_value(s_high, i_high, idx):
            return True
    return False

def validate_structure(s_high, int_high, s_low, int_low, start_idx, end_idx):
    dist_debut = line_value(s_high, int_high, start_idx) - line_value(s_low, int_low, start_idx)
    dist_fin   = line_value(s_high, int_high, end_idx)   - line_value(s_low, int_low, end_idx)
    if dist_debut <= 0 or dist_fin <= 0:
        return False
    if dist_fin > dist_debut * 2:
        return False
    duree      = max(end_idx - start_idx, 1)
    projection = end_idx + duree * 2
    for idx in range(end_idx, int(projection)):
        val_high = line_value(s_high, int_high, idx)
        val_low  = line_value(s_low,  int_low,  idx)
        if val_low >= val_high:
            if idx - end_idx < duree:
                return False
            break
    seuil_pente = (dist_debut / duree) * 0.5
    if s_high > seuil_pente and s_low < -seuil_pente:
        return False
    if s_high < -seuil_pente and s_low > seuil_pente:
        return False
    return True

def detect_structures(df, high_idx, low_idx, min_points=3, break_tol=0.002):
    structures = []
    prices = df["Close"].values
    highs  = df["High"].values
    lows   = df["Low"].values
    n      = len(prices)
    h_idx  = list(high_idx)
    l_idx  = list(low_idx)
    i      = 0
    while i < len(h_idx) - min_points + 1:
        h_group = h_idx[i:i + min_points]
        first_high_pos   = h_group[0]
        l_group_filtered = [idx for idx in l_idx if idx >= first_high_pos]
        if len(l_group_filtered) < min_points:
            i += 1
            continue
        l_group   = l_group_filtered[:min_points]
        s_high, int_high, r2_high = calc_regression(h_group, highs[h_group])
        s_low,  int_low,  r2_low  = calc_regression(l_group, lows[l_group])
        start_idx = min(h_group[0], l_group[0])
        end_idx   = max(h_group[-1], l_group[-1])
        if lines_cross_in_past(s_high, int_high, s_low, int_low, start_idx, end_idx):
            i += 1
            continue
        if not validate_structure(s_high, int_high, s_low, int_low, start_idx, end_idx):
            i += 1
            continue
        j         = end_idx + 1
        broken    = False
        break_idx = None
        while j < n:
            val_high = line_value(s_high, int_high, j)
            val_low  = line_value(s_low,  int_low,  j)
            if prices[j] > val_high * (1 + break_tol):
                broken    = True
                break_idx = j
                break
            if prices[j] < val_low * (1 - break_tol):
                broken    = True
                break_idx = j
                break
            j += 1

        if abs(s_high) < 0.001 and abs(s_low) < 0.001:
            pattern = "Range horizontal"
        elif s_high > 0.001 and s_low > 0.001:
            pattern = "Canal ascendant"
        elif s_high < -0.001 and s_low < -0.001:
            pattern = "Canal descendant"
        elif s_high < -0.001 and s_low > 0.001:
            pattern = "Triangle symétrique"
        elif abs(s_high) < 0.001 and s_low > 0.001:
            pattern = "Triangle ascendant"
        elif s_high < -0.001 and abs(s_low) < 0.001:
            pattern = "Triangle descendant"
        else:
            pattern = "Canal"

        structures.append({
            "start"    : start_idx,
            "end"      : break_idx if break_idx else j,
            "s_high"   : s_high,
            "int_high" : int_high,
            "s_low"    : s_low,
            "int_low"  : int_low,
            "r2_high"  : r2_high,
            "r2_low"   : r2_low,
            "broken"   : broken,
            "break_idx": break_idx,
            "pattern"  : pattern
        })
        if broken and break_idx:
            i = next((k for k, idx in enumerate(h_idx) if idx >= break_idx), len(h_idx))
        else:
            break
    return structures

def calc_zones(structures, high_idx, low_idx, highs, lows, prices):
    ZONE_FACTOR  = 1.2
    ZONE_MAX_PCT = 0.015
    for s in structures:
        h_group    = [idx for idx in high_idx if s["start"] <= idx <= s["end"]]
        l_group    = [idx for idx in low_idx  if s["start"] <= idx <= s["end"]]
        prix_moyen = np.mean(prices[s["start"]:s["end"]])
        def thickness(indices, values, slope, intercept):
            indices   = np.array(indices)
            values    = np.array(values)
            predicted = slope * indices + intercept
            distances = np.abs(values - predicted)
            mean_d    = np.mean(distances)
            mask      = distances <= mean_d * 1.5
            if mask.sum() >= 2:
                distances = distances[mask]
            return min(np.mean(distances) * ZONE_FACTOR, prix_moyen * ZONE_MAX_PCT)
        s["zone_high"] = thickness(h_group, highs[h_group], s["s_high"], s["int_high"]) if len(h_group) >= 2 else 0
        s["zone_low"]  = thickness(l_group, lows[l_group],  s["s_low"],  s["int_low"])  if len(l_group) >= 2 else 0
    return structures

def detect_breakouts(structures, prices, confirm_factor=0.2):
    breakouts = []
    for s in structures:
        if s["broken"] and s["break_idx"]:
            j         = s["break_idx"]
            line_high = line_value(s["s_high"], s["int_high"], j)
            line_low  = line_value(s["s_low"],  s["int_low"],  j)
            close     = prices[j]
            mid       = (line_high + line_low) / 2
            btype     = "Haussière" if close > mid else "Baissière"
            breakouts.append({
                "idx"      : j,
                "direction": btype,
                "price"    : close,
                "structure": s,
            })
    return breakouts

def generate_signals(breakouts, prices, rr_ratio=2.0, rr_wide=1.5):
    all_zones = []
    for b in breakouts:
        all_zones.append(b["structure"]["zone_high"])
        all_zones.append(b["structure"]["zone_low"])
    zone_plafond = np.median(all_zones) * 1.2 if all_zones else 0
    signals = []
    for b in breakouts:
        s     = b["structure"]
        idx   = b["idx"]
        entry = b["price"]
        line_high = line_value(s["s_high"], s["int_high"], idx)
        line_low  = line_value(s["s_low"],  s["int_low"],  idx)
        if b["direction"] == "Haussière":
            zone_utilisee = min(s["zone_high"], zone_plafond)
            rr_utilise    = rr_wide if s["zone_high"] > zone_plafond else rr_ratio
            sl   = line_high - zone_utilisee
            risk = entry - sl
            tp   = entry + (risk * rr_utilise)
        else:
            zone_utilisee = min(s["zone_low"], zone_plafond)
            rr_utilise    = rr_wide if s["zone_low"] > zone_plafond else rr_ratio
            sl   = line_low + zone_utilisee
            risk = sl - entry
            tp   = entry - (risk * rr_utilise)
        rr = round(abs(tp - entry) / abs(entry - sl), 2) if abs(entry - sl) > 0 else 0
        signals.append({
            "idx"      : idx,
            "direction": b["direction"],
            "entry"    : round(entry, 2),
            "sl"       : round(sl,    2),
            "tp"       : round(tp,    2),
            "rr"       : rr,
            "pattern"  : s["pattern"],
            "is_last"  : False
        })
    if signals:
        signals[-1]["is_last"] = True
    return signals

def generate_chart(df, structures, signals, high_idx, low_idx, dark=True, zoom=100):
    prices = df["Close"].values
    highs  = df["High"].values
    lows   = df["Low"].values
    opens  = df["Open"].values
    closes = df["Close"].values
    n      = len(prices)

    zoom_start = max(0, n - zoom)
    bg_color   = "#0a0f2c" if dark else "#f0f4ff"
    text_color = "white"   if dark else "#0a0f2c"
    grid_color = "#1a2a5e" if dark else "#cccccc"

    fig, ax = plt.subplots(figsize=(16, 7), facecolor=bg_color)
    ax.set_facecolor(bg_color)

    # --- BOUGIES JAPONAISES ---
    for i in range(zoom_start, n):
        x     = i - zoom_start
        op    = opens[i]
        cl    = closes[i]
        hi    = highs[i]
        lo    = lows[i]
        color = "#00ff88" if cl >= op else "#ff4444"

        # Mèche
        ax.plot([x, x], [lo, hi], color=color, linewidth=0.8, zorder=2)

        # Corps
        body_bottom = min(op, cl)
        body_height = abs(cl - op)
        if body_height == 0:
            body_height = hi * 0.0001
        rect = mpatches.Rectangle(
            (x - 0.4, body_bottom),
            0.8, body_height,
            facecolor=color, edgecolor=color,
            linewidth=0.5, zorder=3
        )
        ax.add_patch(rect)

    # --- STRUCTURES ---
    for s in structures:
        if s["end"] < zoom_start:
            continue
        x_start   = max(s["start"], zoom_start) - zoom_start
        x_end_s   = min(s["end"], n - 1) - zoom_start
        idx_range = np.arange(max(s["start"], zoom_start), min(s["end"], n - 1) + 1)
        x_range   = idx_range - zoom_start

        lh = np.array([line_value(s["s_high"], s["int_high"], i) for i in idx_range])
        ll = np.array([line_value(s["s_low"],  s["int_low"],  i) for i in idx_range])

        color_h = "#333355" if s["broken"] else "#ff4444"
        color_l = "#333355" if s["broken"] else "#00ff88"
        lw      = 1.0       if s["broken"] else 2.0
        alpha   = 0.4       if s["broken"] else 1.0

        ax.plot(x_range, lh, color=color_h, linewidth=lw, alpha=alpha, zorder=4)
        ax.plot(x_range, ll, color=color_l, linewidth=lw, alpha=alpha, zorder=4)

        if not s["broken"]:
            ax.fill_between(x_range, ll, lh,
                            color="#1a3a2a", alpha=0.10, zorder=1)
            ax.fill_between(x_range, lh, lh + s["zone_high"],
                            color="#ff4444", alpha=0.10, zorder=1)
            ax.fill_between(x_range, ll - s["zone_low"], ll,
                            color="#00ff88", alpha=0.10, zorder=1)

    # --- ZIGZAG ---
    for s in structures:
        if s["end"] < zoom_start:
            continue
        points_high = [{"idx": i, "price": highs[i], "type": "HIGH"}
                       for i in high_idx if s["start"] <= i <= s["end"]]
        points_low  = [{"idx": i, "price": lows[i],  "type": "LOW"}
                       for i in low_idx  if s["start"] <= i <= s["end"]]
        all_points  = sorted(points_high + points_low, key=lambda x: x["idx"])
        filtered    = []
        for p in all_points:
            if not filtered:
                filtered.append(p)
            elif p["type"] != filtered[-1]["type"]:
                filtered.append(p)
            else:
                if p["type"] == "HIGH" and p["price"] > filtered[-1]["price"]:
                    filtered[-1] = p
                elif p["type"] == "LOW" and p["price"] < filtered[-1]["price"]:
                    filtered[-1] = p

        for m in range(len(filtered) - 1):
            p1 = filtered[m]
            p2 = filtered[m + 1]
            if p1["idx"] >= zoom_start and p2["idx"] >= zoom_start:
                ax.plot(
                    [p1["idx"] - zoom_start, p2["idx"] - zoom_start],
                    [p1["price"], p2["price"]],
                    color="#00aaff", linewidth=1.0, alpha=0.7, zorder=4
                )

    # --- SIGNAUX ---
    for sig in signals:
        idx   = sig["idx"]
        x_sig = max(idx, zoom_start) - zoom_start
        width = min(30, n - zoom_start - x_sig)
        x_end = x_sig + width

        if width <= 0:
            continue

        if idx < zoom_start:
            color_tp = "#444444"
            color_sl = "#444444"
            color_en = "#666666"
        else:
            color_tp = "#00ff88"
            color_sl = "#ff2222"
            color_en = "#ffffff"

        ax.hlines(sig["tp"],    x_sig, x_end, colors=color_tp,
                  linewidth=1.5, linestyle="--", zorder=5)
        ax.hlines(sig["entry"], x_sig, x_end, colors=color_en,
                  linewidth=1.5, zorder=5)
        ax.hlines(sig["sl"],    x_sig, x_end, colors=color_sl,
                  linewidth=1.5, linestyle="--", zorder=5)

        ax.fill_between(range(x_sig, x_end), sig["entry"], sig["tp"],
                        color=color_tp, alpha=0.06, zorder=1)
        ax.fill_between(range(x_sig, x_end), sig["sl"], sig["entry"],
                        color=color_sl, alpha=0.06, zorder=1)

        if idx >= zoom_start:
            ax.text(x_end + 1, sig["tp"],
                    f"TP {sig['tp']:.2f}",
                    color=color_tp, fontsize=7, va="center")
            ax.text(x_end + 1, sig["entry"],
                    f"{sig['direction']} {sig['entry']:.2f}",
                    color=color_en, fontsize=7, va="center", fontweight="bold")
            ax.text(x_end + 1, sig["sl"],
                    f"SL {sig['sl']:.2f}",
                    color=color_sl, fontsize=7, va="center")

    # --- STYLE ---
    ax.tick_params(colors=text_color)
    ax.spines[["top", "right", "left", "bottom"]].set_edgecolor(grid_color)
    ax.grid(color=grid_color, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_title("TRADAMAR — Analyse PATRAD",
                 color="#00ff88", fontsize=13, pad=10)
    plt.tight_layout()
    return fig

# ============================================================
# INTERFACE STREAMLIT
# ============================================================

st.markdown(f"""
<div style='text-align:center; padding: 10px 0;'>
    <h1 style='color:#00ff88; font-size:2.5rem; letter-spacing:4px;'>
        📈 TRADAMAR
    </h1>
    <p style='color:#00aaff; font-size:0.9rem;'>
        Intelligence Artificielle d'Analyse Technique
    </p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🔴 LIVE", "🗂️ Archives", "⚙️ Paramètres"])

# ============================================================
# TAB LIVE
# ============================================================
with tab1:

    col1, col2 = st.columns([3, 1])
    with col1:
        symbole_nom = st.selectbox("", list(ACTIFS.values()),
                                   label_visibility="collapsed")
        symbol      = [k for k, v in ACTIFS.items() if v == symbole_nom][0]
    with col2:
        analyser = st.button("🔍 Analyser", use_container_width=True)

    if analyser:
        with st.spinner("⏳ Analyse en cours..."):

            df                = load_data(symbol, INTERVAL, PERIOD)
            prices            = df["Close"].values
            highs             = df["High"].values
            lows              = df["Low"].values
            high_idx, low_idx = detect_points(df, WINDOW)
            structures        = detect_structures(df, high_idx, low_idx)
            structures        = calc_zones(structures, high_idx, low_idx, highs, lows, prices)
            breakouts         = detect_breakouts(structures, prices)
            signals           = generate_signals(breakouts, prices)

            fig = generate_chart(df, structures, signals,
                                 high_idx, low_idx,
                                 dark=DARK, zoom=ZOOM_BOUGIES)
            st.pyplot(fig)
            st.markdown("---")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"""
                <div class='info-box'>
                    <h3 style='color:#00ff88; margin:0;'>📊 PATRAD</h3>
                    <p style='color:#00aaff; font-size:0.8rem;'>Analyse Price Action</p>
                </div>
                """, unsafe_allow_html=True)

                if structures:
                    last_s    = structures[-1]
                    last_idx  = last_s["end"]
                    res_level = line_value(last_s["s_high"], last_s["int_high"], last_idx)
                    sup_level = line_value(last_s["s_low"],  last_s["int_low"],  last_idx)
                    zone_key  = (res_level + sup_level) / 2

                    if breakouts:
                        direction = breakouts[-1]["direction"]
                        dir_color = "#00ff88" if direction == "Haussière" else "#ff4444"
                        dir_icon  = "📈" if direction == "Haussière" else "📉"
                    else:
                        direction = "En attente"
                        dir_color = "#00aaff"
                        dir_icon  = "⏳"

                    st.markdown(f"""
                    <div style='background:{BG_SEC}; padding:15px; border-radius:10px;
                                border:1px solid #1a2a5e;'>
                        <p style='color:#aaaaaa; margin:2px;'>Pattern détecté</p>
                        <p style='color:#00aaff; font-size:1.1rem; font-weight:bold;
                                  margin:2px;'>{last_s['pattern']}</p>
                        <hr style='border-color:#1a2a5e;'>
                        <p style='color:#aaaaaa; margin:2px;'>Direction probable</p>
                        <p style='color:{dir_color}; font-size:1.1rem;
                                  font-weight:bold; margin:2px;'>
                            {dir_icon} {direction}
                        </p>
                        <hr style='border-color:#1a2a5e;'>
                        <p style='color:#aaaaaa; margin:2px;'>Résistance</p>
                        <p style='color:#ff4444; font-size:1rem; margin:2px;'>
                            {res_level:.2f}
                        </p>
                        <p style='color:#aaaaaa; margin:2px;'>Support</p>
                        <p style='color:#00ff88; font-size:1rem; margin:2px;'>
                            {sup_level:.2f}
                        </p>
                        <p style='color:#aaaaaa; margin:2px;'>Zone clé</p>
                        <p style='color:#00aaff; font-size:1rem; margin:2px;'>
                            {zone_key:.2f}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown(f"""
                    <div class='warning-box'>
                        ⚠️ Analyse indicative uniquement.<br>
                        Ne constitue pas un conseil financier.
                    </div>
                    """, unsafe_allow_html=True)

                else:
                    st.warning("Aucune structure détectée.")

            with col2:
                st.markdown(f"""
                <div class='info-box'>
                    <h3 style='color:#00ff88; margin:0;'>🤖 INTRAD</h3>
                    <p style='color:#00aaff; font-size:0.8rem;'>
                        Analyse Indicateurs Techniques
                    </p>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(f"""
                <div style='background:{BG_SEC}; padding:15px; border-radius:10px;
                            border:1px solid #1a2a5e; text-align:center;'>
                    <p style='color:#555555; font-size:2rem;'>🔧</p>
                    <p style='color:#555555;'>Module en développement</p>
                    <p style='color:#333366; font-size:0.8rem;'>Bientôt disponible</p>
                </div>
                """, unsafe_allow_html=True)

# ============================================================
# TAB ARCHIVES
# ============================================================
with tab2:
    st.markdown(f"""
    <div class='info-box'>
        <h3 style='color:#00ff88;'>🗂️ Archives</h3>
        <p style='color:#00aaff;'>
            Les 2 dernières analyses par actif seront affichées ici.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# TAB PARAMÈTRES
# ============================================================
with tab3:
    st.markdown(f"""
    <div class='info-box'>
        <h3 style='color:#00ff88;'>⚙️ Paramètres</h3>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"<p style='color:{TEXT};'><b>🎨 Thème :</b></p>",
                unsafe_allow_html=True)
    theme_choice = st.radio(
        "Thème",
        ["🌙 Sombre", "☀️ Clair"],
        index=0 if DARK else 1,
        horizontal=True,
        label_visibility="collapsed"
    )
    if st.button("Appliquer le thème"):
        st.session_state.theme = "sombre" if "Sombre" in theme_choice else "clair"
        st.rerun()

    st.markdown("---")
    st.markdown(f"<p style='color:{TEXT};'><b>⏱️ Timeframe :</b></p>",
                unsafe_allow_html=True)
    timeframe = st.selectbox(
        "Intervalle", ["1h", "4h", "1d"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown(f"<p style='color:{TEXT};'><b>📊 Actifs surveillés :</b></p>",
                unsafe_allow_html=True)
    cols = st.columns(2)
    for i, (symbol_key, nom) in enumerate(ACTIFS.items()):
        with cols[i % 2]:
            st.checkbox(nom, value=True, key=f"actif_{symbol_key}")

    st.markdown("---")
    st.markdown(f"""
    <div style='background:{BG_SEC}; padding:15px; border-radius:10px;
                border:1px solid #1a2a5e; text-align:center;'>
        <p style='color:#00ff88; font-size:1.2rem; font-weight:bold;'>
            TRADAMAR v1.0
        </p>
        <p style='color:#00aaff;'>
            Intelligence Artificielle d'Analyse Technique
        </p>
        <p style='color:#555555; font-size:0.8rem;'>
            © 2026 — Tous droits réservés
        </p>
    </div>
    """, unsafe_allow_html=True)
