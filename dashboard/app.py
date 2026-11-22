from pathlib import Path
import math
import html
import json

import joblib
import xgboost as xgb_lib

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

# ============================================================
# MAGNOVA // ULTIMATE MISSION CONTROL
# Historical geomagnetic forecasting research dashboard
# ============================================================
ROOT = Path(__file__).resolve().parent.parent

st.set_page_config(
    page_title="MAGNOVA // Geomagnetic Intelligence",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

C = {
    "bg": "#050812", "panel": "#0B1425", "panel2": "#101D32",
    "line": "rgba(142,185,220,.14)", "cyan": "#55E6FF",
    "blue": "#659BFF", "violet": "#A78BFA", "pink": "#FF6EAA",
    "green": "#69F0C1", "amber": "#FFC078", "red": "#FF657A",
    "white": "#F2F7FF", "muted": "#91A4BF",
}

# ----------------------------- CSS -----------------------------
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
:root {{ --bg:{C['bg']}; --cyan:{C['cyan']}; --violet:{C['violet']}; --text:{C['white']}; --muted:{C['muted']}; }}
html, body, [class*="css"] {{ font-family:'Manrope',sans-serif; color:var(--text); }}
.stApp {{
 background:
  radial-gradient(ellipse at 78% 5%, rgba(40,112,170,.18), transparent 31%),
  radial-gradient(ellipse at 6% 22%, rgba(106,62,170,.15), transparent 32%),
  radial-gradient(ellipse at 52% 100%, rgba(18,92,126,.10), transparent 38%),
  linear-gradient(145deg,#040711 0%,#081120 48%,#050914 100%);
 background-attachment:fixed;
}}
.stApp:before {{ content:""; position:fixed; inset:0; pointer-events:none; z-index:0; opacity:.33;
 background-image:radial-gradient(1px 1px at 8% 18%,#fff 99%,transparent),radial-gradient(1px 1px at 22% 82%,#b6edff 99%,transparent),radial-gradient(1.5px 1.5px at 41% 39%,#fff 99%,transparent),radial-gradient(1px 1px at 69% 16%,#d5c4ff 99%,transparent),radial-gradient(1.5px 1.5px at 88% 71%,#fff 99%,transparent),radial-gradient(1px 1px at 56% 91%,#8beaff 99%,transparent);
 background-size:390px 330px; animation:stars 95s linear infinite; }}
@keyframes stars {{ from{{transform:translateY(0)}} to{{transform:translateY(-330px)}} }}
@keyframes pulse {{ 0%{{opacity:.38;transform:scale(.94)}} 100%{{opacity:.9;transform:scale(1.06)}} }}
@keyframes sweep {{ 0%{{transform:translateX(-120%) skewX(-18deg);opacity:0}} 15%{{opacity:.6}} 45%{{opacity:0}} 100%{{transform:translateX(350%) skewX(-18deg);opacity:0}} }}
@keyframes blink {{ 0%,100%{{opacity:.4}} 50%{{opacity:1}} }}
@keyframes floaty {{ 0%,100%{{transform:translateY(0)}} 50%{{transform:translateY(-7px)}} }}
@keyframes scan {{ from{{background-position:0 0}} to{{background-position:0 32px}} }}
.block-container {{ max-width:1560px; padding-top:1.35rem; padding-bottom:3rem; position:relative; z-index:1; }}
[data-testid="stHeader"] {{ background:rgba(4,8,18,.72); backdrop-filter:blur(18px); }}
[data-testid="stSidebar"] {{ background:linear-gradient(180deg,rgba(8,17,32,.98),rgba(5,10,22,.99)); border-right:1px solid rgba(85,230,255,.13); }}
[data-testid="stSidebar"] > div {{ padding-top:1.5rem; }}
[data-testid="stSidebar"] * {{ color:#DCE8F8; }}
[data-testid="stSidebar"] [data-testid="stRadio"] label {{ font-size:.9rem; }}
[data-testid="stSidebar"] [data-testid="stRadio"] > div {{ gap:.32rem; }}
h1,h2,h3,h4 {{ color:#F5F8FF; letter-spacing:-.045em; }}
p {{ color:#C4D1E5; }}
hr {{ border-color:rgba(142,185,220,.13); }}
/* top rail */
.toprail {{ display:flex; align-items:center; justify-content:space-between; gap:1rem; margin-bottom:1.05rem; color:#8EA5C2; font:500 .68rem 'DM Mono',monospace; letter-spacing:.13em; text-transform:uppercase; }}
.rail-left {{ display:flex; align-items:center; gap:.65rem; }}
.live-dot {{ width:7px;height:7px;border-radius:50%;background:#69F0C1;box-shadow:0 0 12px #69F0C1;display:inline-block;animation:blink 1.7s infinite; }}
.rail-tag {{ border:1px solid rgba(85,230,255,.25);background:rgba(85,230,255,.06);padding:.36rem .62rem;border-radius:999px;color:#A8F3FF; }}
/* hero */
.hero-wrap {{ position:relative; overflow:hidden; border-radius:26px; border:1px solid rgba(116,190,233,.23);
 background:linear-gradient(112deg,rgba(12,25,45,.97) 0%,rgba(12,25,45,.91) 48%,rgba(22,24,57,.89) 100%);
 box-shadow:0 22px 75px rgba(0,0,0,.34),inset 0 1px rgba(255,255,255,.035); margin-bottom:1.2rem; }}
.hero-wrap:after {{ content:""; position:absolute; inset:0 auto 0 -30%; width:24%; background:linear-gradient(90deg,transparent,rgba(110,225,255,.08),transparent); transform:skewX(-18deg); animation:sweep 10s ease-in-out infinite; pointer-events:none; }}
.hero-inner {{ position:relative;z-index:1;min-height:330px;padding:1.9rem 2rem 1.65rem;display:flex;flex-direction:column;justify-content:center; }}
.kicker {{ color:#71E8FF;font:500 .67rem 'DM Mono',monospace;letter-spacing:.2em;text-transform:uppercase;margin-bottom:.8rem; }}
.hero-title {{ font-size:clamp(2.5rem,5vw,4.9rem);font-weight:800;line-height:.96;letter-spacing:-.075em;margin:0;
 background:linear-gradient(100deg,#FFFFFF 2%,#9BEFFF 48%,#BBA6FF 90%);-webkit-background-clip:text;-webkit-text-fill-color:transparent; }}
.hero-copy {{ color:#A8BAD3;line-height:1.72;font-size:.96rem;max-width:690px;margin-top:1rem; }}
.chips {{display:flex;flex-wrap:wrap;gap:.48rem;margin-top:1.25rem}}
.chip {{font:500 .64rem 'DM Mono',monospace;letter-spacing:.07em;color:#BDEFFF;border:1px solid rgba(85,230,255,.22);background:rgba(85,230,255,.055);border-radius:999px;padding:.39rem .68rem}}
/* headings/cards */
.section-kicker {{font:500 .66rem 'DM Mono',monospace;letter-spacing:.18em;color:#68DFF5;text-transform:uppercase;margin:.3rem 0 .36rem}}
.section-heading {{font-size:1.36rem;font-weight:800;letter-spacing:-.05em;color:#F3F7FF;margin:0 0 .25rem}}
.section-sub {{font-size:.82rem;color:#8FA4C0;line-height:1.5;margin-bottom:.85rem}}
.metric {{position:relative;overflow:hidden;min-height:132px;padding:1.05rem 1.05rem .95rem;border:1px solid rgba(133,174,211,.16);border-radius:17px;
 background:linear-gradient(145deg,rgba(19,36,59,.86),rgba(9,18,34,.82));box-shadow:0 12px 34px rgba(0,0,0,.16);transition:transform .22s,border-color .22s,box-shadow .22s;}}
.metric:hover {{transform:translateY(-4px);border-color:rgba(85,230,255,.43);box-shadow:0 16px 42px rgba(0,0,0,.27),0 0 25px rgba(85,230,255,.07)}}
.metric:after {{content:"";position:absolute;right:-27px;top:-42px;width:110px;height:110px;border-radius:50%;background:radial-gradient(circle,rgba(85,230,255,.13),transparent 68%);}}
.metric-label {{font:500 .64rem 'DM Mono',monospace;letter-spacing:.11em;color:#8FA8C7;text-transform:uppercase}}
.metric-value {{font-size:1.9rem;font-weight:800;letter-spacing:-.055em;color:#F4F8FF;margin:.52rem 0 .15rem;line-height:1.05}}
.metric-note {{font-size:.72rem;color:#8FA3BE;line-height:1.45}}
.panel {{border:1px solid rgba(133,174,211,.15);border-radius:19px;background:linear-gradient(145deg,rgba(14,27,47,.87),rgba(8,16,31,.86));padding:1rem 1.05rem;box-shadow:0 14px 38px rgba(0,0,0,.15)}}
.signal {{border:1px solid rgba(133,174,211,.13);border-left:3px solid #55E6FF;border-radius:0 13px 13px 0;background:rgba(85,230,255,.045);padding:.8rem .9rem;margin:.55rem 0;color:#BFCFE3;font-size:.8rem;line-height:1.55}}
.signal.violet {{border-left-color:#A78BFA;background:rgba(167,139,250,.045)}}
.signal.pink {{border-left-color:#FF6EAA;background:rgba(255,110,170,.045)}}
.signal strong {{color:#F1F7FF}}
.status {{display:inline-flex;align-items:center;gap:.42rem;border:1px solid rgba(255,192,120,.28);background:rgba(255,192,120,.07);color:#FFD19D;border-radius:999px;padding:.38rem .66rem;font:500 .65rem 'DM Mono',monospace;letter-spacing:.04em}}
.micro {{font:500 .64rem 'DM Mono',monospace;letter-spacing:.12em;color:#7891B0;text-transform:uppercase}}
.data-note {{color:#7E93AE;font-size:.72rem;line-height:1.55}}
.footer {{text-align:center;color:#647B98;font:500 .63rem 'DM Mono',monospace;letter-spacing:.12em;padding:2rem 0 .4rem}}
/* Streamlit controls */
div[data-testid="stButton"] button {{background:linear-gradient(110deg,rgba(25,94,124,.44),rgba(91,63,146,.38));border:1px solid rgba(85,230,255,.3);border-radius:11px;color:#E8FAFF;transition:all .2s}}
div[data-testid="stButton"] button:hover {{border-color:#55E6FF;box-shadow:0 0 18px rgba(85,230,255,.14);transform:translateY(-2px)}}
div[data-testid="stSelectbox"] label,div[data-testid="stSlider"] label,div[data-testid="stMultiSelect"] label {{color:#B5C9E1;font-size:.8rem}}
[data-testid="stDataFrame"] {{border:1px solid rgba(133,174,211,.14);border-radius:13px;overflow:hidden}}
[data-testid="stTabs"] button {{color:#9CB0CA}}
[data-testid="stTabs"] button[aria-selected="true"] {{color:#85EFFF}}
.stAlert {{border-radius:13px}}
@media(max-width:800px) {{.hero-inner{{padding:1.45rem}}.toprail{{font-size:.56rem}}.metric-value{{font-size:1.5rem}}}}
@media(prefers-reduced-motion:reduce) {{*,*:before,*:after{{animation-duration:.01ms!important;transition-duration:.01ms!important}}}}
</style>
""", unsafe_allow_html=True)

# ----------------------------- Cursor FX -----------------------------
# Inject the canvas into the top-level Streamlit document from a tiny
# components.html iframe. This avoids relying on script execution in st.html.
CURSOR_FX = r"""
<script>
(() => {
  try {
    const hostWindow = window.parent;
    const host = hostWindow.document;
    if (host.getElementById('magnova-cursor-layer')) return;
    if (hostWindow.matchMedia('(pointer: coarse)').matches ||
        hostWindow.matchMedia('(prefers-reduced-motion: reduce)').matches) return;

    const style = host.createElement('style');
    style.id = 'magnova-cursor-style';
    style.textContent = `
      #magnova-cursor-layer { position:fixed; inset:0; z-index:999999; pointer-events:none; overflow:hidden; }
      #magnova-cursor-layer canvas { width:100%; height:100%; display:block; }
    `;
    host.head.appendChild(style);

    const root = host.createElement('div');
    root.id = 'magnova-cursor-layer';
    root.setAttribute('aria-hidden', 'true');
    const canvas = host.createElement('canvas');
    root.appendChild(canvas);
    host.body.appendChild(root);

    const ctx = canvas.getContext('2d');
    let width = 0, height = 0, dpr = 1, particles = [];
    const pointer = { x: -100, y: -100, active: false };
    let lastSpawn = 0;

    const resize = () => {
      width = hostWindow.innerWidth;
      height = hostWindow.innerHeight;
      dpr = Math.min(hostWindow.devicePixelRatio || 1, 2);
      canvas.width = width * dpr;
      canvas.height = height * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    resize();
    hostWindow.addEventListener('resize', resize, { passive: true });
    host.addEventListener('pointermove', event => {
      pointer.x = event.clientX;
      pointer.y = event.clientY;
      pointer.active = true;
      const now = performance.now();
      if (now - lastSpawn > 12) {
        lastSpawn = now;
        for (let i = 0; i < 2; i++) {
          const angle = Math.random() * Math.PI * 2;
          const speed = Math.random() * 1.5 + 0.2;
          particles.push({
            x: pointer.x, y: pointer.y,
            vx: Math.cos(angle) * speed, vy: Math.sin(angle) * speed,
            life: 1, size: Math.random() * 2.3 + 0.6,
            hue: Math.random() > 0.22 ? 190 : 265
          });
        }
      }
    }, { passive: true });
    host.addEventListener('pointerleave', () => { pointer.active = false; });

    const draw = () => {
      if (!host.getElementById('magnova-cursor-layer')) return;
      ctx.clearRect(0, 0, width, height);
      for (let i = particles.length - 1; i >= 0; i--) {
        const particle = particles[i];
        particle.x += particle.vx;
        particle.y += particle.vy;
        particle.vx *= 0.985;
        particle.vy *= 0.985;
        particle.life -= 0.019;
        particle.size *= 0.992;
        if (particle.life <= 0) {
          particles.splice(i, 1);
          continue;
        }
        ctx.beginPath();
        ctx.fillStyle = `hsla(${particle.hue},100%,78%,${particle.life * 0.78})`;
        ctx.shadowBlur = 13;
        ctx.shadowColor = `hsla(${particle.hue},100%,70%,${particle.life})`;
        ctx.arc(particle.x, particle.y, particle.size * particle.life, 0, Math.PI * 2);
        ctx.fill();
      }
      if (pointer.active) {
        const glow = ctx.createRadialGradient(pointer.x, pointer.y, 0, pointer.x, pointer.y, 25);
        glow.addColorStop(0, 'rgba(150,245,255,.25)');
        glow.addColorStop(1, 'rgba(85,230,255,0)');
        ctx.fillStyle = glow;
        ctx.beginPath();
        ctx.arc(pointer.x, pointer.y, 25, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.shadowBlur = 0;
      hostWindow.requestAnimationFrame(draw);
    };
    draw();
  } catch (error) {
    console.error('MAGNOVA cursor installation failed:', error);
  }
})();
</script>
"""
components.html(CURSOR_FX, height=1, width=1, scrolling=False)

# ----------------------------- Data -----------------------------
@st.cache_data(show_spinner=False)
def load_csv(relative_path: str) -> pd.DataFrame:
    path = ROOT / relative_path
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_image(relative_path: str):
    path = ROOT / relative_path
    return str(path) if path.exists() else None

results = load_csv("reports/final_test_results.csv")
predictions = load_csv("reports/final_test_predictions.csv")
period_results = load_csv("reports/test_analysis/period_wise_results.csv")
storm_results = load_csv("reports/test_analysis/storm_analysis.csv")
onset_results = load_csv("reports/test_analysis/storm_onset_analysis.csv")
magnitude_results = load_csv("reports/test_period_analysis/error_by_dst_magnitude.csv")
dst_distribution = load_csv("reports/period_analysis/dst_distribution.csv")
missing_data = load_csv("reports/period_analysis/missing_data_comparison.csv")

# ----------------------------- Helpers -----------------------------
def fmt(v, digits=2):
    try:
        if pd.isna(v): return "—"
        return f"{float(v):,.{digits}f}"
    except (TypeError, ValueError):
        return "—"

def section(kicker, title, subtitle=None):
    st.markdown(f'<div class="section-kicker">{kicker}</div><div class="section-heading">{title}</div>' + (f'<div class="section-sub">{subtitle}</div>' if subtitle else ''), unsafe_allow_html=True)

def metric(label, value, note, color="cyan"):
    st.markdown(f'<div class="metric"><div class="metric-label">{label}</div><div class="metric-value" style="text-shadow:0 0 22px {C.get(color,C["cyan"])}30">{value}</div><div class="metric-note">{note}</div></div>', unsafe_allow_html=True)

def show_image(relative_path, caption=None):
    path = load_image(relative_path)
    if path:
        st.image(path, caption=caption, use_container_width=True)
    else:
        st.caption(f"Optional plot not found: {relative_path}")

def chart_theme(fig, height=390, title=None):
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(7,14,27,.42)",
        font=dict(family="Manrope", color="#DCE9F8", size=11), height=height,
        margin=dict(l=12,r=12,t=48 if title else 20,b=12),
        title=dict(text=title, x=.02, y=.98, font=dict(size=14,color="#EAF4FF")) if title else None,
        legend=dict(bgcolor="rgba(8,16,30,.65)",bordercolor="rgba(142,185,220,.12)",borderwidth=1),
        hoverlabel=dict(bgcolor="#101D32",bordercolor="#55E6FF",font=dict(color="#F2F7FF")),
    )
    fig.update_xaxes(gridcolor="rgba(140,174,210,.09)",zerolinecolor="rgba(140,174,210,.13)",linecolor="rgba(140,174,210,.12)")
    fig.update_yaxes(gridcolor="rgba(140,174,210,.09)",zerolinecolor="rgba(140,174,210,.13)",linecolor="rgba(140,174,210,.12)")
    return fig

def plot(fig, height=390):
    st.plotly_chart(chart_theme(fig,height),use_container_width=True,config={"displayModeBar":False,"scrollZoom":True})

def earth_visual():
    # Self-contained animated SVG scene: no external JS execution required.
    markup = r'''<!doctype html><html><head><meta charset="utf-8"><style>
    *{box-sizing:border-box}body{margin:0;background:transparent;overflow:hidden;font-family:Arial,sans-serif}
    .scene{height:330px;width:100%;position:relative;display:grid;place-items:center;overflow:hidden;border:1px solid rgba(116,190,233,.23);border-radius:26px;background:linear-gradient(145deg,rgba(12,25,45,.97),rgba(18,22,49,.94));box-shadow:0 22px 75px rgba(0,0,0,.34),inset 0 1px rgba(255,255,255,.035)}
    .halo{position:absolute;width:280px;height:280px;border-radius:50%;background:radial-gradient(circle,rgba(47,174,255,.19),rgba(83,90,255,.07) 48%,transparent 72%);filter:blur(12px);animation:breathe 5s ease-in-out infinite alternate}
    svg{width:min(100%,390px);height:320px;overflow:visible;position:relative;z-index:2}
    .orbit{fill:none;stroke:rgba(121,214,255,.22);stroke-width:1;stroke-dasharray:3 8;transform-origin:190px 160px;animation:spin 28s linear infinite}
    .orbit2{fill:none;stroke:rgba(176,145,255,.25);stroke-width:1.2;stroke-dasharray:2 7;transform-origin:190px 160px;animation:spin 20s linear infinite reverse}
    .field{fill:none;stroke:url(#field);stroke-width:1.7;opacity:.7;stroke-dasharray:5 8;animation:flow 8s linear infinite}
    .particle{fill:#9df3ff;filter:url(#glow);animation:blink 2s ease-in-out infinite alternate}
    .earth{filter:url(#earthGlow)}
    .gridline{fill:none;stroke:rgba(132,222,255,.21);stroke-width:.8}
    .shine{fill:url(#shine)}
    .label{position:absolute;z-index:3;color:#86A9C9;font:10px monospace;letter-spacing:2px;text-transform:uppercase}
    .l1{top:17px;left:12px}.l2{right:9px;bottom:28px;text-align:right}.cross{position:absolute;width:7px;height:7px;border:1px solid rgba(115,225,255,.7);transform:rotate(45deg);}
    @keyframes spin{to{transform:rotate(360deg)}}@keyframes flow{to{stroke-dashoffset:-130}}@keyframes breathe{to{transform:scale(1.12);opacity:.75}}@keyframes blink{to{opacity:.25;transform:scale(.7)}}
    </style></head><body><div class="scene"><div class="halo"></div><div class="label l1">MAGNETOSPHERE // SIMULATED VISUAL</div><div class="label l2">SOLAR WIND → EARTH<br>Dst FIELD STUDY</div>
    <svg viewBox="0 0 380 320" role="img" aria-label="Stylized animated Earth and magnetosphere illustration">
    <defs>
    <radialGradient id="earth" cx="32%" cy="25%"><stop offset="0" stop-color="#8EEBFF"/><stop offset=".36" stop-color="#267EBB"/><stop offset=".72" stop-color="#12406F"/><stop offset="1" stop-color="#071B38"/></radialGradient>
    <linearGradient id="field" x1="0" y1="0" x2="1" y2="1"><stop stop-color="#6EEBFF"/><stop offset=".5" stop-color="#9A9BFF"/><stop offset="1" stop-color="#5BD8F1"/></linearGradient>
    <radialGradient id="shine"><stop stop-color="white" stop-opacity=".3"/><stop offset="1" stop-color="white" stop-opacity="0"/></radialGradient>
    <filter id="earthGlow"><feGaussianBlur stdDeviation="8" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
    <filter id="glow"><feGaussianBlur stdDeviation="3" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
    <clipPath id="clip"><circle cx="190" cy="160" r="69"/></clipPath>
    </defs>
    <ellipse class="orbit" cx="190" cy="160" rx="154" ry="93"/><ellipse class="orbit2" cx="190" cy="160" rx="129" ry="116"/>
    <path class="field" d="M190 91 C65 20 21 107 35 160 C21 213 65 300 190 229"/><path class="field" d="M190 91 C315 20 359 107 345 160 C359 213 315 300 190 229"/>
    <path class="field" d="M190 96 C103 66 84 119 87 160 C84 201 103 254 190 224"/><path class="field" d="M190 96 C277 66 296 119 293 160 C296 201 277 254 190 224"/>
    <g class="earth"><circle cx="190" cy="160" r="71" fill="none" stroke="#6CE6FF" stroke-opacity=".26" stroke-width="4"/><circle cx="190" cy="160" r="69" fill="url(#earth)"/>
    <g clip-path="url(#clip)"><path d="M127 130l21-24 17 7 8 18-14 10-7 22-17-4-10-16zM178 100l17 7 6 17-10 12-19-8-4-16zM202 143l22-15 22 11 9 23-11 12-7 25-18 8-7-24-20-12zM146 188l20-9 12 14-4 25-18 13-13-18zM233 208l14-9 12 16-14 17-13-7z" fill="#69C9A9" opacity=".65"/><path class="gridline" d="M120 160h140M190 90v140M132 125q58 40 116 0M132 195q58-40 116 0"/><ellipse cx="165" cy="132" rx="57" ry="35" class="shine"/></g></g>
    <circle class="particle" cx="48" cy="113" r="3"/><circle class="particle" cx="322" cy="214" r="2.5" style="animation-delay:.7s"/><circle class="particle" cx="273" cy="51" r="2" style="animation-delay:1.2s"/><circle class="particle" cx="103" cy="257" r="2.3" style="animation-delay:.3s"/>
    <path d="M22 160h18M340 160h18M190 11v17M190 292v17" stroke="rgba(108,230,255,.35)" stroke-width="1"/>
    <circle cx="190" cy="160" r="3" fill="#D6FAFF" filter="url(#glow)"/>
    </svg><div class="cross" style="top:70px;right:45px"></div><div class="cross" style="bottom:73px;left:36px"></div></div></body></html>'''
    components.html(markup, height=330, scrolling=False)

def make_error_comparison(frame, metric_key="rmse"):
    fig=go.Figure()
    fig.add_trace(go.Bar(name="XGBoost",x=frame["target"].map({"dst_t0":"T+1h","dst_t1":"T+2h"}).fillna(frame["target"].str.upper()),y=frame[f"xgb_{metric_key}"],marker=dict(color="#55E6FF",line=dict(color="#B5F6FF",width=1)),text=frame[f"xgb_{metric_key}"].map(fmt),textposition="outside"))
    fig.add_trace(go.Bar(name="Persistence",x=frame["target"].map({"dst_t0":"T+1h","dst_t1":"T+2h"}).fillna(frame["target"].str.upper()),y=frame[f"persistence_{metric_key}"],marker=dict(color="#A78BFA",line=dict(color="#D4C8FF",width=1)),text=frame[f"persistence_{metric_key}"].map(fmt),textposition="outside"))
    fig.update_layout(barmode="group",yaxis_title=f"{metric_key.upper()} (nT)")
    return fig


# ----------------------------- Scientific evaluation helpers -----------------------------
def dst_alert_band(value):
    """Explanatory Dst severity bands used by this research dashboard."""
    if pd.isna(value):
        return "Unavailable"
    value = float(value)
    if value <= -200:
        return "Severe storm"
    if value <= -100:
        return "Intense storm"
    if value <= -50:
        return "Moderate storm"
    return "Quiet / below storm threshold"

def subset_skill_table(prediction_frame):
    """Compute matched XGBoost-vs-persistence scores for storm and quiet targets."""
    required = {
        "dst_t0", "dst_t1",
        "dst_t0_xgb_prediction", "dst_t0_persistence_prediction",
        "dst_t1_xgb_prediction", "dst_t1_persistence_prediction",
    }
    if prediction_frame.empty or not required.issubset(prediction_frame.columns):
        return pd.DataFrame()
    rows = []
    for target in ["dst_t0", "dst_t1"]:
        xgb_col = f"{target}_xgb_prediction"
        persistence_col = f"{target}_persistence_prediction"
        valid = prediction_frame[[target, xgb_col, persistence_col]].apply(pd.to_numeric, errors="coerce").dropna()
        for subset_name, mask in [("Storm (actual Dst ≤ -50 nT)", valid[target] <= -50),
                                  ("Quiet (actual Dst > -50 nT)", valid[target] > -50)]:
            part = valid.loc[mask]
            if part.empty:
                continue
            actual = part[target].to_numpy(dtype=float)
            xgb_pred = part[xgb_col].to_numpy(dtype=float)
            persistence_pred = part[persistence_col].to_numpy(dtype=float)
            xgb_rmse = float(((actual - xgb_pred) ** 2).mean() ** 0.5)
            persistence_rmse = float(((actual - persistence_pred) ** 2).mean() ** 0.5)
            xgb_mae = float(abs(actual - xgb_pred).mean())
            persistence_mae = float(abs(actual - persistence_pred).mean())
            skill = (1.0 - xgb_rmse / persistence_rmse) if persistence_rmse > 0 else float("nan")
            rows.append({
                "horizon": "T+1 hour" if target == "dst_t0" else "T+2 hours",
                "subset": subset_name,
                "samples": len(part),
                "xgb_rmse_nT": xgb_rmse,
                "persistence_rmse_nT": persistence_rmse,
                "xgb_mae_nT": xgb_mae,
                "persistence_mae_nT": persistence_mae,
                "xgb_skill_vs_persistence": skill,
            })
    return pd.DataFrame(rows)

# ----------------------------- Offline AI inference -----------------------------
RAW_SOLAR_FEATURES = [
    f"{name}_{stat}"
    for name in ["bx_gsm", "by_gsm", "bz_gsm", "bt", "density", "speed", "temperature"]
    for stat in ["mean", "min", "max", "std"]
]
TEMPORAL_BASE_FEATURES = ["bz_gsm_mean", "bz_gsm_min", "speed_mean", "density_mean", "bt_mean"]

@st.cache_resource(show_spinner=False)
def load_inference_assets():
    """Load the exact final models, metadata and temporal training imputer."""
    metadata_path = ROOT / "models" / "final" / "final_model_metadata.json"
    imputer_path = ROOT / "models" / "temporal_median_imputer.pkl"
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    imputer = joblib.load(imputer_path)
    models = {}
    for target in ["dst_t0", "dst_t1"]:
        model_path = ROOT / metadata["models"][target]["model_path"]
        model = xgb_lib.XGBRegressor()
        model.load_model(str(model_path))
        models[target] = model
    return metadata, imputer, models

def _sort_hourly_rows(frame):
    """Sort a period chronologically without lexicographically sorting Timedelta labels."""
    frame = frame.copy()
    if "hour" not in frame.columns:
        return frame

    raw_hour = frame["hour"]

    # Numeric hour counters (including numeric strings) take precedence.
    numeric_hour = pd.to_numeric(raw_hour, errors="coerce")
    if numeric_hour.notna().all():
        frame["_hour_sort"] = numeric_hour
    else:
        # The competition data stores elapsed time as strings such as
        # "999 days 23:00:00"; parse these as Timedelta before trying timestamps.
        elapsed_hour = pd.to_timedelta(raw_hour, errors="coerce")
        if elapsed_hour.notna().all():
            frame["_hour_sort"] = elapsed_hour
        else:
            timestamp_hour = pd.to_datetime(raw_hour, errors="coerce", utc=True)
            if timestamp_hour.notna().all():
                frame["_hour_sort"] = timestamp_hour
            else:
                bad_count = int(timestamp_hour.isna().sum())
                raise ValueError(
                    f"Could not interpret {bad_count} value(s) in the 'hour' column. "
                    "Use numeric hour indices, elapsed-hour values, or valid timestamps."
                )

    return frame.sort_values("_hour_sort", kind="stable").drop(columns="_hour_sort")


def _validate_hourly_continuity(frame, period):
    """Reject duplicate or missing hourly observations before lag/rolling features."""
    if "hour" not in frame.columns or len(frame) < 2:
        return

    raw_hour = frame["hour"]
    numeric_hour = pd.to_numeric(raw_hour, errors="coerce")
    if numeric_hour.notna().all():
        ordered = numeric_hour
        expected_step = 1
    else:
        elapsed_hour = pd.to_timedelta(raw_hour, errors="coerce")
        if elapsed_hour.notna().all():
            ordered = elapsed_hour
            expected_step = pd.Timedelta(hours=1)
        else:
            timestamp_hour = pd.to_datetime(raw_hour, errors="coerce", utc=True)
            if not timestamp_hour.notna().all():
                raise ValueError(
                    f"Period {period!r} contains unparseable 'hour' values; "
                    "cannot verify hourly continuity."
                )
            ordered = timestamp_hour
            expected_step = pd.Timedelta(hours=1)

    differences = ordered.diff().iloc[1:]
    duplicate_count = int((differences == 0).sum())
    gap_count = int((differences != expected_step).sum()) - duplicate_count
    if duplicate_count or gap_count:
        raise ValueError(
            f"Period {period!r} is not a continuous hourly sequence "
            f"({duplicate_count} duplicate step(s), {gap_count} missing/irregular step(s)). "
            "Provide one observation per consecutive hour; lag and rolling features "
            "are only valid for evenly spaced hourly data."
        )

def engineer_inference_features(uploaded_frame):
    """Reproduce build_temporal_features.py for uploaded hourly aggregates."""
    missing = [c for c in RAW_SOLAR_FEATURES if c not in uploaded_frame.columns]
    if missing:
        raise ValueError("Missing required hourly aggregate columns: " + ", ".join(missing))

    data = uploaded_frame.copy()
    for col in RAW_SOLAR_FEATURES:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    if "period" not in data.columns:
        data["period"] = "uploaded_period"
    data["_upload_order"] = range(len(data))
    engineered_groups = []

    for period, group in data.groupby("period", sort=False, dropna=False):
        group = _sort_hourly_rows(group).copy()
        _validate_hourly_continuity(group, period)
        raw = group[RAW_SOLAR_FEATURES].copy()

        # Missingness indicators are created before imputation, as in training.
        group["missing_feature_count"] = raw.isna().sum(axis=1)
        group["missing_feature_fraction"] = raw.isna().mean(axis=1)
        for feature in TEMPORAL_BASE_FEATURES:
            group[f"{feature}_missing"] = raw[feature].isna().astype(int)

        for feature in TEMPORAL_BASE_FEATURES:
            for lag in range(1, 6):
                group[f"{feature}_lag_{lag}"] = raw[feature].shift(lag)

        for feature in TEMPORAL_BASE_FEATURES:
            series = raw[feature]
            for window in [6, 12, 24]:
                group[f"{feature}_rolling_mean_{window}h"] = series.rolling(window, min_periods=1).mean()
                group[f"{feature}_rolling_min_{window}h"] = series.rolling(window, min_periods=1).min()

        bz_south = (-raw["bz_gsm_mean"]).clip(lower=0)
        group["southward_bz"] = bz_south
        group["vbs_proxy"] = raw["speed_mean"] * bz_south
        group["dynamic_pressure_proxy"] = raw["density_mean"] * raw["speed_mean"] ** 2
        group["_upload_order"] = group["_upload_order"]
        engineered_groups.append(group)

    engineered = pd.concat(engineered_groups, axis=0).sort_values("_upload_order")
    engineered = engineered.drop(columns=["_upload_order"], errors="ignore")
    return engineered

def format_forecast_origin_hour(value):
    """Render elapsed-hour labels clearly without implying they are UTC timestamps."""
    if pd.isna(value):
        return "Unknown"

    # Competition data uses elapsed-time labels such as "1200 days 23:00:00".
    elapsed = pd.to_timedelta(value, errors="coerce")
    if not pd.isna(elapsed):
        total_hours = int(elapsed.total_seconds() // 3600)
        day, hour = divmod(total_hours, 24)
        return f"Day {day}, {hour:02d}:00"

    # Keep numeric hour counters explicit as elapsed hours.
    numeric = pd.to_numeric(value, errors="coerce")
    if not pd.isna(numeric):
        total_hours = int(numeric)
        day, hour = divmod(total_hours, 24)
        return f"Day {day}, {hour:02d}:00"

    # For genuine timestamps, retain the calendar date/time and timezone if present.
    timestamp = pd.to_datetime(value, errors="coerce", utc=True)
    if not pd.isna(timestamp):
        return timestamp.strftime("%Y-%m-%d %H:%M UTC")

    return str(value)


def run_uploaded_inference(uploaded_frame):
    metadata, imputer, models = load_inference_assets()
    engineered = engineer_inference_features(uploaded_frame)

    # One forecast origin per uploaded period: the latest complete row.
    origins = []
    for period, group in engineered.groupby("period", sort=False, dropna=False):
        group = _sort_hourly_rows(group)
        if len(group) < 24:
            raise ValueError(
                f"Period {period!r} has only {len(group)} hourly rows; "
                "at least 24 consecutive hourly rows are required for the model's history window."
            )
        origins.append(group.iloc[[-1]].copy())
    origin_frame = pd.concat(origins, ignore_index=True)

    imputer_features = list(getattr(imputer, "feature_names_in_", []))
    if not imputer_features:
        raise ValueError("The saved temporal imputer does not expose its training feature names.")
    absent = [c for c in imputer_features if c not in origin_frame.columns]
    if absent:
        raise ValueError("Feature engineering did not create expected imputer columns: " + ", ".join(absent))

    # Match the imputer's exact 93-column training order, then select the final
    # model's exact 83 features in metadata order.
    X_93 = origin_frame.reindex(columns=imputer_features)
    X_imputed = pd.DataFrame(
        imputer.transform(X_93),
        columns=imputer_features,
        index=origin_frame.index,
    )
    final_features = metadata["features"]
    X_final = X_imputed.reindex(columns=final_features)
    if X_final.isna().any().any():
        raise ValueError("NaN values remain after applying the saved temporal imputer.")

    predictions_out = origin_frame[["period"]].reset_index(drop=True)
    if "hour" in origin_frame.columns:
        predictions_out["forecast_origin_hour"] = origin_frame["hour"].map(format_forecast_origin_hour).reset_index(drop=True)
    predictions_out["dst_t0_prediction_nT"] = models["dst_t0"].predict(X_final)
    predictions_out["dst_t1_prediction_nT"] = models["dst_t1"].predict(X_final)
    predictions_out["dst_t0_alert_band"] = predictions_out["dst_t0_prediction_nT"].map(dst_alert_band)
    predictions_out["dst_t1_alert_band"] = predictions_out["dst_t1_prediction_nT"].map(dst_alert_band)
    return predictions_out, len(imputer_features), len(final_features)

# ----------------------------- Sidebar -----------------------------
with st.sidebar:
    st.markdown("""<div style='padding:.35rem .2rem 1.1rem'>
      <div style='font-size:1.5rem;font-weight:800;letter-spacing:-.07em;color:#F1F8FF'><span style='color:#55E6FF'>✦</span> MAGNOVA</div>
      <div style='font:500 .62rem DM Mono,monospace;letter-spacing:.18em;color:#7795B6;margin-top:.24rem'>GEOMAGNETIC INTELLIGENCE</div>
      <div style='height:1px;background:linear-gradient(90deg,rgba(85,230,255,.5),transparent);margin-top:1rem'></div></div>""",unsafe_allow_html=True)
    page=st.radio("MISSION NAVIGATION",["Command Deck","Forecast Lab","Model Arena","Storm Observatory","Research Vault","Mission Dossier"],label_visibility="visible")
    st.markdown("<div style='height:1px;background:rgba(142,185,220,.13);margin:1rem 0'></div>",unsafe_allow_html=True)
    st.markdown('<div class="micro">SYSTEM READOUT</div>',unsafe_allow_html=True)
    st.markdown('<div style="margin:.65rem 0"><span class="live-dot"></span> <span style="font:500 .68rem DM Mono;color:#A7F3D0;letter-spacing:.08em">ARCHIVE MODE / READY</span></div>',unsafe_allow_html=True)
    st.caption("TEAM G6 · INTRODUCTION TO AI")
    st.caption("Kirat Singh Walia  /  Amrit")
    st.markdown("<div style='height:1px;background:rgba(142,185,220,.13);margin:.8rem 0'></div>",unsafe_allow_html=True)
    st.markdown('<div class="data-note">This interface reads saved historical experiment outputs. It is not connected to live solar-wind feeds.</div>',unsafe_allow_html=True)

# ----------------------------- Top rail + hero -----------------------------
st.markdown('''<div class="toprail"><div class="rail-left"><span class="live-dot"></span> <span>G6 SPACE WEATHER RESEARCH MISSION</span></div><div class="rail-tag">HISTORICAL DATA // OFFLINE ANALYSIS</div></div>''',unsafe_allow_html=True)
left,right=st.columns([1.32,.88],gap="small")
with left:
    st.markdown('''<div class="hero-wrap"><div class="hero-inner">
      <div class="kicker">MISSION CONTROL &nbsp; / &nbsp; SPACE WEATHER INTELLIGENCE</div>
      <div class="hero-title">MAGNOVA<span style="color:#55E6FF">.</span></div>
      <div class="hero-copy">Reading the solar wind. Tracking geomagnetic response. Testing whether machine learning can anticipate disturbance in Earth’s magnetic environment.</div>
      <div class="chips"><span class="chip">XGBOOST FORECASTING</span><span class="chip">DST · T+1H / T+2H</span><span class="chip">SOLAR-WIND FEATURES</span><span class="chip">HELD-OUT EVALUATION</span></div>
    </div></div>''',unsafe_allow_html=True)
with right:
    earth_visual()

# ----------------------------- Command Deck -----------------------------
if page=="Command Deck":
    section("01 / COMMAND DECK","Mission overview","Final held-out test readout · Historical experiment results")
    if results.empty or not {"target","xgb_rmse","xgb_mae","persistence_rmse","persistence_mae","test_samples"}.issubset(results.columns):
        st.error("Could not load a valid reports/final_test_results.csv. Keep the dashboard inside the project’s dashboard folder so ROOT points to the project root.")
        st.stop()
    t0=results[results.target=="dst_t0"].iloc[0]; t1=results[results.target=="dst_t1"].iloc[0]
    total=int(t0.get("test_samples",0))
    a,b,c,d=st.columns(4)
    with a: metric("HELD-OUT OBSERVATIONS",f"{total:,}","Hourly test samples")
    with b: metric("FORECAST HORIZONS","T+1h / T+2h","Next-hour and two-hour targets","violet")
    with c: metric("BEST T+1H RMSE",f"{fmt(t0.persistence_rmse)} nT","Persistence benchmark leads","violet")
    with d: metric("XGBOOST T+1H RMSE",f"{fmt(t0.xgb_rmse)} nT","Solar-wind-only model","pink")
    st.markdown("<div style='height:1rem'></div>",unsafe_allow_html=True)
    lc,rc=st.columns([1.2,.8],gap="large")
    with lc:
        section("01A / MODEL TELEMETRY","Prediction error scoreboard","Lower is better · evaluated on held-out test rows")
        plot(make_error_comparison(results,"rmse"),380)
    with rc:
        section("01B / RESEARCH SIGNAL","What the experiment tells us")
        st.markdown('<div class="signal"><strong>Persistence is the overall winner.</strong><br>The simple benchmark achieved lower RMSE and MAE than XGBoost at both forecast horizons.</div><div class="signal violet"><strong>Storm conditions remain difficult.</strong><br>XGBoost errors rise sharply for strong negative Dst values.</div><div class="signal pink"><strong>Storm onset needs caution.</strong><br>Some onset RMSE comparisons favour XGBoost, but the subset is small and does not establish reliable early warning.</div>',unsafe_allow_html=True)
    st.markdown("<div style='height:.8rem'></div>",unsafe_allow_html=True)
    section("01C / MODEL VISION","Actual vs predicted Dst","Saved test-set visualizations · not a live forecast")
    p1,p2=st.columns(2)
    with p1: show_image("reports/test_analysis/plots/dst_t0_actual_vs_predicted.png","T+1h · actual vs predicted")
    with p2: show_image("reports/test_analysis/plots/dst_t1_actual_vs_predicted.png","T+2h · actual vs predicted")
    st.markdown('<div class="data-note">Interpretation: the benchmark comparison is based on the same held-out test observations. Lower error does not by itself establish operational suitability.</div>',unsafe_allow_html=True)

# ----------------------------- Forecast Lab -----------------------------
elif page=="Forecast Lab":
    section("02 / FORECAST LAB","Explore the prediction stream","Slice the historical test predictions and inspect model behaviour sample by sample.")
    historical_tab, inference_tab = st.tabs(["HISTORICAL PREDICTIONS", "AI INFERENCE LAB"])
    with historical_tab:
        if predictions.empty:
            st.error("Prediction file missing: reports/final_test_predictions.csv")
            st.stop()
        required={"period","dst_t0","dst_t1","dst_t0_xgb_prediction","dst_t0_persistence_prediction","dst_t1_xgb_prediction","dst_t1_persistence_prediction"}
        if not required.issubset(predictions.columns):
            st.error(f"Prediction CSV is missing expected columns: {', '.join(sorted(required-set(predictions.columns)))}")
            st.stop()
        c1,c2,c3=st.columns([1,1,1.1])
        with c1: period_opts=["All periods"]+sorted(predictions.period.dropna().astype(str).unique().tolist()); chosen=st.selectbox("Historical segment",period_opts)
        with c2: horizon=st.selectbox("Forecast horizon",["T+1 hour","T+2 hours"])
        with c3: max_points=st.slider("Chart resolution",100,4000,1200,100)
        df=predictions.copy()
        if chosen!="All periods": df=df[df.period.astype(str)==chosen]
        actual,xgb,pers=("dst_t0","dst_t0_xgb_prediction","dst_t0_persistence_prediction") if horizon=="T+1 hour" else ("dst_t1","dst_t1_xgb_prediction","dst_t1_persistence_prediction")
        df=df.reset_index(drop=True)
        step=max(1,math.ceil(len(df)/max_points))
        chart_df=df.iloc[::step].copy()
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=chart_df.index,y=chart_df[actual],name="ACTUAL DST",mode="lines",line=dict(color="#F3F7FF",width=1.8),hovertemplate="Sample %{x}<br>Actual %{y:.2f} nT<extra></extra>"))
        fig.add_trace(go.Scatter(x=chart_df.index,y=chart_df[xgb],name="XGBOOST",mode="lines",line=dict(color="#55E6FF",width=1.45),hovertemplate="Sample %{x}<br>XGBoost %{y:.2f} nT<extra></extra>"))
        fig.add_trace(go.Scatter(x=chart_df.index,y=chart_df[pers],name="PERSISTENCE",mode="lines",line=dict(color="#A78BFA",width=1.4,dash="dot"),hovertemplate="Sample %{x}<br>Persistence %{y:.2f} nT<extra></extra>"))
        fig.add_hrect(y0=-50,y1=50,fillcolor="rgba(105,240,193,.035)",line_width=0,annotation_text="QUIET" ,annotation_position="top left",annotation_font_color="#82CDB5")
        fig.add_hrect(y0=-100,y1=-50,fillcolor="rgba(255,192,120,.075)",line_width=0,annotation_text="MODERATE",annotation_position="top left",annotation_font_color="#D7AD79")
        fig.add_hrect(y0=-200,y1=-100,fillcolor="rgba(255,110,170,.065)",line_width=0,annotation_text="INTENSE",annotation_position="top left",annotation_font_color="#E49AB8")
        fig.add_hrect(y0=-450,y1=-200,fillcolor="rgba(255,101,122,.075)",line_width=0,annotation_text="SEVERE",annotation_position="top left",annotation_font_color="#F08B9B")
        fig.update_layout(hovermode="x unified",xaxis_title="Sample index within selected segment",yaxis_title="Dst (nT)")
        plot(fig,470)
        st.caption(f"Showing {len(chart_df):,} chart points from {len(df):,} filtered rows. The x-axis is the row order within the selected historical segment, not calendar time.")
        st.markdown("<div style='height:.7rem'></div>",unsafe_allow_html=True)
        section("SAMPLE INSPECTOR","One observation, three values","Choose a row to inspect its actual target and both model predictions.")
        if len(df):
            idx=st.slider("Select test observation",0,len(df)-1,min(len(df)-1, int(len(df)*.55)))
            row=df.iloc[idx]
            q1,q2,q3,q4=st.columns(4)
            with q1: metric("ACTUAL DST",f"{fmt(row[actual])} nT","Observed target")
            with q2: metric("XGBOOST",f"{fmt(row[xgb])} nT","Solar-wind model","cyan")
            with q3: metric("PERSISTENCE",f"{fmt(row[pers])} nT","Baseline prediction","violet")
            with q4: metric("XGB ABS ERROR",f"{fmt(abs(float(row[actual])-float(row[xgb])))} nT","For selected observation","pink")
            if "hour" in df.columns: st.caption(f"Stored hour label: {row['hour']} · Period: {row['period']}")


    with inference_tab:
        section("02B / AI INFERENCE LAB", "Run the trained forecasting models",
                "Offline inference using uploaded hourly solar-wind aggregates and the saved training preprocessing.")
        st.markdown(
            '<div class="signal"><strong>Input contract:</strong> upload at least 24 consecutive hourly rows per period, '
            'containing the 28 hourly aggregate columns used by the training pipeline. The latest row in each period '
            'becomes the forecast origin.</div>',
            unsafe_allow_html=True,
        )
        with st.expander("Required CSV columns", expanded=False):
            st.code(", ".join(RAW_SOLAR_FEATURES), language="text")
            st.caption("Optional columns: period and hour. If period is omitted, all rows are treated as one period. "
                       "Dst is not required because the XGBoost models use solar-wind features only.")
        inference_file = st.file_uploader(
            "Upload hourly solar-wind CSV",
            type=["csv"],
            key="magnova_inference_csv",
            help="Use the same hourly aggregation schema as data/processed/hourly_dataset.csv.",
        )
        if inference_file is not None:
            try:
                inference_input = pd.read_csv(inference_file)
                st.caption(f"Loaded {len(inference_input):,} rows · {len(inference_input.columns)} columns")
                with st.expander("Preview uploaded data", expanded=False):
                    st.dataframe(inference_input.head(8), use_container_width=True, hide_index=True)
                if st.button("RUN AI FORECAST", type="primary", use_container_width=True):
                    with st.spinner("Engineering temporal features and running both XGBoost models..."):
                        forecast_output, imputer_count, model_count = run_uploaded_inference(inference_input)
                    st.success("Inference completed successfully.")
                    st.caption(
                        f"Preprocessing verified: {imputer_count} temporal features → "
                        f"{model_count} final model features. One forecast origin per period."
                    )
                    st.dataframe(
                        forecast_output.style.format({
                            "dst_t0_prediction_nT": "{:.3f}",
                            "dst_t1_prediction_nT": "{:.3f}",
                        }),
                        use_container_width=True,
                        hide_index=True,
                    )
                    st.caption("Alert bands are explanatory Dst severity categories: quiet > -50 nT; moderate -50 to -100 nT; intense -100 to -200 nT; severe ≤ -200 nT. They are not operational warnings.")
                    chart = go.Figure()
                    chart.add_trace(go.Bar(
                        x=forecast_output["period"].astype(str),
                        y=forecast_output["dst_t0_prediction_nT"],
                        name="T+1h prediction",
                        marker_color="#55E6FF",
                    ))
                    chart.add_trace(go.Bar(
                        x=forecast_output["period"].astype(str),
                        y=forecast_output["dst_t1_prediction_nT"],
                        name="T+2h prediction",
                        marker_color="#A78BFA",
                    ))
                    chart.update_layout(
                        barmode="group",
                        xaxis_title="Uploaded period",
                        yaxis_title="Predicted Dst (nT)",
                    )
                    plot(chart, 390)
                    csv_bytes = forecast_output.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "DOWNLOAD FORECAST CSV",
                        data=csv_bytes,
                        file_name="magnova_ai_inference_predictions.csv",
                        mime="text/csv",
                    )
                    st.markdown(
                        '<div class="data-note">These are offline model outputs for the uploaded data. '
                        'They are research predictions, not live alerts or operational grid-protection guidance. '
                        'The historical test results remain unchanged.</div>',
                        unsafe_allow_html=True,
                    )
            except Exception as exc:
                st.error(f"Inference could not be completed: {exc}")
                st.info("Check that each period contains at least 24 consecutive hourly rows and that the CSV uses the required hourly aggregate columns.")
        else:
            st.info("Upload an hourly solar-wind CSV to activate model inference.")

# ----------------------------- Model Arena -----------------------------
elif page=="Model Arena":
    section("03 / MODEL ARENA","Benchmark face-off","Compare the machine-learning model against the persistence baseline.")
    if results.empty:
        st.error("Final metrics are missing."); st.stop()
    c1,c2=st.columns([1.1,.9],gap="large")
    with c1:
        metric_choice=st.radio("Scoring metric",["RMSE","MAE"],horizontal=True)
        metric_key=metric_choice.lower()
        plot(make_error_comparison(results,metric_key),420)
    with c2:
        section("03A / FINAL SCORECARD","Exact test metrics")
        view=results[["target",f"xgb_{metric_key}",f"persistence_{metric_key}"]].copy()
        view["target"] = view["target"].map({"dst_t0":"T+1 hour","dst_t1":"T+2 hours"}).fillna(view["target"])
        view.columns=["Forecast horizon","XGBoost (nT)","Persistence (nT)"]
        st.dataframe(view.round(3),use_container_width=True,hide_index=True)
        st.markdown('<div class="signal violet"><strong>Reading the scoreboard</strong><br>RMSE penalizes large errors more strongly. MAE reports the average absolute error. Lower values are better for both.</div>',unsafe_allow_html=True)
    st.markdown("<div style='height:.8rem'></div>",unsafe_allow_html=True)
    section("03B / GENERALIZATION","Performance by historical segment","Each segment is reported separately; the source periods are non-contiguous and should not be treated as one continuous timeline.")
    if not period_results.empty:
        target=st.selectbox("Forecast target",["dst_t0","dst_t1"],format_func=lambda x:"T+1 hour" if x=="dst_t0" else "T+2 hours")
        d=period_results[period_results.target==target]
        fig=go.Figure()
        fig.add_trace(go.Bar(x=d.period,y=d.xgb_rmse,name="XGBoost",marker_color="#55E6FF"))
        fig.add_trace(go.Bar(x=d.period,y=d.persistence_rmse,name="Persistence",marker_color="#A78BFA"))
        fig.update_layout(barmode="group",yaxis_title="RMSE (nT)")
        plot(fig,360)
        st.dataframe(d.round(3),use_container_width=True,hide_index=True)

# ----------------------------- Storm Observatory -----------------------------
elif page=="Storm Observatory":
    section("04 / STORM OBSERVATORY","When the magnetosphere gets disturbed","Dive into storm-period error, onset samples and the Dst magnitude bands used in this visualization.")
    # Recompute matched subset metrics directly from held-out predictions so
    # storm skill is explicitly measured against persistence on identical rows.
    subset_scores = subset_skill_table(predictions)
    if not subset_scores.empty:
        section("04A / STORM-PERIOD SKILL", "Skill against mandatory persistence baseline",
                "Storm subset: actual target Dst ≤ -50 nT. Quiet subset: actual target Dst > -50 nT. Positive skill means XGBoost RMSE is lower than persistence.")
        display_scores = subset_scores.copy()
        display_scores["xgb_skill_vs_persistence"] = display_scores["xgb_skill_vs_persistence"].map(lambda v: f"{v:.1%}" if pd.notna(v) else "—")
        st.dataframe(display_scores.rename(columns={
            "horizon":"Forecast horizon", "subset":"Evaluation subset", "samples":"Samples",
            "xgb_rmse_nT":"XGBoost RMSE (nT)", "persistence_rmse_nT":"Persistence RMSE (nT)",
            "xgb_mae_nT":"XGBoost MAE (nT)", "persistence_mae_nT":"Persistence MAE (nT)",
            "xgb_skill_vs_persistence":"XGBoost skill vs persistence",
        }).round(3), use_container_width=True, hide_index=True)
        st.caption("Skill = 1 − (XGBoost RMSE / persistence RMSE). Negative values mean persistence performs better. Metrics are calculated on matched held-out samples.")
    if not storm_results.empty:
        t0=storm_results[storm_results.target=="dst_t0"].iloc[0]
        t1=storm_results[storm_results.target=="dst_t1"].iloc[0]
        s1,s2,s3=st.columns(3)
        with s1: metric("STORM SAMPLES",f"{int(t0.samples):,}","Actual target Dst ≤ -50 nT")
        with s2: metric("T+1H XGB STORM RMSE",f"{fmt(t0.xgb_rmse)} nT","Storm subset","pink")
        with s3: metric("T+1H PERSISTENCE RMSE",f"{fmt(t0.persistence_rmse)} nT","Storm subset","violet")
        section("04A / STORM SCOREBOARD","Model error in disturbed conditions")
        st.dataframe(storm_results.round(3),use_container_width=True,hide_index=True)
    st.markdown("<div style='height:.8rem'></div>",unsafe_allow_html=True)
    if not onset_results.empty:
        section("04B / STORM ONSET","Crossing into storm conditions","Onset is defined as current Dst above -50 nT and target Dst at or below -50 nT.")
        st.dataframe(onset_results.round(3),use_container_width=True,hide_index=True)
        st.markdown('<div class="signal pink"><strong>Exploratory only.</strong><br>The onset subsets are small. An RMSE improvement on these samples does not establish robust early-warning capability.</div>',unsafe_allow_html=True)
    if not magnitude_results.empty:
        st.markdown("<div style='height:.8rem'></div>",unsafe_allow_html=True)
        section("04C / SEVERITY PROFILE","Error by Dst magnitude","These Dst bands are explanatory research categories, not universal operational warning thresholds.")
        a,b=st.columns(2)
        with a: target=st.selectbox("Forecast target",["dst_t0","dst_t1"],key="storm_target",format_func=lambda x:"T+1 hour" if x=="dst_t0" else "T+2 hours")
        with b: error_metric=st.selectbox("Error metric",["rmse","mae"],format_func=str.upper)
        d=magnitude_results[magnitude_results.target==target]
        pivot=d.pivot(index="dst_band",columns="model",values=error_metric)
        fig=go.Figure()
        for model,color in [("xgb","#55E6FF"),("persistence","#A78BFA")]:
            if model in pivot.columns: fig.add_trace(go.Bar(x=pivot.index,y=pivot[model],name=model.upper(),marker_color=color))
        fig.update_layout(barmode="group",yaxis_title=f"{error_metric.upper()} (nT)")
        plot(fig,400)
        st.dataframe(d.round(3),use_container_width=True,hide_index=True)

# ----------------------------- Research Vault -----------------------------
elif page=="Research Vault":
    section("05 / RESEARCH VAULT","Inside the model and the data","Explore feature importance, historical distributions and data-quality diagnostics.")
    tab1,tab2,tab3=st.tabs(["◈ FEATURE INTELLIGENCE","◉ Dst ARCHIVE","⌁ SOLAR WIND & QUALITY"])
    with tab1:
        section("MODEL INTERPRETATION","What the model relied on","Importance indicates model behaviour, not causality.")
        x,y=st.columns(2)
        with x:
            show_image("reports/dst_t0_feature_importance.png","T+1h · built-in feature importance")
            show_image("reports/dst_t0_grouped_permutation.png","T+1h · grouped permutation importance")
        with y:
            show_image("reports/dst_t1_feature_importance.png","T+2h · built-in feature importance")
            show_image("reports/dst_t1_grouped_permutation.png","T+2h · grouped permutation importance")
        st.markdown('<div class="data-note">Built-in importance is split-based importance. Permutation importance measures performance sensitivity when a feature group is shuffled; neither method proves a physical causal relationship.</div>',unsafe_allow_html=True)
    with tab2:
        section("HISTORICAL ACTIVITY","Dst distribution across periods")
        show_image("reports/period_analysis/dst_period_boxplot.png","Dst distribution by historical period")
        if not dst_distribution.empty: st.dataframe(dst_distribution,use_container_width=True,hide_index=True)
    with tab3:
        section("SOLAR-WIND CONDITIONS","Measurement distributions and missingness")
        variable=st.selectbox("Solar-wind variable",["bx_gsm_mean","by_gsm_mean","bz_gsm_mean","bt_mean","density_mean","speed_mean","temperature_mean"])
        show_image(f"reports/period_analysis/{variable}_period_boxplot.png",variable)
        if not missing_data.empty:
            section("DATA QUALITY","Missing measurement comparison")
            st.dataframe(missing_data,use_container_width=True,hide_index=True)

# ----------------------------- Mission Dossier -----------------------------
elif page=="Mission Dossier":
    section("06 / MISSION DOSSIER","Research question, methodology & constraints","A compact technical briefing for the project review.")
    st.markdown('''<div class="panel"><div class="section-kicker">CORE RESEARCH QUESTION</div><div style="font-size:1.2rem;font-weight:700;line-height:1.55;color:#EDF7FF">Can machine learning models predict future geomagnetic disturbances using solar-wind measurements more accurately than a simple persistence-based forecasting method?</div></div>''',unsafe_allow_html=True)
    st.markdown("<div style='height:.7rem'></div>",unsafe_allow_html=True)
    l,r=st.columns(2,gap="large")
    with l:
        section("PIPELINE","Methodology")
        st.markdown("""
        1. Process historical solar-wind measurements and Dst labels.
        2. Aggregate measurements into completed hourly buckets.
        3. Engineer lag, rolling-window and physics-inspired features.
        4. Split chronologically into train, validation and held-out test partitions.
        5. Fit missing-value imputation on training data only.
        6. Train separate XGBoost models for the next-hour (T+1h) and two-hour-ahead (T+2h) Dst targets.
        7. Evaluate once on the held-out test set against persistence.
        """)
        section("FORECAST ALIGNMENT","Timing assumptions")
        st.markdown("Forecast origins use completed hourly solar-wind buckets. The stored target dst_t0 is the next hourly Dst observation (T+1h), and dst_t1 is the observation two hours ahead (T+2h), as defined by shift(-1) and shift(-2). The solar-wind-only XGBoost feature matrix does not include current Dst; current Dst is used separately for the persistence benchmark.")
    with r:
        section("DST VISUALIZATION","Illustrative magnitude bands")
        st.markdown("""
        | Category | Dst range |
        |---|---:|
        | Moderate | -50 to -100 nT |
        | Intense | -100 to -200 nT |
        | Severe | ≤ -200 nT |
        """)
        section("LIMITATIONS","What this prototype does not claim")
        st.markdown("""
        - Historical academic experiment; no live data connection.
        - Persistence outperformed XGBoost on overall test metrics.
        - Severe-storm observations are rare in the held-out sample.
        - Storm-onset analysis is exploratory.
        - Not validated for operational grid protection or real-time decisions.
        """)
    st.markdown('<div class="signal pink"><strong>Research prototype only.</strong><br>This dashboard visualizes saved historical outputs. It is not an operational space-weather warning system and must not be used to make real-world grid-protection decisions.</div>',unsafe_allow_html=True)

# ----------------------------- Footer -----------------------------
st.markdown('''<div class="footer">✦ MAGNOVA / TEAM G6 &nbsp; · &nbsp; GEOMAGNETIC INTELLIGENCE<br><br>BUILT FOR RESEARCH &nbsp; / &nbsp; NOT AN OPERATIONAL FORECAST SYSTEM</div>''',unsafe_allow_html=True)
