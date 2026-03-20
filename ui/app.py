"""
DeepFake Scanner — Hugging Face Space Entry Point
Author: Bhavya Kansal | Multimodex AI
Run: streamlit run app.py
"""

import os, sys, threading, time
import streamlit as st
import requests
from pathlib import Path

# ─── Hugging Face: start FastAPI in background thread ───────────────────────
def _start_api():
    """Boots the FastAPI server on port 8000 inside the same container."""
    import subprocess
    subprocess.Popen(
        [sys.executable, "api_server.py"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

# Start API once per session
if "api_started" not in st.session_state:
    t = threading.Thread(target=_start_api, daemon=True)
    t.start()
    st.session_state["api_started"] = True
    time.sleep(4)  # give FastAPI time to boot

API_BASE = "http://localhost:8000"

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DeepFake Scanner | Multimodex AI",
    page_icon="🔬",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─── Global CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Inter:wght@300;400;500;600&display=swap');

:root {
    --neon-cyan:   #00f5ff;
    --neon-red:    #ff3366;
    --neon-green:  #00ff88;
    --neon-yellow: #ffcc00;
    --bg-dark:     #030712;
    --card-bg:     rgba(255,255,255,0.03);
    --card-border: rgba(0,245,255,0.15);
    --text-muted:  rgba(255,255,255,0.5);
}

html, body, [class*="css"] {
    background-color: var(--bg-dark) !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif;
}

/* Header */
.df-header {
    text-align: center;
    padding: 2.5rem 0 1.5rem;
}
.df-header h1 {
    font-family: 'Orbitron', monospace;
    font-size: 2.8rem;
    font-weight: 900;
    background: linear-gradient(135deg, var(--neon-cyan), #7c3aed, var(--neon-cyan));
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shimmer 4s linear infinite;
    letter-spacing: 0.05em;
    margin: 0;
}
@keyframes shimmer {
    0%   { background-position: 0% center; }
    100% { background-position: 200% center; }
}
.df-header .tagline {
    color: var(--text-muted);
    font-size: 0.95rem;
    margin-top: 0.5rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}
.df-badge {
    display: inline-block;
    background: rgba(0,245,255,0.08);
    border: 1px solid var(--card-border);
    border-radius: 2rem;
    padding: 0.25rem 0.9rem;
    font-size: 0.75rem;
    color: var(--neon-cyan);
    margin: 0.2rem;
    letter-spacing: 0.06em;
    font-family: 'Orbitron', monospace;
}

/* Stat cards */
.stat-row {
    display: flex;
    gap: 1rem;
    justify-content: center;
    flex-wrap: wrap;
    margin: 1.5rem 0;
}
.stat-card {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 12px;
    padding: 1rem 1.5rem;
    text-align: center;
    min-width: 120px;
}
.stat-card .val {
    font-family: 'Orbitron', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--neon-cyan);
}
.stat-card .label {
    font-size: 0.7rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 0.2rem;
}

/* Upload zone */
.stFileUploader > div {
    background: rgba(0,245,255,0.03) !important;
    border: 2px dashed rgba(0,245,255,0.25) !important;
    border-radius: 16px !important;
    transition: all 0.3s ease;
}
.stFileUploader > div:hover {
    border-color: var(--neon-cyan) !important;
    background: rgba(0,245,255,0.06) !important;
}

/* Result card */
.result-card {
    background: var(--card-bg);
    border-radius: 16px;
    padding: 1.8rem;
    margin: 1.5rem 0;
    position: relative;
    overflow: hidden;
}
.result-card.fake  { border: 1px solid rgba(255,51,102,0.5); }
.result-card.real  { border: 1px solid rgba(0,255,136,0.5);  }
.result-card.unsure{ border: 1px solid rgba(255,204,0,0.5);  }

.verdict-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    font-family: 'Orbitron', monospace;
    font-size: 1.8rem;
    font-weight: 900;
    padding: 0.4rem 1.4rem;
    border-radius: 2rem;
    letter-spacing: 0.08em;
}
.verdict-badge.fake  { color: var(--neon-red);   background: rgba(255,51,102,0.12); }
.verdict-badge.real  { color: var(--neon-green);  background: rgba(0,255,136,0.12);  }
.verdict-badge.unsure{ color: var(--neon-yellow); background: rgba(255,204,0,0.12);  }

/* Probability bar */
.prob-bar-wrap { margin: 1.2rem 0; }
.prob-bar-wrap .label-row {
    display: flex; justify-content: space-between;
    font-size: 0.78rem; color: var(--text-muted);
    text-transform: uppercase; letter-spacing: 0.08em;
    margin-bottom: 0.4rem;
}
.prob-bar-bg {
    background: rgba(255,255,255,0.06);
    border-radius: 4px; height: 10px; overflow: hidden;
}
.prob-bar-fill {
    height: 100%; border-radius: 4px;
    transition: width 0.8s cubic-bezier(0.4,0,0.2,1);
}
.prob-bar-fill.fake  { background: linear-gradient(90deg, #ff3366, #ff0044); }
.prob-bar-fill.real  { background: linear-gradient(90deg, #00ff88, #00cc6a); }

/* Detail grid */
.detail-grid {
    display: grid; grid-template-columns: 1fr 1fr;
    gap: 0.8rem; margin-top: 1.2rem;
}
.detail-item {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px; padding: 0.8rem 1rem;
}
.detail-item .dk  { font-size: 0.7rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.08em; }
.detail-item .dv  { font-family: 'Orbitron', monospace; font-size: 1rem; color: var(--neon-cyan); margin-top: 0.15rem; }

/* Spinner */
.stSpinner > div { border-top-color: var(--neon-cyan) !important; }

/* Footer */
.df-footer {
    text-align: center; margin-top: 3rem; padding-top: 2rem;
    border-top: 1px solid rgba(255,255,255,0.06);
    color: var(--text-muted); font-size: 0.78rem;
}
.df-footer a { color: var(--neon-cyan); text-decoration: none; }
.df-footer .brand { font-family: 'Orbitron', monospace; color: var(--neon-cyan); }

/* Hide streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 0 !important; max-width: 780px; }
</style>
""", unsafe_allow_html=True)

# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="df-header">
  <h1>🔬 DEEPFAKE SCANNER</h1>
  <p class="tagline">Neural Truth Engine · Multimodex AI</p>
  <div style="margin-top:0.8rem;">
    <span class="df-badge">EfficientNet-B0</span>
    <span class="df-badge">AudioMLP</span>
    <span class="df-badge">FastAPI</span>
    <span class="df-badge">Dual-Model Fusion</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ─── Stats ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="stat-row">
  <div class="stat-card"><div class="val">95.8%</div><div class="label">Face-Swap Acc</div></div>
  <div class="stat-card"><div class="val">98.1%</div><div class="label">GAN AUC</div></div>
  <div class="stat-card"><div class="val">99.6%</div><div class="label">Voice Clone Acc</div></div>
  <div class="stat-card"><div class="val">&lt;200ms</div><div class="label">Inference</div></div>
</div>
""", unsafe_allow_html=True)

# ─── Upload ───────────────────────────────────────────────────────────────────
st.markdown("### Upload Media to Analyze")
st.caption("Supported: Images (JPG, PNG, WEBP) · Audio (WAV, MP3, FLAC) · Video (MP4, AVI, MOV)")

uploaded = st.file_uploader(
    "Drop your file here",
    type=["jpg", "jpeg", "png", "webp", "bmp", "wav", "mp3", "flac", "m4a", "ogg", "mp4", "avi", "mov", "mkv"],
    label_visibility="collapsed"
)

if uploaded:
    ext = Path(uploaded.name).suffix.lower()

    # Preview
    if ext in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(uploaded, use_container_width=True, caption=uploaded.name)
    elif ext in {".wav", ".mp3", ".flac", ".m4a", ".ogg"}:
        st.audio(uploaded)
    elif ext in {".mp4", ".avi", ".mov", ".mkv"}:
        st.video(uploaded)

    analyze_btn = st.button("🔬 Analyze Media", use_container_width=True, type="primary")

    if analyze_btn:
        with st.spinner("Neural engines processing..."):
            try:
                files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type or "application/octet-stream")}
                resp = requests.post(f"{API_BASE}/detect", files=files, timeout=120)
                result = resp.json()
            except requests.exceptions.ConnectionError:
                st.error("⚠️ API server is still starting up. Please wait 5 seconds and retry.")
                st.stop()
            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()

        if "error" in result:
            st.error(f"Detection error: {result['error']}")
        else:
            fake_prob = result.get("fake_probability", 0)
            real_prob = result.get("real_probability", 1)
            verdict   = result.get("verdict", "UNKNOWN")
            confidence= result.get("confidence", "LOW")
            modality  = result.get("modality", "unknown")
            latency   = result.get("latency_ms", 0)

            css_cls = "fake" if verdict == "FAKE" else ("real" if verdict == "REAL" else "unsure")
            icon    = "🚨" if verdict == "FAKE" else ("✅" if verdict == "REAL" else "⚠️")

            st.markdown(f"""
            <div class="result-card {css_cls}">
              <div style="margin-bottom:1.2rem;">
                <span class="verdict-badge {css_cls}">{icon} {verdict}</span>
                <span style="margin-left:1rem; font-size:0.8rem; color:var(--text-muted); font-family:Orbitron,monospace; letter-spacing:0.06em;">
                  {confidence} CONFIDENCE
                </span>
              </div>

              <div class="prob-bar-wrap">
                <div class="label-row"><span>Real</span><span>{real_prob*100:.1f}%</span></div>
                <div class="prob-bar-bg">
                  <div class="prob-bar-fill real" style="width:{real_prob*100:.1f}%"></div>
                </div>
              </div>

              <div class="prob-bar-wrap">
                <div class="label-row"><span>Fake</span><span>{fake_prob*100:.1f}%</span></div>
                <div class="prob-bar-bg">
                  <div class="prob-bar-fill fake" style="width:{fake_prob*100:.1f}%"></div>
                </div>
              </div>

              <div class="detail-grid">
                <div class="detail-item"><div class="dk">Modality</div><div class="dv">{modality.upper()}</div></div>
                <div class="detail-item"><div class="dk">Latency</div><div class="dv">{latency} ms</div></div>
                <div class="detail-item"><div class="dk">Fake Prob</div><div class="dv">{fake_prob:.4f}</div></div>
                <div class="detail-item"><div class="dk">Real Prob</div><div class="dv">{real_prob:.4f}</div></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Component scores (images)
            detail = result.get("detail", {})
            if detail:
                with st.expander("🔍 Component Scores & Debug Info"):
                    st.json(detail)

            # Warning
            st.info("⚠️ Results are probabilistic. Use as supporting evidence, not sole proof.")

else:
    # Empty state
    st.markdown("""
    <div style="text-align:center; padding:3rem 1rem; color:rgba(255,255,255,0.3);">
      <div style="font-size:3rem; margin-bottom:1rem;">🔬</div>
      <div style="font-family:Orbitron,monospace; letter-spacing:0.1em; font-size:0.85rem;">
        AWAITING MEDIA INPUT
      </div>
      <div style="font-size:0.8rem; margin-top:0.5rem;">
        Upload an image, video or audio file to begin analysis
      </div>
    </div>
    """, unsafe_allow_html=True)

# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="df-footer">
  <div class="brand">DEEPFAKE SCANNER</div>
  <div style="margin-top:0.4rem;">
    Built by <a href="https://bhavyakansal.dev" target="_blank">Bhavya Kansal</a> ·
    <a href="https://multimodexai.vercel.app" target="_blank">Multimodex AI</a> ·
    <a href="https://github.com/BhavyaKansal20/DeepFake-Detector" target="_blank">GitHub</a>
  </div>
  <div style="margin-top:0.3rem; font-size:0.72rem; opacity:0.6;">
    EfficientNet-B0 · AudioMLP · Dual-Model Fusion · FastAPI · Streamlit
  </div>
</div>
""", unsafe_allow_html=True)
