"""
ui/app.py — Glassmorphism Dark Neon Deepfake Detector UI
Run: streamlit run ui/app.py
"""

import os, io, time, tempfile
from pathlib import Path
import streamlit as st
import plotly.graph_objects as go
import requests

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="DEEPFAKE.AI", page_icon="🔬", layout="wide", initial_sidebar_state="collapsed")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600;700&family=Share+Tech+Mono&display=swap');
:root{--neon-cyan:#00f5ff;--neon-pink:#ff006e;--neon-green:#00ff88;--neon-purple:#bf00ff;--glass-bg:rgba(255,255,255,0.03);--glass-border:rgba(255,255,255,0.08);--dark-bg:#020408;--dark-card:rgba(6,12,20,0.85)}
*{box-sizing:border-box}
html,body,.stApp{background:var(--dark-bg)!important;font-family:'Rajdhani',sans-serif!important;color:#c0cfe0!important}
.stApp::before{content:'';position:fixed;inset:0;background-image:linear-gradient(rgba(0,245,255,0.03) 1px,transparent 1px),linear-gradient(90deg,rgba(0,245,255,0.03) 1px,transparent 1px);background-size:60px 60px;pointer-events:none;z-index:0}
.stApp::after{content:'';position:fixed;width:600px;height:600px;background:radial-gradient(circle,rgba(0,245,255,0.06) 0%,transparent 70%);top:-200px;right:-200px;border-radius:50%;pointer-events:none;animation:drift 20s ease-in-out infinite alternate;z-index:0}
@keyframes drift{0%{transform:translate(0,0) scale(1)}100%{transform:translate(-100px,150px) scale(1.3)}}
#MainMenu,footer,header,.stDeployButton{display:none!important}
.block-container{padding:2rem 3rem!important;max-width:1400px!important;position:relative;z-index:1}
section[data-testid="stSidebar"]{display:none!important}
.hero-title{font-family:'Orbitron',monospace;font-size:clamp(2.5rem,6vw,5rem);font-weight:900;text-align:center;letter-spacing:.15em;background:linear-gradient(135deg,var(--neon-cyan) 0%,#fff 40%,var(--neon-pink) 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;filter:drop-shadow(0 0 30px rgba(0,245,255,.4));margin-bottom:.2rem;line-height:1.1}
.hero-sub{text-align:center;font-family:'Share Tech Mono',monospace;font-size:.9rem;color:rgba(0,245,255,.6);letter-spacing:.3em;text-transform:uppercase;margin-bottom:3rem}
.glass-card{background:var(--dark-card);border:1px solid var(--glass-border);border-radius:20px;padding:2rem;backdrop-filter:blur(20px);-webkit-backdrop-filter:blur(20px);position:relative;overflow:hidden;transition:all .3s ease}
.glass-card::before{content:'';position:absolute;inset:0;border-radius:20px;padding:1px;background:linear-gradient(135deg,rgba(0,245,255,.2),transparent,rgba(255,0,110,.15));-webkit-mask:linear-gradient(#fff 0 0) content-box,linear-gradient(#fff 0 0);-webkit-mask-composite:xor;mask-composite:exclude;pointer-events:none}
.verdict-fake{background:linear-gradient(135deg,rgba(255,0,110,.15),rgba(255,107,0,.1));border:1px solid rgba(255,0,110,.4);border-radius:20px;padding:2.5rem;text-align:center;animation:pulse-red 2s ease-in-out infinite}
.verdict-real{background:linear-gradient(135deg,rgba(0,255,136,.1),rgba(0,245,255,.08));border:1px solid rgba(0,255,136,.4);border-radius:20px;padding:2.5rem;text-align:center;animation:pulse-green 2s ease-in-out infinite}
@keyframes pulse-red{0%,100%{box-shadow:0 0 20px rgba(255,0,110,.2),inset 0 0 30px rgba(255,0,110,.05)}50%{box-shadow:0 0 50px rgba(255,0,110,.4),inset 0 0 50px rgba(255,0,110,.1)}}
@keyframes pulse-green{0%,100%{box-shadow:0 0 20px rgba(0,255,136,.2),inset 0 0 30px rgba(0,255,136,.05)}50%{box-shadow:0 0 50px rgba(0,255,136,.4),inset 0 0 50px rgba(0,255,136,.1)}}
.verdict-label-fake{font-family:'Orbitron',monospace;font-size:clamp(1.8rem,4vw,3.5rem);font-weight:900;color:var(--neon-pink);text-shadow:0 0 30px var(--neon-pink),0 0 60px rgba(255,0,110,.5);letter-spacing:.2em}
.verdict-label-real{font-family:'Orbitron',monospace;font-size:clamp(1.8rem,4vw,3.5rem);font-weight:900;color:var(--neon-green);text-shadow:0 0 30px var(--neon-green),0 0 60px rgba(0,255,136,.5);letter-spacing:.2em}
.verdict-sub{font-family:'Share Tech Mono',monospace;font-size:.85rem;letter-spacing:.2em;margin-top:.75rem;opacity:.7}
.stat-box{background:rgba(0,245,255,.04);border:1px solid rgba(0,245,255,.12);border-radius:12px;padding:1rem 1.5rem;text-align:center}
.stat-value{font-family:'Orbitron',monospace;font-size:1.6rem;font-weight:700;color:var(--neon-cyan);text-shadow:0 0 15px rgba(0,245,255,.5)}
.stat-label{font-family:'Share Tech Mono',monospace;font-size:.7rem;letter-spacing:.2em;color:rgba(192,207,224,.5);text-transform:uppercase;margin-top:.3rem}
.badge-online{display:inline-flex;align-items:center;gap:6px;background:rgba(0,255,136,.1);border:1px solid rgba(0,255,136,.3);border-radius:20px;padding:4px 12px;font-family:'Share Tech Mono',monospace;font-size:.75rem;color:var(--neon-green)}
.badge-dot{width:6px;height:6px;background:var(--neon-green);border-radius:50%;box-shadow:0 0 8px var(--neon-green);animation:blink 1.5s ease-in-out infinite}
@keyframes blink{50%{opacity:.3}}
.section-header{font-family:'Orbitron',monospace;font-size:.75rem;letter-spacing:.3em;color:rgba(0,245,255,.5);text-transform:uppercase;border-bottom:1px solid rgba(0,245,255,.1);padding-bottom:.5rem;margin-bottom:1.5rem}
div[data-testid="stFileUploadDropzone"]{background:rgba(0,245,255,.02)!important;border:2px dashed rgba(0,245,255,.2)!important;border-radius:16px!important;transition:all .3s ease!important}
div[data-testid="stFileUploadDropzone"]:hover{border-color:rgba(0,245,255,.5)!important;background:rgba(0,245,255,.04)!important}
.stButton>button{background:linear-gradient(135deg,rgba(0,245,255,.15),rgba(191,0,255,.1))!important;border:1px solid rgba(0,245,255,.4)!important;color:var(--neon-cyan)!important;font-family:'Orbitron',monospace!important;font-size:.75rem!important;letter-spacing:.15em!important;border-radius:10px!important;padding:.6rem 1.5rem!important;transition:all .3s ease!important;text-transform:uppercase!important}
.stButton>button:hover{background:linear-gradient(135deg,rgba(0,245,255,.3),rgba(191,0,255,.2))!important;box-shadow:0 0 25px rgba(0,245,255,.3)!important;transform:translateY(-1px)!important}
.stTabs [data-baseweb="tab-list"]{background:transparent!important;border-bottom:1px solid rgba(0,245,255,.1)!important;gap:0!important}
.stTabs [data-baseweb="tab"]{background:transparent!important;color:rgba(192,207,224,.5)!important;font-family:'Orbitron',monospace!important;font-size:.7rem!important;letter-spacing:.15em!important;border:none!important;padding:.75rem 1.5rem!important;transition:all .3s!important}
.stTabs [aria-selected="true"]{color:var(--neon-cyan)!important;border-bottom:2px solid var(--neon-cyan)!important;text-shadow:0 0 15px rgba(0,245,255,.5)!important}
hr{border-color:rgba(0,245,255,.08)!important}
div[data-testid="stMetricValue"]{font-family:'Orbitron',monospace!important;color:var(--neon-cyan)!important;font-size:1.8rem!important}
.streamlit-expanderHeader{font-family:'Share Tech Mono',monospace!important;color:rgba(0,245,255,.6)!important;font-size:.8rem!important;letter-spacing:.1em!important;background:rgba(0,245,255,.03)!important;border:1px solid rgba(0,245,255,.1)!important;border-radius:8px!important}
::-webkit-scrollbar{width:4px}::-webkit-scrollbar-track{background:transparent}::-webkit-scrollbar-thumb{background:rgba(0,245,255,.2);border-radius:2px}
.scanlines{position:fixed;inset:0;pointer-events:none;z-index:9999;background:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,0,0,.03) 2px,rgba(0,0,0,.03) 4px)}
</style>
<div class="scanlines"></div>
"""
st.markdown(CSS, unsafe_allow_html=True)

@st.cache_data(ttl=5)
def get_health():
    try: return requests.get(f"{API_URL}/health", timeout=2).json()
    except: return {"status":"offline","models":{}}

def call_api(file_bytes, filename):
    ext = Path(filename).suffix.lower()
    mime = {
        ".jpg":"image/jpeg",
        ".jpeg":"image/jpeg",
        ".png":"image/png",
        ".webp":"image/webp",
        ".bmp":"image/bmp",
        ".wav":"audio/wav",
        ".mp3":"audio/mpeg",
        ".flac":"audio/flac",
        ".m4a":"audio/mp4",
        ".ogg":"audio/ogg",
        ".mp4":"video/mp4",
        ".avi":"video/x-msvideo",
        ".mov":"video/quicktime",
        ".mkv":"video/x-matroska",
        ".webm":"video/webm",
    }.get(ext,"application/octet-stream")
    r = requests.post(f"{API_URL}/detect", files={"file":(filename,io.BytesIO(file_bytes),mime)}, timeout=180)
    r.raise_for_status()
    return r.json()

def gauge_chart(fake_prob):
    pct = fake_prob * 100
    color = f"rgba(255,0,110,{0.6+fake_prob*0.4})" if fake_prob >= 0.5 else f"rgba(0,255,136,{0.6+(1-fake_prob)*0.4})"
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=round(pct,1),
        number={"suffix":"%","font":{"family":"Orbitron","size":40,"color":color}},
        title={"text":"DEEPFAKE PROBABILITY","font":{"family":"Orbitron","size":11,"color":"rgba(192,207,224,0.5)"}},
        gauge={"axis":{"range":[0,100],"tickfont":{"family":"Share Tech Mono","size":9,"color":"rgba(192,207,224,0.3)"},"tickcolor":"rgba(192,207,224,0.2)","tickwidth":1},
               "bar":{"color":color,"thickness":0.25},"bgcolor":"rgba(0,0,0,0)","bordercolor":"rgba(0,0,0,0)",
               "steps":[{"range":[0,35],"color":"rgba(0,255,136,0.08)"},{"range":[35,65],"color":"rgba(255,165,0,0.06)"},{"range":[65,100],"color":"rgba(255,0,110,0.08)"}],
               "threshold":{"line":{"color":"rgba(255,255,255,0.3)","width":2},"thickness":0.8,"value":50}}))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",height=280,margin=dict(l=20,r=20,t=50,b=20),font={"color":"rgba(192,207,224,0.7)"})
    return fig

def prob_bars(fake_p, real_p):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=["AUTHENTIC","DEEPFAKE"],y=[real_p*100,fake_p*100],
        marker=dict(color=["rgba(0,255,136,0.7)","rgba(255,0,110,0.7)"],line=dict(color=["rgba(0,255,136,1)","rgba(255,0,110,1)"],width=1)),
        text=[f"{real_p*100:.1f}%",f"{fake_p*100:.1f}%"],textposition="outside",textfont={"family":"Orbitron","size":11,"color":"rgba(192,207,224,0.8)"},width=0.5))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(6,12,20,0.5)",height=220,margin=dict(l=10,r=10,t=20,b=10),
        font={"family":"Share Tech Mono","color":"rgba(192,207,224,0.5)","size":9},
        xaxis=dict(showgrid=False,tickfont={"family":"Orbitron","size":9}),
        yaxis=dict(showgrid=True,gridcolor="rgba(255,255,255,0.04)",range=[0,115],ticksuffix="%",tickfont={"family":"Share Tech Mono","size":8}),
        bargap=0.4,showlegend=False)
    return fig

def render_result(result):
    verdict = result.get("verdict","UNKNOWN")
    conf = result.get("confidence","LOW")
    fake_p = result.get("fake_probability",0.5)
    real_p = result.get("real_probability",0.5)
    modality = result.get("modality","unknown").upper()
    latency = result.get("latency_ms","—")
    is_fake = verdict == "FAKE"
    v_class = "verdict-fake" if is_fake else "verdict-real"
    v_label = "verdict-label-fake" if is_fake else "verdict-label-real"
    icon = "⚠" if is_fake else "✓"
    label = "DEEPFAKE DETECTED" if is_fake else "AUTHENTIC MEDIA"
    conf_color = {"HIGH":"#00ff88","MEDIUM":"#ffa500","LOW":"#ff6b6b"}.get(conf,"#888")
    st.markdown(f'<div class="{v_class}"><div class="{v_label}">{icon} {label}</div><div class="verdict-sub" style="color:{conf_color};">CONFIDENCE: {conf} &nbsp;|&nbsp; MODALITY: {modality} &nbsp;|&nbsp; {latency} ms</div></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1: st.plotly_chart(gauge_chart(fake_p), use_container_width=True, key="gauge")
    with col2: st.plotly_chart(prob_bars(fake_p, real_p), use_container_width=True, key="bars")
    st.markdown(f'<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;margin-top:.5rem;"><div class="stat-box"><div class="stat-value">{fake_p*100:.1f}%</div><div class="stat-label">Fake Score</div></div><div class="stat-box"><div class="stat-value" style="color:{"#ff006e" if is_fake else "#00ff88"};">{verdict}</div><div class="stat-label">Verdict</div></div><div class="stat-box"><div class="stat-value" style="color:{conf_color};">{conf}</div><div class="stat-label">Confidence</div></div></div>', unsafe_allow_html=True)
    with st.expander("◈ RAW JSON RESPONSE"): st.json(result)

# ── HEADER ──
health = get_health()
api_ok = health.get("status") == "ok"
col_title, col_status = st.columns([5,1])
with col_title:
    st.markdown('<div class="hero-title">DEEPFAKE.AI</div><div class="hero-sub">◈ Neural Truth Engine v1.0 ◈ Real-Time Detection ◈</div>', unsafe_allow_html=True)
with col_status:
    st.markdown("<br><br>", unsafe_allow_html=True)
    if api_ok:
        st.markdown('<div class="badge-online"><div class="badge-dot"></div> SYSTEMS ONLINE</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="display:inline-flex;align-items:center;gap:6px;background:rgba(255,0,110,0.1);border:1px solid rgba(255,0,110,0.3);border-radius:20px;padding:4px 12px;font-family:\'Share Tech Mono\',monospace;font-size:0.75rem;color:#ff006e;">● API OFFLINE</div>', unsafe_allow_html=True)

# ── MODEL STATUS ──
models = health.get("models",{})
m_cols = st.columns(3)
model_info = [
    ("🖼","IMAGE MODEL","EfficientNet-B0","95.8%",models.get("image",False)),
    ("🎵","AUDIO MODEL","MLP + Features","99.6%",models.get("audio",False)),
    ("🎬","VIDEO MODEL","Frame+Audio Hybrid","Active",models.get("video",False)),
]
for col,(icon,name,arch,acc,loaded) in zip(m_cols,model_info):
    with col:
        sc = "#00ff88" if loaded else "rgba(192,207,224,0.2)"
        st_text = "LOADED" if loaded else "OFFLINE"
        st.markdown(f'<div class="glass-card" style="padding:1.2rem;"><div style="display:flex;justify-content:space-between;align-items:center;"><div><div style="font-family:\'Orbitron\',monospace;font-size:.7rem;letter-spacing:.15em;color:rgba(192,207,224,.4);margin-bottom:4px;">{icon} {name}</div><div style="font-family:\'Share Tech Mono\',monospace;font-size:.75rem;color:var(--neon-cyan);">{arch}</div></div><div style="text-align:right;"><div style="font-family:\'Orbitron\',monospace;font-size:1.1rem;color:{sc};text-shadow:0 0 10px {sc};">{acc}</div><div style="font-family:\'Share Tech Mono\',monospace;font-size:.65rem;color:{sc};opacity:.7;">{st_text}</div></div></div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── TABS ──
tab1, tab2, tab3 = st.tabs(["◈ ANALYZE","◈ BATCH SCAN","◈ INTEL"])

with tab1:
    up_col, res_col = st.columns([1,1], gap="large")
    with up_col:
        st.markdown('<div class="section-header">◈ Upload Target Media</div>', unsafe_allow_html=True)
        st.markdown('<div style="font-family:\'Share Tech Mono\',monospace;font-size:.75rem;color:rgba(192,207,224,.4);margin-bottom:1rem;letter-spacing:.1em;">SUPPORTED: JPG · PNG · WEBP · MP4 · AVI · MOV · MKV · WAV · MP3 · FLAC</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("Drop file here", type=["jpg","jpeg","png","webp","bmp","mp4","avi","mov","mkv","webm","wav","mp3","flac","m4a","ogg"], label_visibility="collapsed")
        if uploaded:
            ext = Path(uploaded.name).suffix.lower()
            if ext in {".jpg",".jpeg",".png",".webp",".bmp"}: st.image(uploaded, use_container_width=True, caption=f"◈ {uploaded.name}")
            elif ext in {".mp4",".avi",".mov",".mkv",".webm"}: st.video(uploaded)
            elif ext in {".wav",".mp3",".flac",".m4a"}: st.audio(uploaded)
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("◈ RUN ANALYSIS", use_container_width=True):
                with res_col:
                    try:
                        result = call_api(uploaded.read(), uploaded.name)
                        st.session_state["last_result"] = result
                    except requests.exceptions.ConnectionError:
                        st.error("◈ API OFFLINE — Start with: python3 api_server.py")
                    except Exception as e:
                        st.error(f"◈ ERROR: {e}")
    with res_col:
        st.markdown('<div class="section-header">◈ Analysis Output</div>', unsafe_allow_html=True)
        if "last_result" in st.session_state:
            render_result(st.session_state["last_result"])
        else:
            st.markdown('<div style="text-align:center;padding:4rem 2rem;border:1px dashed rgba(0,245,255,.1);border-radius:16px;background:rgba(0,245,255,.01);"><div style="font-size:3rem;margin-bottom:1rem;opacity:.3;">◈</div><div style="font-family:\'Orbitron\',monospace;font-size:.75rem;letter-spacing:.2em;color:rgba(0,245,255,.3);">AWAITING INPUT</div></div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="section-header">◈ Batch Scan Protocol</div>', unsafe_allow_html=True)
    files = st.file_uploader("Upload multiple files", type=["jpg","jpeg","png","webp","bmp","mp4","avi","mov","mkv","webm","wav","mp3","flac","m4a","ogg"], accept_multiple_files=True, label_visibility="collapsed")
    if files and st.button("◈ INITIATE BATCH SCAN"):
        progress = st.progress(0)
        results = []
        for i,f in enumerate(files):
            try:
                r = call_api(f.read(), f.name)
                results.append({"File":f.name,"Verdict":r.get("verdict","?"),"Fake %":f"{r.get('fake_probability',0)*100:.1f}%","Confidence":r.get("confidence","?"),"Latency":f"{r.get('latency_ms','?')} ms"})
            except: results.append({"File":f.name,"Verdict":"ERROR","Fake %":"—","Confidence":"—","Latency":"—"})
            progress.progress((i+1)/len(files))
        if results:
            import pandas as pd
            df = pd.DataFrame(results)
            fakes = sum(1 for r in results if r["Verdict"]=="FAKE")
            c1,c2,c3 = st.columns(3)
            c1.metric("Files Scanned",len(files)); c2.metric("Deepfakes Found",fakes); c3.metric("Rate",f"{fakes/len(files)*100:.0f}%")
            st.dataframe(df, use_container_width=True, hide_index=True)

with tab3:
    st.markdown('<div class="section-header">◈ System Intelligence</div>', unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        st.markdown('<div class="glass-card"><div style="font-family:\'Orbitron\',monospace;font-size:.8rem;color:var(--neon-cyan);letter-spacing:.15em;margin-bottom:1.5rem;">◈ DETECTION ARCHITECTURE</div><div style="font-family:\'Share Tech Mono\',monospace;font-size:.8rem;color:rgba(192,207,224,.6);line-height:2;">IMAGE ENGINE → EfficientNet-B0 + Attention<br>AUDIO ENGINE → MLP on MFCC + Spectral<br>VIDEO ENGINE → EfficientNet + BiLSTM<br>FUSION LAYER → Weighted Ensemble<br>TRAINING DATA → 140k faces + WaveFake<br>FRAMEWORK → PyTorch 2.2 + MPS</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="glass-card"><div style="font-family:\'Orbitron\',monospace;font-size:.8rem;color:var(--neon-cyan);letter-spacing:.15em;margin-bottom:1.5rem;">◈ PERFORMANCE METRICS</div><div style="font-family:\'Share Tech Mono\',monospace;font-size:.8rem;color:rgba(192,207,224,.6);line-height:2;">IMAGE ACCURACY → 95.8%<br>IMAGE AUC-ROC → 0.9926<br>AUDIO ACCURACY → 99.6%<br>AUDIO AUC-ROC → 0.9999<br>AVG LATENCY → &lt; 200ms<br>DEVICE → Apple M4 MPS</div></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="glass-card"><div style="font-family:\'Orbitron\',monospace;font-size:.8rem;color:var(--neon-pink);letter-spacing:.15em;margin-bottom:1rem;">⚠ MODEL LIMITATIONS</div><div style="font-family:\'Share Tech Mono\',monospace;font-size:.78rem;color:rgba(192,207,224,.55);line-height:1.9;">IMAGE MODEL: Trained on face-swap deepfakes (FaceForensics++, DFDC).<br>It detects facial manipulation artifacts — NOT general AI-generated images.<br>Gemini/DALL-E/Midjourney images may score as authentic because they lack<br>face-swap artifacts. A separate diffusion detector is needed for AI art.<br><br>AUDIO MODEL: Detects voice cloning and TTS synthesis accurately.<br>ACCURACY CAVEAT: Results vary on unseen deepfake generators.</div></div>', unsafe_allow_html=True) )

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
