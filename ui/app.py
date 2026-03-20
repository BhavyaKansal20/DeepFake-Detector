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
    st.markdown('<div class="glass-card"><div style="font-family:\'Orbitron\',monospace;font-size:.8rem;color:var(--neon-pink);letter-spacing:.15em;margin-bottom:1rem;">⚠ MODEL LIMITATIONS</div><div style="font-family:\'Share Tech Mono\',monospace;font-size:.78rem;color:rgba(192,207,224,.55);line-height:1.9;">IMAGE MODEL: Trained on face-swap deepfakes (FaceForensics++, DFDC).<br>It detects facial manipulation artifacts — NOT general AI-generated images.<br>Gemini/DALL-E/Midjourney images may score as authentic because they lack<br>face-swap artifacts. A separate diffusion detector is needed for AI art.<br><br>AUDIO MODEL: Detects voice cloning and TTS synthesis accurately.<br>ACCURACY CAVEAT: Results vary on unseen deepfake generators.</div></div>', unsafe_allow_html=True) 