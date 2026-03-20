"""
app_render.py — All-in-one Streamlit app for Render deployment
No separate API server needed — detection runs directly in Streamlit
"""
import os, time, pickle, io, tempfile
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import timm
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path
from PIL import Image as PILImage
import albumentations as A
from albumentations.pytorch import ToTensorV2

st.set_page_config(page_title="DEEPFAKE.AI", page_icon="🔬", layout="wide", initial_sidebar_state="collapsed")

# ── CSS (same neon theme) ─────────────────────────────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600;700&family=Share+Tech+Mono&display=swap');
:root{--neon-cyan:#00f5ff;--neon-pink:#ff006e;--neon-green:#00ff88;--dark-bg:#020408;--dark-card:rgba(6,12,20,0.85)}
html,body,.stApp{background:var(--dark-bg)!important;font-family:'Rajdhani',sans-serif!important;color:#c0cfe0!important}
.stApp::before{content:'';position:fixed;inset:0;background-image:linear-gradient(rgba(0,245,255,0.03) 1px,transparent 1px),linear-gradient(90deg,rgba(0,245,255,0.03) 1px,transparent 1px);background-size:60px 60px;pointer-events:none;z-index:0}
#MainMenu,footer,header{display:none!important}
.block-container{padding:2rem 3rem!important;max-width:1400px!important;position:relative;z-index:1}
section[data-testid="stSidebar"]{display:none!important}
.hero-title{font-family:'Orbitron',monospace;font-size:clamp(2.5rem,6vw,5rem);font-weight:900;text-align:center;letter-spacing:.15em;background:linear-gradient(135deg,var(--neon-cyan) 0%,#fff 40%,var(--neon-pink) 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;filter:drop-shadow(0 0 30px rgba(0,245,255,.4));margin-bottom:.2rem}
.hero-sub{text-align:center;font-family:'Share Tech Mono',monospace;font-size:.9rem;color:rgba(0,245,255,.6);letter-spacing:.3em;text-transform:uppercase;margin-bottom:3rem}
.glass-card{background:var(--dark-card);border:1px solid rgba(255,255,255,0.08);border-radius:20px;padding:2rem;backdrop-filter:blur(20px)}
.verdict-fake{background:linear-gradient(135deg,rgba(255,0,110,.15),rgba(255,107,0,.1));border:1px solid rgba(255,0,110,.4);border-radius:20px;padding:2.5rem;text-align:center;animation:pulse-red 2s ease-in-out infinite}
.verdict-real{background:linear-gradient(135deg,rgba(0,255,136,.1),rgba(0,245,255,.08));border:1px solid rgba(0,255,136,.4);border-radius:20px;padding:2.5rem;text-align:center;animation:pulse-green 2s ease-in-out infinite}
@keyframes pulse-red{0%,100%{box-shadow:0 0 20px rgba(255,0,110,.2)}50%{box-shadow:0 0 50px rgba(255,0,110,.4)}}
@keyframes pulse-green{0%,100%{box-shadow:0 0 20px rgba(0,255,136,.2)}50%{box-shadow:0 0 50px rgba(0,255,136,.4)}}
.verdict-label-fake{font-family:'Orbitron',monospace;font-size:clamp(1.8rem,4vw,3.5rem);font-weight:900;color:var(--neon-pink);text-shadow:0 0 30px var(--neon-pink);letter-spacing:.2em}
.verdict-label-real{font-family:'Orbitron',monospace;font-size:clamp(1.8rem,4vw,3.5rem);font-weight:900;color:var(--neon-green);text-shadow:0 0 30px var(--neon-green);letter-spacing:.2em}
.verdict-sub{font-family:'Share Tech Mono',monospace;font-size:.85rem;letter-spacing:.2em;margin-top:.75rem;opacity:.7}
.stat-box{background:rgba(0,245,255,.04);border:1px solid rgba(0,245,255,.12);border-radius:12px;padding:1rem 1.5rem;text-align:center}
.stat-value{font-family:'Orbitron',monospace;font-size:1.6rem;font-weight:700;color:var(--neon-cyan);text-shadow:0 0 15px rgba(0,245,255,.5)}
.stat-label{font-family:'Share Tech Mono',monospace;font-size:.7rem;letter-spacing:.2em;color:rgba(192,207,224,.5);text-transform:uppercase;margin-top:.3rem}
.section-header{font-family:'Orbitron',monospace;font-size:.75rem;letter-spacing:.3em;color:rgba(0,245,255,.5);text-transform:uppercase;border-bottom:1px solid rgba(0,245,255,.1);padding-bottom:.5rem;margin-bottom:1.5rem}
div[data-testid="stFileUploadDropzone"]{background:rgba(0,245,255,.02)!important;border:2px dashed rgba(0,245,255,.2)!important;border-radius:16px!important}
.stButton>button{background:linear-gradient(135deg,rgba(0,245,255,.15),rgba(191,0,255,.1))!important;border:1px solid rgba(0,245,255,.4)!important;color:var(--neon-cyan)!important;font-family:'Orbitron',monospace!important;font-size:.75rem!important;letter-spacing:.15em!important;border-radius:10px!important;text-transform:uppercase!important}
.stTabs [data-baseweb="tab-list"]{background:transparent!important;border-bottom:1px solid rgba(0,245,255,.1)!important}
.stTabs [data-baseweb="tab"]{background:transparent!important;color:rgba(192,207,224,.5)!important;font-family:'Orbitron',monospace!important;font-size:.7rem!important;letter-spacing:.15em!important;border:none!important}
.stTabs [aria-selected="true"]{color:var(--neon-cyan)!important;border-bottom:2px solid var(--neon-cyan)!important}
.scanlines{position:fixed;inset:0;pointer-events:none;z-index:9999;background:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,0,0,.03) 2px,rgba(0,0,0,.03) 4px)}
</style>
<div class="scanlines"></div>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ── Load models (cached so they load only once) ───────────────────────────────
@st.cache_resource
def load_models():
    device = torch.device("cpu")  # Render free = CPU

    def build_effb0():
        bb = timm.create_model('efficientnet_b0', pretrained=False, num_classes=0, global_pool='avg')
        head = nn.Sequential(nn.Linear(bb.num_features,256), nn.LayerNorm(256), nn.ReLU(True), nn.Dropout(0.3), nn.Linear(256,2))
        return nn.Sequential(bb, head)

    def build_gan():
        bb = timm.create_model('efficientnet_b0', pretrained=False, num_classes=0, global_pool='avg')
        head = nn.Sequential(nn.Linear(bb.num_features,256), nn.LayerNorm(256), nn.ReLU(True), nn.Dropout(0.35), nn.Linear(256,64), nn.ReLU(True), nn.Dropout(0.2), nn.Linear(64,2))
        return nn.Sequential(bb, head)

    class AudioMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(26,256), nn.BatchNorm1d(256), nn.ReLU(True), nn.Dropout(0.4),
                nn.Linear(256,512), nn.BatchNorm1d(512), nn.ReLU(True), nn.Dropout(0.4),
                nn.Linear(512,256), nn.BatchNorm1d(256), nn.ReLU(True), nn.Dropout(0.2),
                nn.Linear(256,128), nn.ReLU(True), nn.Dropout(0.2), nn.Linear(128,2),
            )
        def forward(self, x): return self.net(x)

    # Face-swap
    fs = build_effb0().to(device)
    fs.load_state_dict(torch.load('models/image_model/best_model.pt', map_location=device)['model_state_dict'])
    fs.eval()

    # GAN
    gn = None
    if os.path.exists('models/gan_detector/best_model.pt'):
        gn = build_gan().to(device)
        gn.load_state_dict(torch.load('models/gan_detector/best_model.pt', map_location=device)['model_state_dict'])
        gn.eval()

    # Audio
    am = AudioMLP().to(device)
    am.load_state_dict(torch.load('models/audio_model/best_model.pt', map_location=device)['model_state_dict'])
    am.eval()
    with open('models/audio_model/feature_scaler.pkl', 'rb') as f:
        sc = pickle.load(f)

    return fs, gn, am, sc, device

IMG_TF = A.Compose([A.Resize(224,224), A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]), ToTensorV2()])

def detect_image(uploaded_file, faceswap_model, gan_model, device):
    t0 = time.time()
    img = np.array(PILImage.open(uploaded_file).convert('RGB'))
    tensor = IMG_TF(image=img)['image'].unsqueeze(0).to(device)
    with torch.no_grad():
        p1 = F.softmax(faceswap_model(tensor), dim=-1).cpu().numpy()[0]
    fs = float(p1[1])
    gn = 0.0
    if gan_model:
        with torch.no_grad():
            p2 = F.softmax(gan_model(tensor), dim=-1).cpu().numpy()[0]
        gn = float(p2[1])
    final = max(fs, gn) if (fs>0.8 or gn>0.8) else (fs*0.45 + gn*0.55) if gan_model else fs
    v = "FAKE" if final >= 0.5 else "REAL"
    c = "HIGH" if abs(final-0.5)>0.35 else "MEDIUM" if abs(final-0.5)>0.15 else "LOW"
    return {"verdict":v,"confidence":c,"fake_probability":round(final,4),"real_probability":round(1-final,4),
            "modality":"image","latency_ms":round((time.time()-t0)*1000,1),
            "detail":{"faceswap_score":round(fs,4),"gan_score":round(gn,4)}}

def detect_audio(uploaded_file, audio_model, scaler, device):
    t0 = time.time()
    import librosa, soundfile as sf
    with tempfile.NamedTemporaryFile(suffix=Path(uploaded_file.name).suffix, delete=False) as f:
        f.write(uploaded_file.read()); tmp = f.name
    try:
        y, sr = librosa.load(tmp, sr=22050, mono=True)
        feats = {
            'chroma_stft': float(np.mean(librosa.feature.chroma_stft(y=y,sr=sr))),
            'rms': float(np.mean(librosa.feature.rms(y=y))),
            'spectral_centroid': float(np.mean(librosa.feature.spectral_centroid(y=y,sr=sr))),
            'spectral_bandwidth': float(np.mean(librosa.feature.spectral_bandwidth(y=y,sr=sr))),
            'rolloff': float(np.mean(librosa.feature.spectral_rolloff(y=y,sr=sr))),
            'zero_crossing_rate': float(np.mean(librosa.feature.zero_crossing_rate(y))),
        }
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        for i in range(20): feats[f'mfcc{i+1}'] = float(np.mean(mfcc[i]))
        X = scaler.transform(np.array(list(feats.values())).reshape(1,-1))
        tensor = torch.tensor(X, dtype=torch.float32).to(device)
        with torch.no_grad():
            probs = F.softmax(audio_model(tensor), dim=-1).cpu().numpy()[0]
        fp = float(probs[1])
        v = "FAKE" if fp >= 0.5 else "REAL"
        c = "HIGH" if abs(fp-0.5)>0.35 else "MEDIUM" if abs(fp-0.5)>0.15 else "LOW"
        return {"verdict":v,"confidence":c,"fake_probability":round(fp,4),"real_probability":round(float(probs[0]),4),"modality":"audio","latency_ms":round((time.time()-t0)*1000,1)}
    finally:
        os.unlink(tmp)

def gauge_chart(fake_prob):
    pct = fake_prob * 100
    color = f"rgba(255,0,110,{0.6+fake_prob*0.4})" if fake_prob >= 0.5 else f"rgba(0,255,136,{0.6+(1-fake_prob)*0.4})"
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=round(pct,1),
        number={"suffix":"%","font":{"family":"Orbitron","size":40,"color":color}},
        title={"text":"DEEPFAKE PROBABILITY","font":{"family":"Orbitron","size":11,"color":"rgba(192,207,224,0.5)"}},
        gauge={"axis":{"range":[0,100]},"bar":{"color":color,"thickness":0.25},"bgcolor":"rgba(0,0,0,0)","bordercolor":"rgba(0,0,0,0)",
               "steps":[{"range":[0,35],"color":"rgba(0,255,136,0.08)"},{"range":[35,65],"color":"rgba(255,165,0,0.06)"},{"range":[65,100],"color":"rgba(255,0,110,0.08)"}],
               "threshold":{"line":{"color":"rgba(255,255,255,0.3)","width":2},"thickness":0.8,"value":50}}))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",height=260,margin=dict(l=20,r=20,t=50,b=20))
    return fig

def prob_bars(fake_p, real_p):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=["AUTHENTIC","DEEPFAKE"],y=[real_p*100,fake_p*100],
        marker=dict(color=["rgba(0,255,136,0.7)","rgba(255,0,110,0.7)"],line=dict(color=["rgba(0,255,136,1)","rgba(255,0,110,1)"],width=1)),
        text=[f"{real_p*100:.1f}%",f"{fake_p*100:.1f}%"],textposition="outside",textfont={"family":"Orbitron","size":11},width=0.5))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(6,12,20,0.5)",height=220,
        margin=dict(l=10,r=10,t=20,b=10),xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True,gridcolor="rgba(255,255,255,0.04)",range=[0,115],ticksuffix="%"),
        bargap=0.4,showlegend=False,font={"color":"rgba(192,207,224,0.5)","size":9})
    return fig

def render_result(result):
    verdict = result.get("verdict","?")
    conf = result.get("confidence","?")
    fake_p = result.get("fake_probability",0.5)
    real_p = result.get("real_probability",0.5)
    modality = result.get("modality","?").upper()
    latency = result.get("latency_ms","?")
    is_fake = verdict == "FAKE"
    conf_color = {"HIGH":"#00ff88","MEDIUM":"#ffa500","LOW":"#ff6b6b"}.get(conf,"#888")
    st.markdown(f'''<div class="{'verdict-fake' if is_fake else 'verdict-real'}">
        <div class="{'verdict-label-fake' if is_fake else 'verdict-label-real'}">{'⚠ DEEPFAKE DETECTED' if is_fake else '✓ AUTHENTIC MEDIA'}</div>
        <div class="verdict-sub" style="color:{conf_color};">CONFIDENCE: {conf} &nbsp;|&nbsp; {modality} &nbsp;|&nbsp; {latency} ms</div>
    </div>''', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1: st.plotly_chart(gauge_chart(fake_p), use_container_width=True, key="g")
    with c2: st.plotly_chart(prob_bars(fake_p, real_p), use_container_width=True, key="b")
    st.markdown(f'''<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;margin-top:.5rem;">
        <div class="stat-box"><div class="stat-value">{fake_p*100:.1f}%</div><div class="stat-label">Fake Score</div></div>
        <div class="stat-box"><div class="stat-value" style="color:{'#ff006e' if is_fake else '#00ff88'};">{verdict}</div><div class="stat-label">Verdict</div></div>
        <div class="stat-box"><div class="stat-value" style="color:{conf_color};">{conf}</div><div class="stat-label">Confidence</div></div>
    </div>''', unsafe_allow_html=True)
    if "detail" in result:
        with st.expander("◈ MODEL BREAKDOWN"):
            d = result["detail"]
            st.markdown(f"""
            <div style="font-family:'Share Tech Mono',monospace;font-size:.8rem;color:rgba(192,207,224,.6);line-height:2;">
            Face-Swap Score: &nbsp;<b style="color:var(--neon-cyan);">{d.get('faceswap_score',0)*100:.1f}%</b><br>
            GAN/AI Score: &nbsp;&nbsp;&nbsp;<b style="color:var(--neon-cyan);">{d.get('gan_score',0)*100:.1f}%</b><br>
            Fusion: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b style="color:var(--neon-cyan);">Dual Model Max</b>
            </div>
            """, unsafe_allow_html=True)

# ── Load models ───────────────────────────────────────────────────────────────
with st.spinner("🧠 Loading neural networks..."):
    try:
        faceswap_model, gan_model, audio_model, scaler, device = load_models()
        models_loaded = True
    except Exception as e:
        st.error(f"Model load error: {e}")
        models_loaded = False

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">DEEPFAKE.AI</div><div class="hero-sub">◈ Neural Truth Engine v1.0 ◈ Real-Time Detection ◈</div>', unsafe_allow_html=True)

# Model status
c1,c2,c3 = st.columns(3)
for col, icon, name, acc, loaded in [
    (c1,"🖼","FACE-SWAP","95.8%", models_loaded),
    (c2,"🤖","GAN DETECTOR","98.1%", models_loaded and gan_model is not None),
    (c3,"🎵","AUDIO","99.6%", models_loaded),
]:
    with col:
        sc = "#00ff88" if loaded else "#ff006e"
        st.markdown(f'<div class="glass-card" style="padding:1rem;"><div style="font-family:\'Orbitron\',monospace;font-size:.7rem;color:rgba(192,207,224,.4);">{icon} {name}</div><div style="font-family:\'Orbitron\',monospace;font-size:1.1rem;color:{sc};text-shadow:0 0 10px {sc};">{acc} {"✓" if loaded else "✗"}</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Main UI ───────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["◈ ANALYZE", "◈ INTEL"])

with tab1:
    st.markdown('<div class="section-header">◈ Multi-Modal Analysis</div>', unsafe_allow_html=True)

    # VIDEO SECTION
    st.markdown('<div style="font-family:\'Orbitron\',monospace;font-size:.9rem;letter-spacing:.15em;color:var(--neon-cyan);margin-top:1.5rem;margin-bottom:1rem;">🎬 VIDEO INPUT</div>', unsafe_allow_html=True)
    video_col1, video_col2 = st.columns([1,1], gap="large")
    with video_col1:
        st.markdown('<div style="font-family:\'Share Tech Mono\',monospace;font-size:.75rem;color:rgba(192,207,224,.4);margin-bottom:1rem;">SUPPORTED: MP4 · AVI · MOV · MKV · WEBM</div>', unsafe_allow_html=True)
        uploaded_video = st.file_uploader("Drop video file", type=["mp4","avi","mov","mkv","webm"], label_visibility="collapsed", key="video_uploader")
        if uploaded_video:
            st.video(uploaded_video)
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("◈ RUN VIDEO ANALYSIS", use_container_width=True, key="video_analyze"):
                if not models_loaded:
                    st.error("Models not loaded!")
                else:
                    with st.spinner("🧠 Analyzing video..."):
                        st.error("Video detection not yet implemented in standalone mode. Use API mode.")
    with video_col2:
        st.markdown('<div class="section-header">◈ Video Analysis Output</div>', unsafe_allow_html=True)
        if "video_result" in st.session_state:
            render_result(st.session_state["video_result"])
        else:
            st.markdown('<div style="text-align:center;padding:3rem;border:1px dashed rgba(0,245,255,.1);border-radius:16px;"><div style="font-size:2.5rem;margin-bottom:1rem;opacity:.3;">🎬</div><div style="font-family:\'Orbitron\',monospace;font-size:.75rem;letter-spacing:.2em;color:rgba(0,245,255,.3);">AWAITING VIDEO</div></div>', unsafe_allow_html=True)

    st.markdown("<hr style='margin:2rem 0;'>", unsafe_allow_html=True)

    # IMAGE SECTION
    st.markdown('<div style="font-family:\'Orbitron\',monospace;font-size:.9rem;letter-spacing:.15em;color:var(--neon-cyan);margin-bottom:1rem;">🖼 IMAGE INPUT</div>', unsafe_allow_html=True)
    image_col1, image_col2 = st.columns([1,1], gap="large")
    with image_col1:
        st.markdown('<div style="font-family:\'Share Tech Mono\',monospace;font-size:.75rem;color:rgba(192,207,224,.4);margin-bottom:1rem;">SUPPORTED: JPG · PNG · WEBP · BMP</div>', unsafe_allow_html=True)
        uploaded_image = st.file_uploader("Drop image file", type=["jpg","jpeg","png","webp","bmp"], label_visibility="collapsed", key="image_uploader")
        if uploaded_image:
            st.image(uploaded_image, use_container_width=True)
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("◈ RUN IMAGE ANALYSIS", use_container_width=True, key="image_analyze"):
                if not models_loaded:
                    st.error("Models not loaded!")
                else:
                    with st.spinner("🧠 Analyzing..."):
                        try:
                            result = detect_image(uploaded_image, faceswap_model, gan_model, device)
                            st.session_state["image_result"] = result
                        except Exception as e:
                            st.error(f"Error: {e}")
    with image_col2:
        st.markdown('<div class="section-header">◈ Image Analysis Output</div>', unsafe_allow_html=True)
        if "image_result" in st.session_state:
            render_result(st.session_state["image_result"])
        else:
            st.markdown('<div style="text-align:center;padding:3rem;border:1px dashed rgba(0,245,255,.1);border-radius:16px;"><div style="font-size:2.5rem;margin-bottom:1rem;opacity:.3;">🖼</div><div style="font-family:\'Orbitron\',monospace;font-size:.75rem;letter-spacing:.2em;color:rgba(0,245,255,.3);">AWAITING IMAGE</div></div>', unsafe_allow_html=True)

    st.markdown("<hr style='margin:2rem 0;'>", unsafe_allow_html=True)

    # AUDIO SECTION
    st.markdown('<div style="font-family:\'Orbitron\',monospace;font-size:.9rem;letter-spacing:.15em;color:var(--neon-cyan);margin-bottom:1rem;">🎵 AUDIO INPUT</div>', unsafe_allow_html=True)
    audio_col1, audio_col2 = st.columns([1,1], gap="large")
    with audio_col1:
        st.markdown('<div style="font-family:\'Share Tech Mono\',monospace;font-size:.75rem;color:rgba(192,207,224,.4);margin-bottom:1rem;">SUPPORTED: WAV · MP3 · FLAC · M4A</div>', unsafe_allow_html=True)
        uploaded_audio = st.file_uploader("Drop audio file", type=["wav","mp3","flac","m4a"], label_visibility="collapsed", key="audio_uploader")
        if uploaded_audio:
            st.audio(uploaded_audio)
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("◈ RUN AUDIO ANALYSIS", use_container_width=True, key="audio_analyze"):
                if not models_loaded:
                    st.error("Models not loaded!")
                else:
                    with st.spinner("🧠 Analyzing..."):
                        try:
                            result = detect_audio(uploaded_audio, audio_model, scaler, device)
                            st.session_state["audio_result"] = result
                        except Exception as e:
                            st.error(f"Error: {e}")
    with audio_col2:
        st.markdown('<div class="section-header">◈ Audio Analysis Output</div>', unsafe_allow_html=True)
        if "audio_result" in st.session_state:
            render_result(st.session_state["audio_result"])
        else:
            st.markdown('<div style="text-align:center;padding:3rem;border:1px dashed rgba(0,245,255,.1);border-radius:16px;"><div style="font-family:\'Orbitron\',monospace;font-size:.75rem;letter-spacing:.2em;color:rgba(0,245,255,.3);">AWAITING AUDIO</div></div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="glass-card"><div style="font-family:\'Orbitron\',monospace;font-size:.8rem;color:var(--neon-pink);letter-spacing:.15em;margin-bottom:1rem;">⚠ MODEL LIMITATIONS</div><div style="font-family:\'Share Tech Mono\',monospace;font-size:.78rem;color:rgba(192,207,224,.55);line-height:1.9;">IMAGE MODEL: Trained on face-swap deepfakes. Detects facial manipulation artifacts.<br>GAN DETECTOR: Detects AI-generated faces — StyleGAN, Stable Diffusion, DALL-E.<br>AUDIO MODEL: Detects voice cloning and TTS synthesis (99.6% accuracy).<br>NOTE: Results may vary on unseen deepfake generators.</div></div>', unsafe_allow_html=True)