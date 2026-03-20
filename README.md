<div align="center">

```
██████╗ ███████╗███████╗██████╗ ███████╗ █████╗ ██╗  ██╗███████╗
██╔══██╗██╔════╝██╔════╝██╔══██╗██╔════╝██╔══██╗██║ ██╔╝██╔════╝
██║  ██║█████╗  █████╗  ██████╔╝█████╗  ███████║█████╔╝ █████╗  
██║  ██║██╔══╝  ██╔══╝  ██╔═══╝ ██╔══╝  ██╔══██║██╔═██╗ ██╔══╝  
██████╔╝███████╗███████╗██║     ██║     ██║  ██║██║  ██╗███████╗
╚═════╝ ╚══════╝╚══════╝╚═╝     ╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝


 ██████╗  ██████╗ █████╗ ███╗   ██╗███╗  ██╗███████╗██████╗
██╔════╝ ██╔════╝██╔══██╗████╗  ██║████╗ ██║██╔════╝██╔══██╗
╚█████╗  ██║     ███████║██╔██╗ ██║██╔██╗██║█████╗  ██████╔╝
 ╚═══██╗ ██║     ██╔══██║██║╚██╗██║██║╚████║██╔══╝  ██╔══██╗
██████╔╝ ╚██████╗██║  ██║██║ ╚████║██║ ╚███║███████╗██║  ██║
╚═════╝   ╚═════╝╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝  ╚══╝╚══════╝╚═╝  ╚═╝
```

### ✦ Real-Time AI-Powered Deepfake Detection System ✦
#### Detect Manipulated Images, Videos & Cloned Voices — Powered by Deep Learning

<br>

[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-Visit%20Now-00f5ff?style=for-the-badge&labelColor=0d1117)](https://kansal0920-deepfake-scanner.hf.space)
[![GitHub Repo](https://img.shields.io/badge/⭐%20GitHub-Star%20Repo-ff006e?style=for-the-badge&labelColor=0d1117)](https://github.com/BhavyaKansal20/DeepFake-Detector)

<br>

![Python](https://img.shields.io/badge/Python_3.11-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch_2.2-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit_1.55-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI_0.109-009688?style=flat-square&logo=fastapi&logoColor=white)
![EfficientNet](https://img.shields.io/badge/EfficientNet--B0-timm-blueviolet?style=flat-square)
![Apple MPS](https://img.shields.io/badge/Apple_Silicon-MPS-999999?style=flat-square&logo=apple&logoColor=white)

<br>

> **"Is That Face Real? Is That Voice Cloned? Our Neural Truth Engine Knows."**

</div>

---

## ⚡ At a Glance

| 🖼️ Face-Swap Detection | 🤖 GAN/AI Image Detection | 🎵 Voice Clone Detection | ⚡ Real-Time | 🌐 Public API |
|---|---|---|---|---|
| EfficientNet-B0 | EfficientNet-B0 | AudioMLP + Spectral | < 200ms | FastAPI REST |
| **95.8%** Accuracy | **98.1%** AUC-ROC | **99.6%** Accuracy | MPS / CPU | JSON Response |
| FaceForensics++ | CIFAKE Dataset | WaveFake Dataset | Dual-Model Fusion | Swagger UI |

---

## 🏗️ System Architecture

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                    🔬  DEEPFAKE SCANNER  —  SYSTEM ARCHITECTURE             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║   ┌─────────────┐     HTTPS      ┌──────────────────────────────────────┐   ║
║   │   🌐 USER   │ ─────────────► │         STREAMLIT FRONTEND           ║   ║
║   │   BROWSER   │                │                                      ║   ║
║   └─────────────┘                │  ┌────────────┐  ┌────────────────┐  ║   ║
║                                  │  │  File      │  │  Result        │  ║   ║
║   ┌─────────────┐                │  │  Upload    │  │  Dashboard     │  ║   ║
║   │  📱 MOBILE  │ ─────────────► │  │  Handler   │  │  Neon UI       │  ║   ║
║   │             │                │  └────────────┘  └────────────────┘  ║   ║
║   └─────────────┘                └──────────────┬───────────────────────┘   ║
║                                                 │ HTTP POST /detect         ║
║              ┌──────────────────────────────────▼──────────────────────┐   ║
║              │               FastAPI Backend  :8000                     ║   ║
║              │  POST /detect  POST /detect/image  POST /detect/audio    ║   ║
║              │  POST /detect/video  GET /health   GET /docs             ║   ║
║              └───────┬───────────────────┬────────────────┬────────────┘   ║
║                      │                   │                │                ║
║              ┌───────▼───────┐  ┌────────▼──────┐  ┌─────▼──────────┐   ║
║              │ 🧠 FACE-SWAP  │  │  🤖 GAN/AI    │  │  🔊 AUDIO MLP  ║   ║
║              │ EfficientNet  │  │ EfficientNet  │  │  26 Spectral   ║   ║
║              │ B0 · 95.8%    │  │ B0 · 98.1%AUC │  │  Features MLP  ║   ║
║              │ 140k Faces    │  │ CIFAKE 60k    │  │  WaveFake 11k  ║   ║
║              └───────┬───────┘  └────────┬──────┘  └─────┬──────────┘   ║
║                      │                   │                │                ║
║              ┌────────────────────────────────────────────────────────┐   ║
║              │               🔀 FUSION LAYER                           ║   ║
║              │   Images: max(fs,gan) if >0.8  else  0.45·fs + 0.55·gan ║   ║
║              │   Video:  mean(12 frames) × 0.7  +  audio × 0.3         ║   ║
║              └──────────────────────────────────────────────────────┘   ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

---

## 🤖 ML Pipeline

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                         ML PIPELINE — END TO END                            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  RAW DATA              PREPROCESSING            MODEL                OUTPUT  ║
║  ─────────            ───────────────          ──────────────        ──────  ║
║  140k Faces ────────► Resize 224×224 ─────►  EfficientNet-B0  ──►  .pt     ║
║  Real+Fake           Normalize+Augment        Face-Swap Detector      ↓     ║
║                                                                     FastAPI  ║
║  CIFAKE     ────────► Resize 224×224 ─────►  EfficientNet-B0  ──►  /detect ║
║  60k Images          Normalize+Augment        GAN/AI Detector               ║
║                                                                              ║
║  WaveFake   ────────► MFCC (20 coeff)  ────► AudioMLP          ──►  .pt     ║
║  11k clips           + 6 Spectral Feat.       26-dim input          + .pkl   ║
║                      StandardScaler                               (scaler)   ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

---

## 📊 Model Performance

```
  FACE-SWAP DETECTOR                    GAN/AI IMAGE DETECTOR
  ──────────────────                    ─────────────────────
  Accuracy   ███████████████████░  95.8%  Accuracy  █████████████████░░░  93.5%
  ROC-AUC    ████████████████████  99.3%  ROC-AUC   ███████████████████░  98.1%
  Dataset    ▓▓▓▓▓ 140k Faces            Dataset   ▓▓▓▓▓▓ CIFAKE 60k

  VOICE CLONE DETECTOR
  ────────────────────
  Accuracy   ████████████████████  99.6%
  ROC-AUC    ████████████████████  99.99%
  Dataset    ▓▓▓▓ WaveFake 11,778 clips
```

| Metric | 🖼️ Face-Swap | 🤖 GAN Detector | 🎵 Voice Clone |
|---|---|---|---|
| **Accuracy** | `95.8%` | `93.5%` | `99.6%` |
| **ROC-AUC** | `0.9926` | `0.9810` | `0.9999` |
| **Algorithm** | EfficientNet-B0 | EfficientNet-B0 | AudioMLP |
| **Training Samples** | 20,000 | 20,000 | 11,778 |
| **Inference Time** | `< 150ms` | `< 150ms` | `< 50ms` |

---

## 🔮 Detection Flow

```
                         USER UPLOADS FILE
                               │
                    ┌──────────┴──────────┐
                    │                     │
                    ▼                     ▼
              IMAGE / VIDEO           AUDIO FILE
                    │                     │
                    ▼                     ▼
       ┌────────────────────┐   ┌──────────────────────┐
       │  Face-Swap Model   │   │  Feature Extraction  │
       │  EfficientNet-B0   │   │  MFCC × 20           │
       │  → faceswap_score  │   │  + 6 Spectral Feats  │
       └────────┬───────────┘   └──────────┬───────────┘
                │                          │
                ▼                          ▼
       ┌────────────────────┐   ┌──────────────────────┐
       │  GAN Detector      │   │  AudioMLP Classifier  │
       │  EfficientNet-B0   │   │  StandardScaler       │
       │  → gan_score       │   │  → fake_probability   │
       └────────┬───────────┘   └──────────┬───────────┘
                │                          │
                ▼                          │
       ┌────────────────────┐              │
       │  Dual Model Fusion │              │
       │  max(fs,gn)>0.8 ?  │              │
       │  else weighted avg │              │
       └────────┬───────────┘              │
                └──────────┬───────────────┘
                           │
             ┌─────────────┼─────────────┐
             ▼             ▼             ▼
       prob < 0.5     0.5–0.65      prob > 0.65
     ✅ REAL         ⚠️ SUSPICIOUS  🚨 FAKE
                           │
                  Confidence: HIGH / MEDIUM / LOW
```

---

## ✨ Features

<details>
<summary><b>🖼️ Face-Swap Deepfake Detection</b></summary>

Detects **face replacement** in images and video frames using a fine-tuned EfficientNet-B0.

**Training Data:** 140,000 Real & Fake Face images (Kaggle)

| Deepfake Type | Description |
|---|---|
| Face2Face | Facial expression transfer |
| FaceSwap | Identity swap between subjects |
| Deepfakes | Neural face replacement |
| NeuralTextures | Texture-based manipulation |

Looks for blending artifacts, lighting inconsistencies, and unnatural facial boundaries at pixel level.
</details>

<details>
<summary><b>🤖 GAN / AI-Generated Image Detection</b></summary>

Detects **fully AI-generated faces** from diffusion models and GANs.

**Training Data:** CIFAKE dataset — 60,000 real photos vs Stable Diffusion generated images

| Generator | Type |
|---|---|
| StyleGAN2/3 | thispersondoesnotexist.com |
| Stable Diffusion | AI art generators |
| DALL-E | OpenAI image generation |
| Midjourney | AI image synthesis |
| Gemini Imagen | Google AI images |

Learns frequency-domain artifacts and texture patterns unique to neural synthesis.
</details>

<details>
<summary><b>🎵 Voice Clone / TTS Detection</b></summary>

Detects **synthesised and cloned voices** using spectral feature analysis.

**Training Data:** WaveFake dataset — 11,778 real & AI voice samples

**Input Features (26-dim vector):**

| Feature | Description |
|---|---|
| `chroma_stft` | Chromagram energy |
| `rms` | Root mean square energy |
| `spectral_centroid` | Frequency centre of mass |
| `spectral_bandwidth` | Frequency spread |
| `rolloff` | High-frequency roll-off |
| `zero_crossing_rate` | Signal sign changes |
| `mfcc_1–20` | Mel-frequency cepstral coefficients |

</details>

<details>
<summary><b>🔀 Dual-Model Fusion Engine</b></summary>

For image inputs, both the Face-Swap Detector and GAN Detector run simultaneously:

```python
# If either model is very confident → trust it
if gan_score > 0.8 or faceswap_score > 0.8:
    final = max(faceswap_score, gan_score)
else:
    # Weighted average (GAN gets slightly more weight)
    final = faceswap_score * 0.45 + gan_score * 0.55
```

Catches **both** face-swapped videos AND fully AI-generated faces in one pass.

For video: `final = 0.70 × video_score + 0.30 × audio_score`
</details>

---

## 🌐 API Reference

```
  PUBLIC ENDPOINTS
  ────────────────
  GET  /health              System health + loaded models status
  POST /detect              Auto-detect file type → run appropriate model
  POST /detect/image        Image-only deepfake detection
  POST /detect/audio        Audio-only voice clone detection
  POST /detect/video        Video deepfake + optional audio fusion
  GET  /docs                Swagger UI (interactive API explorer)
```

**Request:**
```bash
curl -X POST http://localhost:8000/detect \
  -F "file=@your_image.jpg"
```

**Response:**
```json
{
  "verdict": "FAKE",
  "confidence": "HIGH",
  "fake_probability": 0.9731,
  "real_probability": 0.0269,
  "modality": "image",
  "latency_ms": 143.2,
  "detail": {
    "faceswap_score": 0.1823,
    "gan_score": 0.9731,
    "fusion": "dual_model"
  }
}
```

---

## 🗂️ Project Structure

```
DeepFake-Detector/
│
├── 📄 api_server.py             ← FastAPI backend  (python3 api_server.py)
├── 📄 start.py                  ← Render / HF start script
├── 📋 requirements.txt          ← Full dependencies
│
├── 📂 src/
│   ├── models/
│   │   ├── image_model.py       ← EfficientNet-B0 face-swap detector
│   │   ├── video_model.py       ← EfficientNet + BiLSTM video detector
│   │   └── audio_model.py       ← AudioMLP (26-dim spectral features)
│   ├── train/
│   │   ├── train_image.py       ← Image model training pipeline
│   │   ├── train_video.py       ← Video model training pipeline
│   │   ├── train_audio.py       ← Audio model training pipeline
│   │   └── train_audio_wavefake.py ← WaveFake-specific trainer
│   ├── preprocessing/
│   │   ├── extract_frames.py    ← Video → frames → face crops
│   │   └── audio_features.py   ← Audio → MFCC/mel/LFCC features
│   ├── inference/
│   │   └── detector.py          ← Unified inference engine
│   ├── fusion/
│   │   └── ensemble.py          ← Weighted avg / voting / meta-clf
│   └── utils/
│       └── helpers.py           ← Seeds, metrics, checkpoints
│
├── 📂 models/                   ← Trained weights (not in repo → download below)
│   ├── image_model/
│   │   └── best_model.pt        ← Face-swap detector weights
│   ├── gan_detector/
│   │   └── best_model.pt        ← GAN/AI image detector weights
│   └── audio_model/
│       ├── best_model.pt        ← Voice clone detector weights
│       └── feature_scaler.pkl   ← StandardScaler for audio features
│
├── 📂 ui/
│   └── app.py                   ← Neon glassmorphism Streamlit UI
│
├── 📂 configs/
│   └── config.yaml              ← All hyperparameters & paths
│
├── 📂 scripts/
│   ├── download_datasets.py     ← Kaggle dataset downloader
│   ├── train_all.py             ← Train all 3 models sequentially
│   └── docker_start.sh          ← Docker entrypoint
│
├── 📄 Dockerfile
├── 📄 docker-compose.yml
└── 📋 requirements.txt
```

---

## ⚙️ Setup & Run Locally

### Prerequisites
```bash
python --version   # Python 3.11+
brew install ffmpeg  # macOS — required for video audio extraction
# sudo apt install ffmpeg  # Ubuntu/Linux
```

### Step 1 — Clone
```bash
git clone https://github.com/BhavyaKansal20/DeepFake-Detector.git
cd DeepFake-Detector
```

### Step 2 — Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate       # Linux/macOS
# venv\Scripts\activate         # Windows
```

### Step 3 — Install Dependencies
```bash
# PyTorch (Apple Silicon MPS)
pip install torch torchvision torchaudio

# Remaining dependencies
pip install -r requirements.txt
```

### Step 4 — Download Datasets & Train
```bash
# Set up Kaggle API key at kaggle.com/settings → API → Create New Token
mkdir -p ~/.kaggle
# paste token into ~/.kaggle/kaggle.json

# Download datasets
kaggle datasets download -d xhlulu/140k-real-and-fake-faces         -p data/raw/images/ --unzip
kaggle datasets download -d birdy654/cifake-real-and-ai-generated-synthetic-images  -p data/raw/images/cifake/ --unzip
kaggle datasets download -d birdy654/deep-voice-deepfake-voice-recognition          -p data/raw/audio/ --unzip

# Train all models (~3 hours on Apple M4 / ~2 hours on A100)
python scripts/train_all.py
```

### Step 5 — Launch
```bash
# Terminal 1 — Start API backend
export PYTHONPATH=$(pwd)
python3 api_server.py
# → Running at http://localhost:8000
# → Swagger UI at http://localhost:8000/docs

# Terminal 2 — Start Streamlit UI
streamlit run app.py
# → Running at http://localhost:8501
```

---

## 🚀 Deploy to Hugging Face Spaces

The fastest way to get this live publicly for free:

### Step 1 — Create a Space
1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. SDK: **Streamlit**
3. Visibility: **Public**

### Step 2 — Upload Model Weights
Upload your `models/` folder to the Space files (or use Git LFS):
```bash
git lfs install
git lfs track "*.pt" "*.pkl"
git add .gitattributes
git add models/
git commit -m "Add model weights"
git push
```

### Step 3 — Push Code
```bash
git remote add space https://huggingface.co/spaces/YOUR_USERNAME/deepfake-scanner
git push space main
```

### Step 4 — Done
Your Space will auto-deploy. The `app.py` at root starts the FastAPI backend in a background thread and serves the Streamlit UI as the main entry point.

> **Note:** HF Free tier has limited RAM (~16GB). The image + audio models load fine. For video processing, increase hardware if needed.

---

## 🐳 Docker (Recommended for Production)

```bash
# Build
docker build -t deepfake-detector .

# Run (GPU)
docker run --gpus all -p 8000:8000 -p 8501:8501 \
  -v $(pwd)/models:/app/models deepfake-detector

# Or with docker-compose
docker-compose up
```

---

## 🧰 Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Language** | Python 3.11 | Core runtime |
| **Deep Learning** | PyTorch 2.2 | Model training & inference |
| **CV Backbone** | timm (EfficientNet-B0) | Pretrained ImageNet weights |
| **Audio** | librosa | MFCC & spectral feature extraction |
| **Backend** | FastAPI 0.109 | REST API, file upload, Swagger UI |
| **Frontend** | Streamlit 1.55 | Neon glassmorphism dashboard |
| **Acceleration** | Apple MPS / CUDA | GPU-accelerated inference |
| **Containerization** | Docker + docker-compose | Reproducible deployment |
| **Deployment** | Render · Hugging Face Spaces | Cloud hosting |

---

## 🗂️ Datasets Used

| Dataset | Modality | Size | Source |
|---|---|---|---|
| 140k Real & Fake Faces | Image | 140,000 images | Kaggle |
| CIFAKE | Image | 60,000 images | Kaggle |
| WaveFake | Audio | 11,778 clips | Kaggle |
| FaceForensics++ | Video | 1,000+ videos | [Request Access](https://github.com/ondyari/FaceForensics) |
| DFDC (Facebook) | Video | 100,000+ videos | Kaggle Competition |
| ASVspoof 2019 | Audio | 121,461 clips | Kaggle Mirror |

---

## ⚠️ Model Limitations

```
  IMAGE MODELS
  ─────────────
  Face-Swap Detector → Works best on: FaceSwap, Face2Face, Deepfake videos
                       May miss: fully AI-generated images (GAN detector handles those)

  GAN Detector       → Works best on: StyleGAN, DALL-E, Midjourney, Gemini
                       May miss: novel, unseen generator architectures

  AUDIO MODEL
  ────────────
  Voice Detector     → Works best on: ElevenLabs, Murf, standard TTS systems
                       May miss: very high-quality unseen voice clones

  GENERAL NOTE
  ─────────────
  All models may perform differently on out-of-distribution samples.
  Results are probabilistic — use as supporting evidence, not sole proof.
```

---

## 🔬 Key Technical Decisions

| Choice | Reason |
|---|---|
| EfficientNet-B0 over ResNet | Better accuracy/compute trade-off; stronger GAN artifact detection |
| AudioMLP over wav2vec2 | Lightweight 26-dim features; no heavy transformer required; 99.6% accuracy |
| Dual-model fusion for images | Catches both face-swapped AND fully AI-generated in one API call |
| 12-frame video sampling | Fast inference without sacrificing accuracy across temporal dimension |
| Focal Loss for images | Class imbalance common in real datasets; down-weights easy negatives |
| Mixed precision (AMP) | 2× speedup on modern GPUs with no accuracy loss |
| StandardScaler for audio | Essential — raw spectral features span vastly different value ranges |

---

## 👨‍💻 Author

```
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   👤  Bhavya Kansal                                          ║
║   🎓  AI Engineer · DeepTech Developer                       ║
║   🏢  Founder & CEO — Multimodex AI                          ║
║   🎓  Diploma CSE → B.Tech AI/ML (TIET, Patiala)            ║
║   🔬  AI/ML Industrial Trainee — NIELIT Ropar × IIT Ropar   ║
║   🌐  bhavyakansal.dev                                       ║
║   📧  kansalbhavya27@gmail.com                               ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

[![Portfolio](https://img.shields.io/badge/🌐%20Portfolio-bhavyakansal.dev-00f5ff?style=for-the-badge&labelColor=0d1117)](https://bhavyakansal.dev)
[![GitHub](https://img.shields.io/badge/GitHub-BhavyaKansal20-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/BhavyaKansal20)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-kansal0920-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/kansal0920)
[![Multimodex AI](https://img.shields.io/badge/🧠%20Multimodex%20AI-Platform-ff006e?style=for-the-badge&labelColor=0d1117)](https://multimodexai.vercel.app)

---

## ⭐ Support

If **DeepFake Scanner** helped you or impressed you:

```
  1. ⭐ Star this repository
  2. 🍴 Fork and build on it
  3. 📣 Share with your network
  4. 🐛 Open issues / PRs
  5. 🔗 Share the live demo link
```

Every star helps this project reach more developers and researchers. 🙏

---

<div align="center">

```
  ╔══════════════════════════════════════════════════════╗
  ║     🔬  D E E P F A K E   S C A N N E R             ║
  ║     Multimodex AI  ·  © 2026 Bhavya Kansal           ║
  ║     Built with ❤️  for a safer digital world          ║
  ╚══════════════════════════════════════════════════════╝
```

[![Live Demo](https://img.shields.io/badge/🚀%20Try%20It%20Live-deepfake--scanner.onrender.com-00f5ff?style=for-the-badge)](https://deepfake-scanner.onrender.com)

</div>![Apple MPS](https://img.shields.io/badge/Apple_Silicon-MPS-999999?style=flat-square&logo=apple&logoColor=white)

<br>

> **"Is That Face Real? Is That Voice Cloned? Our Neural Truth Engine Knows."**

</div>

---

## ⚡ At a Glance

| 🖼️ Face-Swap Detection | 🤖 GAN/AI Image Detection | 🎵 Voice Clone Detection | ⚡ Real-Time | 🌐 Public API |
|---|---|---|---|---|
| EfficientNet-B0 | EfficientNet-B0 | AudioMLP + Spectral | < 200ms | FastAPI REST |
| **95.8%** Accuracy | **98.1%** AUC-ROC | **99.6%** Accuracy | MPS / CPU | JSON Response |
| FaceForensics++ | CIFAKE Dataset | WaveFake Dataset | Dual-Model Fusion | Swagger UI |

---

## 🏗️ System Architecture

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                    🔬  DEEPFAKE SCANNER  —  SYSTEM ARCHITECTURE             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║   ┌─────────────┐     HTTPS      ┌──────────────────────────────────────┐   ║
║   │   🌐 USER   │ ─────────────► │         STREAMLIT FRONTEND           ║   ║
║   │   BROWSER   │                │                                      ║   ║
║   └─────────────┘                │  ┌────────────┐  ┌────────────────┐  ║   ║
║                                  │  │  File      │  │  Result        │  ║   ║
║   ┌─────────────┐                │  │  Upload    │  │  Dashboard     │  ║   ║
║   │  📱 MOBILE  │ ─────────────► │  │  Handler   │  │  Neon UI       │  ║   ║
║   │             │                │  └────────────┘  └────────────────┘  ║   ║
║   └─────────────┘                └──────────────┬───────────────────────┘   ║
║                                                 │ HTTP POST /detect         ║
║              ┌──────────────────────────────────▼──────────────────────┐   ║
║              │               FastAPI Backend  :8000                     ║   ║
║              │  POST /detect  POST /detect/image  POST /detect/audio    ║   ║
║              │  POST /detect/video  GET /health   GET /docs             ║   ║
║              └───────┬───────────────────┬────────────────┬────────────┘   ║
║                      │                   │                │                ║
║              ┌───────▼───────┐  ┌────────▼──────┐  ┌─────▼──────────┐   ║
║              │ 🧠 FACE-SWAP  │  │  🤖 GAN/AI    │  │  🔊 AUDIO MLP  ║   ║
║              │ EfficientNet  │  │ EfficientNet  │  │  26 Spectral   ║   ║
║              │ B0 · 95.8%    │  │ B0 · 98.1%AUC │  │  Features MLP  ║   ║
║              │ 140k Faces    │  │ CIFAKE 60k    │  │  WaveFake 11k  ║   ║
║              └───────┬───────┘  └────────┬──────┘  └─────┬──────────┘   ║
║                      │                   │                │                ║
║              ┌────────────────────────────────────────────────────────┐   ║
║              │               🔀 FUSION LAYER                           ║   ║
║              │   Images: max(fs,gan) if >0.8  else  0.45·fs + 0.55·gan ║   ║
║              │   Video:  mean(12 frames) × 0.7  +  audio × 0.3         ║   ║
║              └──────────────────────────────────────────────────────┘   ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

---

## 🤖 ML Pipeline

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                         ML PIPELINE — END TO END                            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  RAW DATA              PREPROCESSING            MODEL                OUTPUT  ║
║  ─────────            ───────────────          ──────────────        ──────  ║
║  140k Faces ────────► Resize 224×224 ─────►  EfficientNet-B0  ──►  .pt     ║
║  Real+Fake           Normalize+Augment        Face-Swap Detector      ↓     ║
║                                                                     FastAPI  ║
║  CIFAKE     ────────► Resize 224×224 ─────►  EfficientNet-B0  ──►  /detect ║
║  60k Images          Normalize+Augment        GAN/AI Detector               ║
║                                                                              ║
║  WaveFake   ────────► MFCC (20 coeff)  ────► AudioMLP          ──►  .pt     ║
║  11k clips           + 6 Spectral Feat.       26-dim input          + .pkl   ║
║                      StandardScaler                               (scaler)   ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

---

## 📊 Model Performance

```
  FACE-SWAP DETECTOR                    GAN/AI IMAGE DETECTOR
  ──────────────────                    ─────────────────────
  Accuracy   ███████████████████░  95.8%  Accuracy  █████████████████░░░  93.5%
  ROC-AUC    ████████████████████  99.3%  ROC-AUC   ███████████████████░  98.1%
  Dataset    ▓▓▓▓▓ 140k Faces            Dataset   ▓▓▓▓▓▓ CIFAKE 60k

  VOICE CLONE DETECTOR
  ────────────────────
  Accuracy   ████████████████████  99.6%
  ROC-AUC    ████████████████████  99.99%
  Dataset    ▓▓▓▓ WaveFake 11,778 clips
```

| Metric | 🖼️ Face-Swap | 🤖 GAN Detector | 🎵 Voice Clone |
|---|---|---|---|
| **Accuracy** | `95.8%` | `93.5%` | `99.6%` |
| **ROC-AUC** | `0.9926` | `0.9810` | `0.9999` |
| **Algorithm** | EfficientNet-B0 | EfficientNet-B0 | AudioMLP |
| **Training Samples** | 20,000 | 20,000 | 11,778 |
| **Inference Time** | `< 150ms` | `< 150ms` | `< 50ms` |

---

## 🔮 Detection Flow

```
                         USER UPLOADS FILE
                               │
                    ┌──────────┴──────────┐
                    │                     │
                    ▼                     ▼
              IMAGE / VIDEO           AUDIO FILE
                    │                     │
                    ▼                     ▼
       ┌────────────────────┐   ┌──────────────────────┐
       │  Face-Swap Model   │   │  Feature Extraction  │
       │  EfficientNet-B0   │   │  MFCC × 20           │
       │  → faceswap_score  │   │  + 6 Spectral Feats  │
       └────────┬───────────┘   └──────────┬───────────┘
                │                          │
                ▼                          ▼
       ┌────────────────────┐   ┌──────────────────────┐
       │  GAN Detector      │   │  AudioMLP Classifier  │
       │  EfficientNet-B0   │   │  StandardScaler       │
       │  → gan_score       │   │  → fake_probability   │
       └────────┬───────────┘   └──────────┬───────────┘
                │                          │
                ▼                          │
       ┌────────────────────┐              │
       │  Dual Model Fusion │              │
       │  max(fs,gn)>0.8 ?  │              │
       │  else weighted avg │              │
       └────────┬───────────┘              │
                └──────────┬───────────────┘
                           │
             ┌─────────────┼─────────────┐
             ▼             ▼             ▼
       prob < 0.5     0.5–0.65      prob > 0.65
     ✅ REAL         ⚠️ SUSPICIOUS  🚨 FAKE
                           │
                  Confidence: HIGH / MEDIUM / LOW
```

---

## ✨ Features

<details>
<summary><b>🖼️ Face-Swap Deepfake Detection</b></summary>

Detects **face replacement** in images and video frames using a fine-tuned EfficientNet-B0.

**Training Data:** 140,000 Real & Fake Face images (Kaggle)

| Deepfake Type | Description |
|---|---|
| Face2Face | Facial expression transfer |
| FaceSwap | Identity swap between subjects |
| Deepfakes | Neural face replacement |
| NeuralTextures | Texture-based manipulation |

Looks for blending artifacts, lighting inconsistencies, and unnatural facial boundaries at pixel level.
</details>

<details>
<summary><b>🤖 GAN / AI-Generated Image Detection</b></summary>

Detects **fully AI-generated faces** from diffusion models and GANs.

**Training Data:** CIFAKE dataset — 60,000 real photos vs Stable Diffusion generated images

| Generator | Type |
|---|---|
| StyleGAN2/3 | thispersondoesnotexist.com |
| Stable Diffusion | AI art generators |
| DALL-E | OpenAI image generation |
| Midjourney | AI image synthesis |
| Gemini Imagen | Google AI images |

Learns frequency-domain artifacts and texture patterns unique to neural synthesis.
</details>

<details>
<summary><b>🎵 Voice Clone / TTS Detection</b></summary>

Detects **synthesised and cloned voices** using spectral feature analysis.

**Training Data:** WaveFake dataset — 11,778 real & AI voice samples

**Input Features (26-dim vector):**

| Feature | Description |
|---|---|
| `chroma_stft` | Chromagram energy |
| `rms` | Root mean square energy |
| `spectral_centroid` | Frequency centre of mass |
| `spectral_bandwidth` | Frequency spread |
| `rolloff` | High-frequency roll-off |
| `zero_crossing_rate` | Signal sign changes |
| `mfcc_1–20` | Mel-frequency cepstral coefficients |

</details>

<details>
<summary><b>🔀 Dual-Model Fusion Engine</b></summary>

For image inputs, both the Face-Swap Detector and GAN Detector run simultaneously:

```python
# If either model is very confident → trust it
if gan_score > 0.8 or faceswap_score > 0.8:
    final = max(faceswap_score, gan_score)
else:
    # Weighted average (GAN gets slightly more weight)
    final = faceswap_score * 0.45 + gan_score * 0.55
```

Catches **both** face-swapped videos AND fully AI-generated faces in one pass.

For video: `final = 0.70 × video_score + 0.30 × audio_score`
</details>

---

## 🌐 API Reference

```
  PUBLIC ENDPOINTS
  ────────────────
  GET  /health              System health + loaded models status
  POST /detect              Auto-detect file type → run appropriate model
  POST /detect/image        Image-only deepfake detection
  POST /detect/audio        Audio-only voice clone detection
  POST /detect/video        Video deepfake + optional audio fusion
  GET  /docs                Swagger UI (interactive API explorer)
```

**Request:**
```bash
curl -X POST http://localhost:8000/detect \
  -F "file=@your_image.jpg"
```

**Response:**
```json
{
  "verdict": "FAKE",
  "confidence": "HIGH",
  "fake_probability": 0.9731,
  "real_probability": 0.0269,
  "modality": "image",
  "latency_ms": 143.2,
  "detail": {
    "faceswap_score": 0.1823,
    "gan_score": 0.9731,
    "fusion": "dual_model"
  }
}
```

---

## 🗂️ Project Structure

```
DeepFake-Detector/
│
├── 📄 api_server.py             ← FastAPI backend  (python3 api_server.py)
├── 📄 start.py                  ← Render / HF start script
├── 📋 requirements.txt          ← Full dependencies
│
├── 📂 src/
│   ├── models/
│   │   ├── image_model.py       ← EfficientNet-B0 face-swap detector
│   │   ├── video_model.py       ← EfficientNet + BiLSTM video detector
│   │   └── audio_model.py       ← AudioMLP (26-dim spectral features)
│   ├── train/
│   │   ├── train_image.py       ← Image model training pipeline
│   │   ├── train_video.py       ← Video model training pipeline
│   │   ├── train_audio.py       ← Audio model training pipeline
│   │   └── train_audio_wavefake.py ← WaveFake-specific trainer
│   ├── preprocessing/
│   │   ├── extract_frames.py    ← Video → frames → face crops
│   │   └── audio_features.py   ← Audio → MFCC/mel/LFCC features
│   ├── inference/
│   │   └── detector.py          ← Unified inference engine
│   ├── fusion/
│   │   └── ensemble.py          ← Weighted avg / voting / meta-clf
│   └── utils/
│       └── helpers.py           ← Seeds, metrics, checkpoints
│
├── 📂 models/                   ← Trained weights (not in repo → download below)
│   ├── image_model/
│   │   └── best_model.pt        ← Face-swap detector weights
│   ├── gan_detector/
│   │   └── best_model.pt        ← GAN/AI image detector weights
│   └── audio_model/
│       ├── best_model.pt        ← Voice clone detector weights
│       └── feature_scaler.pkl   ← StandardScaler for audio features
│
├── 📂 ui/
│   └── app.py                   ← Neon glassmorphism Streamlit UI
│
├── 📂 configs/
│   └── config.yaml              ← All hyperparameters & paths
│
├── 📂 scripts/
│   ├── download_datasets.py     ← Kaggle dataset downloader
│   ├── train_all.py             ← Train all 3 models sequentially
│   └── docker_start.sh          ← Docker entrypoint
│
├── 📄 Dockerfile
├── 📄 docker-compose.yml
└── 📋 requirements.txt
```

---

## ⚙️ Setup & Run Locally

### Prerequisites
```bash
python --version   # Python 3.11+
brew install ffmpeg  # macOS — required for video audio extraction
# sudo apt install ffmpeg  # Ubuntu/Linux
```

### Step 1 — Clone
```bash
git clone https://github.com/BhavyaKansal20/DeepFake-Detector.git
cd DeepFake-Detector
```

### Step 2 — Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate       # Linux/macOS
# venv\Scripts\activate         # Windows
```

### Step 3 — Install Dependencies
```bash
# PyTorch (Apple Silicon MPS)
pip install torch torchvision torchaudio

# Remaining dependencies
pip install -r requirements.txt
```

### Step 4 — Download Datasets & Train
```bash
# Set up Kaggle API key at kaggle.com/settings → API → Create New Token
mkdir -p ~/.kaggle
# paste token into ~/.kaggle/kaggle.json

# Download datasets
kaggle datasets download -d xhlulu/140k-real-and-fake-faces         -p data/raw/images/ --unzip
kaggle datasets download -d birdy654/cifake-real-and-ai-generated-synthetic-images  -p data/raw/images/cifake/ --unzip
kaggle datasets download -d birdy654/deep-voice-deepfake-voice-recognition          -p data/raw/audio/ --unzip

# Train all models (~3 hours on Apple M4 / ~2 hours on A100)
python scripts/train_all.py
```

### Step 5 — Launch
```bash
# Terminal 1 — Start API backend
export PYTHONPATH=$(pwd)
python3 api_server.py
# → Running at http://localhost:8000
# → Swagger UI at http://localhost:8000/docs

# Terminal 2 — Start Streamlit UI
streamlit run app.py
# → Running at http://localhost:8501
```

---

## 🚀 Deploy to Hugging Face Spaces

The fastest way to get this live publicly for free:

### Step 1 — Create a Space
1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. SDK: **Streamlit**
3. Visibility: **Public**

### Step 2 — Upload Model Weights
Upload your `models/` folder to the Space files (or use Git LFS):
```bash
git lfs install
git lfs track "*.pt" "*.pkl"
git add .gitattributes
git add models/
git commit -m "Add model weights"
git push
```

### Step 3 — Push Code
```bash
git remote add space https://huggingface.co/spaces/YOUR_USERNAME/deepfake-scanner
git push space main
```

### Step 4 — Done
Your Space will auto-deploy. The `app.py` at root starts the FastAPI backend in a background thread and serves the Streamlit UI as the main entry point.

> **Note:** HF Free tier has limited RAM (~16GB). The image + audio models load fine. For video processing, increase hardware if needed.

---

## 🐳 Docker (Recommended for Production)

```bash
# Build
docker build -t deepfake-detector .

# Run (GPU)
docker run --gpus all -p 8000:8000 -p 8501:8501 \
  -v $(pwd)/models:/app/models deepfake-detector

# Or with docker-compose
docker-compose up
```

---

## 🧰 Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Language** | Python 3.11 | Core runtime |
| **Deep Learning** | PyTorch 2.2 | Model training & inference |
| **CV Backbone** | timm (EfficientNet-B0) | Pretrained ImageNet weights |
| **Audio** | librosa | MFCC & spectral feature extraction |
| **Backend** | FastAPI 0.109 | REST API, file upload, Swagger UI |
| **Frontend** | Streamlit 1.55 | Neon glassmorphism dashboard |
| **Acceleration** | Apple MPS / CUDA | GPU-accelerated inference |
| **Containerization** | Docker + docker-compose | Reproducible deployment |
| **Deployment** | Render · Hugging Face Spaces | Cloud hosting |

---

## 🗂️ Datasets Used

| Dataset | Modality | Size | Source |
|---|---|---|---|
| 140k Real & Fake Faces | Image | 140,000 images | Kaggle |
| CIFAKE | Image | 60,000 images | Kaggle |
| WaveFake | Audio | 11,778 clips | Kaggle |
| FaceForensics++ | Video | 1,000+ videos | [Request Access](https://github.com/ondyari/FaceForensics) |
| DFDC (Facebook) | Video | 100,000+ videos | Kaggle Competition |
| ASVspoof 2019 | Audio | 121,461 clips | Kaggle Mirror |

---

## ⚠️ Model Limitations

```
  IMAGE MODELS
  ─────────────
  Face-Swap Detector → Works best on: FaceSwap, Face2Face, Deepfake videos
                       May miss: fully AI-generated images (GAN detector handles those)

  GAN Detector       → Works best on: StyleGAN, DALL-E, Midjourney, Gemini
                       May miss: novel, unseen generator architectures

  AUDIO MODEL
  ────────────
  Voice Detector     → Works best on: ElevenLabs, Murf, standard TTS systems
                       May miss: very high-quality unseen voice clones

  GENERAL NOTE
  ─────────────
  All models may perform differently on out-of-distribution samples.
  Results are probabilistic — use as supporting evidence, not sole proof.
```

---

## 🔬 Key Technical Decisions

| Choice | Reason |
|---|---|
| EfficientNet-B0 over ResNet | Better accuracy/compute trade-off; stronger GAN artifact detection |
| AudioMLP over wav2vec2 | Lightweight 26-dim features; no heavy transformer required; 99.6% accuracy |
| Dual-model fusion for images | Catches both face-swapped AND fully AI-generated in one API call |
| 12-frame video sampling | Fast inference without sacrificing accuracy across temporal dimension |
| Focal Loss for images | Class imbalance common in real datasets; down-weights easy negatives |
| Mixed precision (AMP) | 2× speedup on modern GPUs with no accuracy loss |
| StandardScaler for audio | Essential — raw spectral features span vastly different value ranges |

---

## 👨‍💻 Author

```
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   👤  Bhavya Kansal                                          ║
║   🎓  AI Engineer · DeepTech Developer                       ║
║   🏢  Founder & CEO — Multimodex AI                          ║
║   🎓  Diploma CSE → B.Tech AI/ML (TIET, Patiala)            ║
║   🔬  AI/ML Industrial Trainee — NIELIT Ropar × IIT Ropar   ║
║   🌐  bhavyakansal.dev                                       ║
║   📧  kansalbhavya27@gmail.com                               ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

[![Portfolio](https://img.shields.io/badge/🌐%20Portfolio-bhavyakansal.dev-00f5ff?style=for-the-badge&labelColor=0d1117)](https://bhavyakansal.dev)
[![GitHub](https://img.shields.io/badge/GitHub-BhavyaKansal20-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/BhavyaKansal20)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-kansal0920-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/kansal0920)
[![Multimodex AI](https://img.shields.io/badge/🧠%20Multimodex%20AI-Platform-ff006e?style=for-the-badge&labelColor=0d1117)](https://multimodexai.vercel.app)

---

## ⭐ Support

If **DeepFake Scanner** helped you or impressed you:

```
  1. ⭐ Star this repository
  2. 🍴 Fork and build on it
  3. 📣 Share with your network
  4. 🐛 Open issues / PRs
  5. 🔗 Share the live demo link
```

Every star helps this project reach more developers and researchers. 🙏

---

<div align="center">

```
  ╔══════════════════════════════════════════════════════╗
  ║     🔬  D E E P F A K E   S C A N N E R             ║
  ║     Multimodex AI  ·  © 2026 Bhavya Kansal           ║
  ║     Built with ❤️  for a safer digital world          ║
  ╚══════════════════════════════════════════════════════╝
```

[![Live Demo](https://img.shields.io/badge/🚀%20Try%20It%20Live-deepfake--scanner.onrender.com-00f5ff?style=for-the-badge)](https://deepfake-scanner.onrender.com)

</div>| Video | DFDC (Facebook) | 100,000+ videos | Kaggle Competition |
| Video | Celeb-DF v2 | 6,229 videos | [Request](https://github.com/yuezunli/celeb-deepfakeforensics) |
| Audio | ASVspoof 2019 LA | 121,461 clips | Kaggle Mirror |
| Audio | FakeAVCeleb | 19,500 clips | [Request](https://github.com/DASH-Lab/FakeAVCeleb) |

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Clone the project
git clone <your-repo-url>
cd deepfake-detector

# Create virtual environment
python -m venv venv
source venv/bin/activate          # Linux/Mac
# venv\Scripts\activate           # Windows

# Install requirements
pip install -r requirements.txt
```

> **GPU Required** for training. Minimum 8GB VRAM (16GB+ recommended for video model).
> Google Colab Pro works great for training.

---

### 2. Download Datasets

```bash
# Set up Kaggle API key first:
# kaggle.com → Settings → API → Create New Token → save to ~/.kaggle/kaggle.json

# Download all datasets
python scripts/download_datasets.py --dataset all

# Or test with sample data (no GPU needed)
python scripts/download_datasets.py --dataset sample
```

---

### 3. Preprocess Data

```bash
# Extract video frames + crop faces (FaceForensics++ + DFDC)
python -m src.preprocessing.extract_frames --dataset all

# Extract audio features (ASVspoof + FakeAVCeleb)
python -m src.preprocessing.audio_features --dataset all
```

---

### 4. Train Models

```bash
# Train all models sequentially
python scripts/train_all.py

# Or train individually
python -m src.train.train_image --config configs/config.yaml
python -m src.train.train_video --config configs/config.yaml
python -m src.train.train_audio --config configs/config.yaml

# Skip slow video training (still get image + audio models)
python scripts/train_all.py --skip-video
```

**Expected training times (on A100 GPU):**
| Model | Time |
|-------|------|
| Image (30 epochs) | ~2–3 hours |
| Video (25 epochs) | ~8–12 hours |
| Audio (20 epochs) | ~3–4 hours |

---

### 5. Run the App

```bash
# Terminal 1 — Start API server
python -m api.main

# Terminal 2 — Start UI
streamlit run ui/app.py

# Open browser: http://localhost:8501
# API docs:     http://localhost:8000/docs
```

---

### 6. Docker (Recommended for Production)

```bash
# Build
docker build -t deepfake-detector .

# Run (GPU)
docker run --gpus all -p 8000:8000 -p 8501:8501 \
  -v $(pwd)/models:/app/models deepfake-detector

# Or with docker-compose
docker-compose up
```

---

## 🔌 API Reference

### POST `/detect`
Auto-detects file type and runs the appropriate model(s).

```bash
curl -X POST http://localhost:8000/detect \
  -F "file=@your_image.jpg"
```

**Response:**
```json
{
  "verdict": "FAKE",
  "confidence": "HIGH",
  "fake_probability": 0.9732,
  "real_probability": 0.0268,
  "modality": "image",
  "face_detected": true,
  "latency_ms": 43.2
}
```

### POST `/detect/image`  — Images only
### POST `/detect/video`  — Videos only
### POST `/detect/audio`  — Audio only
### GET  `/health`        — System health
### GET  `/docs`          — Swagger UI

---

## 📁 Project Structure

```
deepfake-detector/
├── configs/
│   ├── config.yaml          ← All hyperparameters & paths
│   └── nginx.conf           ← Production reverse proxy
├── src/
│   ├── preprocessing/
│   │   ├── extract_frames.py    ← Video → frames → face crops
│   │   └── audio_features.py   ← Audio → MFCC/mel/LFCC
│   ├── models/
│   │   ├── image_model.py       ← EfficientNet-B4 + Attention
│   │   ├── video_model.py       ← EfficientNet-B4 + BiLSTM
│   │   └── audio_model.py       ← wav2vec2 + LCNN dual-branch
│   ├── train/
│   │   ├── train_image.py       ← Image training pipeline
│   │   ├── train_video.py       ← Video training pipeline
│   │   └── train_audio.py       ← Audio training pipeline
│   ├── inference/
│   │   └── detector.py          ← Unified inference engine
│   ├── fusion/
│   │   └── ensemble.py          ← Weighted avg / voting / meta-clf
│   └── utils/
│       └── helpers.py           ← Seeds, metrics, checkpoints
├── api/
│   └── main.py              ← FastAPI backend
├── ui/
│   └── app.py               ← Streamlit dashboard
├── scripts/
│   ├── download_datasets.py
│   ├── train_all.py
│   └── docker_start.sh
├── tests/
│   └── test_pipeline.py     ← Full test suite (pytest)
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## 🧪 Run Tests

```bash
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=src --cov-report=html
```

---

## ⚙️ Configuration

Edit `configs/config.yaml` to change:
- Model architectures (`efficientnet_b4` / `xception` / `vit_base_patch16_224`)
- Training hyperparameters (LR, batch size, epochs)
- Dataset paths
- Fusion weights and strategy
- API settings

---

## 📈 Expected Performance

| Model | Dataset | Accuracy | AUC-ROC | EER |
|-------|---------|---------|---------|-----|
| Image | 140k Faces | ~99.1% | 0.999 | — |
| Video | FF++ C23 | ~97.3% | 0.991 | — |
| Audio | ASVspoof LA | ~97.8% | 0.998 | ~2.1% |

*Results on held-out test sets. Generalisation to unseen generators will vary.*

---

## 🔬 Key Technical Decisions

| Choice | Reason |
|--------|--------|
| EfficientNet-B4 over ResNet | Better accuracy/compute trade-off, stronger GAN artifact detection |
| BiLSTM over Transformer | Lower memory for long sequences; captures bidirectional temporal inconsistencies |
| wav2vec2 for audio | Pre-trained on 960h of real speech; excellent transfer for detecting synthesis artifacts |
| LCNN alongside wav2vec2 | Spectral features catch different forgery signatures than waveform features |
| Focal Loss for images | Class imbalance common in real datasets; Focal Loss down-weights easy negatives |
| Mixed precision (AMP) | 2× speedup on modern GPUs with no accuracy loss |
| Face crop before classification | Reduces irrelevant background; focuses model on manipulation region |

---

## 📝 License

MIT License — use freely for research and commercial purposes.

---

## 🙏 Acknowledgements

- [FaceForensics++](https://github.com/ondyari/FaceForensics) — Rössler et al.
- [DFDC](https://ai.facebook.com/datasets/dfdc/) — Facebook AI
- [ASVspoof 2019](https://www.asvspoof.org/) — ASVspoof Challenge
- [wav2vec 2.0](https://arxiv.org/abs/2006.11477) — Baevski et al., Facebook AI
- [EfficientNet](https://arxiv.org/abs/1905.11946) — Tan & Le, Google Brain
# DeepFake-Scanner
