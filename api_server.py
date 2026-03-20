import os, time, pickle, tempfile
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import timm
import cv2
import subprocess
from pathlib import Path
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image as PILImage
import uvicorn

app = FastAPI(title="Deepfake Detector API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}")

def build_effb0():
    bb = timm.create_model('efficientnet_b0', pretrained=False, num_classes=0, global_pool='avg')
    head = nn.Sequential(nn.Linear(bb.num_features,256), nn.LayerNorm(256), nn.ReLU(True), nn.Dropout(0.3), nn.Linear(256,2))
    return nn.Sequential(bb, head)

def build_gan():
    bb = timm.create_model('efficientnet_b0', pretrained=False, num_classes=0, global_pool='avg')
    head = nn.Sequential(nn.Linear(bb.num_features,256), nn.LayerNorm(256), nn.ReLU(True), nn.Dropout(0.35), nn.Linear(256,64), nn.ReLU(True), nn.Dropout(0.2), nn.Linear(64,2))
    return nn.Sequential(bb, head)

# Face-swap model
faceswap_model = build_effb0().to(device)
faceswap_model.load_state_dict(torch.load('models/image_model/best_model.pt', map_location=device)['model_state_dict'])
faceswap_model.eval()
print("✅ Face-swap detector loaded  (95.8% acc)")

# GAN detector
gan_model = None
if os.path.exists('models/gan_detector/best_model.pt'):
    gan_model = build_gan().to(device)
    gan_model.load_state_dict(torch.load('models/gan_detector/best_model.pt', map_location=device)['model_state_dict'])
    gan_model.eval()
    print("✅ GAN/Diffusion detector loaded (98.1% AUC)")
else:
    print("⚠️  GAN model not found")

# Audio model
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

audio_model = AudioMLP().to(device)
audio_model.load_state_dict(torch.load('models/audio_model/best_model.pt', map_location=device)['model_state_dict'])
audio_model.eval()
with open('models/audio_model/feature_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
print("✅ Audio model loaded          (99.6% acc)")

IMG_TF = A.Compose([A.Resize(224,224), A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]), ToTensorV2()])
IMAGE_EXTS = {'.jpg','.jpeg','.png','.webp','.bmp'}
AUDIO_EXTS = {'.wav','.mp3','.flac','.m4a','.ogg'}
VIDEO_EXTS = {'.mp4','.avi','.mov','.mkv','.webm'}

@app.get("/health")
def health():
    return {
        "status":"ok",
        "models":{
            "image":True,
            "video":True,
            "gan_detector":gan_model is not None,
            "audio":True,
        },
        "device":str(device),
    }

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()
    content = await file.read()
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
        f.write(content); tmp = f.name
    try:
        if ext in IMAGE_EXTS: return detect_image(tmp)
        elif ext in AUDIO_EXTS: return detect_audio(tmp)
        elif ext in VIDEO_EXTS: return detect_video(tmp)
        else: return {"error": f"Unsupported: {ext}"}
    finally:
        os.unlink(tmp)

@app.post("/detect/image")
async def detect_image_ep(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()
    content = await file.read()
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
        f.write(content); tmp = f.name
    try: return detect_image(tmp)
    finally: os.unlink(tmp)

@app.post("/detect/audio")
async def detect_audio_ep(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()
    content = await file.read()
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
        f.write(content); tmp = f.name
    try: return detect_audio(tmp)
    finally: os.unlink(tmp)

@app.post("/detect/video")
async def detect_video_ep(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()
    content = await file.read()
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
        f.write(content); tmp = f.name
    try: return detect_video(tmp)
    finally: os.unlink(tmp)

def _image_fake_probability_from_rgb(rgb_image, with_components=False):
    tensor = IMG_TF(image=rgb_image)['image'].unsqueeze(0).to(device)

    with torch.no_grad():
        p1 = F.softmax(faceswap_model(tensor), dim=-1).cpu().numpy()[0]
    faceswap_fake = float(p1[1])

    gan_fake = 0.0
    if gan_model is not None:
        with torch.no_grad():
            p2 = F.softmax(gan_model(tensor), dim=-1).cpu().numpy()[0]
        gan_fake = float(p2[1])

    if gan_model is not None:
        if gan_fake > 0.8 or faceswap_fake > 0.8:
            final_fake = max(faceswap_fake, gan_fake)
        else:
            final_fake = faceswap_fake * 0.45 + gan_fake * 0.55
    else:
        final_fake = faceswap_fake

    if with_components:
        return final_fake, faceswap_fake, gan_fake
    return final_fake

def _extract_audio_from_video(video_path):
    tmp_audio = tempfile.mktemp(suffix=".wav")
    cmd = [
        "ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le",
        "-ar", "22050", "-ac", "1", tmp_audio, "-y", "-loglevel", "error"
    ]
    try:
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode == 0 and os.path.exists(tmp_audio):
            return tmp_audio
    except Exception:
        pass
    return None

def detect_image(path):
    t0 = time.time()
    try:
        img = np.array(PILImage.open(path).convert('RGB'))
        final_fake, faceswap_fake, gan_fake = _image_fake_probability_from_rgb(img, with_components=True)

        verdict    = "FAKE" if final_fake >= 0.5 else "REAL"
        confidence = "HIGH" if abs(final_fake-0.5)>0.35 else "MEDIUM" if abs(final_fake-0.5)>0.15 else "LOW"

        return {
            "verdict": verdict, "confidence": confidence,
            "fake_probability": round(final_fake, 4),
            "real_probability": round(1-final_fake, 4),
            "modality": "image",
            "latency_ms": round((time.time()-t0)*1000, 1),
            "detail": {
                "faceswap_score": round(faceswap_fake, 4),
                "gan_score": round(gan_fake, 4),
                "fusion": "dual_model",
            }
        }
    except Exception as e: return {"error": str(e)}

def detect_video(path):
    t0 = time.time()
    try:
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            return {"error": "Could not read video frames"}

        sample_count = 12
        indices = np.linspace(0, max(total - 1, 0), sample_count, dtype=int)
        frame_scores = []

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame_bgr = cap.read()
            if not ok:
                continue
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_scores.append(_image_fake_probability_from_rgb(frame_rgb))

        cap.release()

        if not frame_scores:
            return {"error": "Could not process video frames"}

        video_fake = float(np.mean(frame_scores))

        audio_score = None
        tmp_audio = _extract_audio_from_video(path)
        if tmp_audio:
            try:
                audio_result = detect_audio(tmp_audio)
                if "error" not in audio_result:
                    audio_score = float(audio_result["fake_probability"])
            finally:
                if os.path.exists(tmp_audio):
                    os.unlink(tmp_audio)

        final_fake = video_fake if audio_score is None else (0.7 * video_fake + 0.3 * audio_score)

        verdict = "FAKE" if final_fake >= 0.5 else "REAL"
        confidence = "HIGH" if abs(final_fake-0.5)>0.35 else "MEDIUM" if abs(final_fake-0.5)>0.15 else "LOW"

        return {
            "verdict": verdict,
            "confidence": confidence,
            "fake_probability": round(float(final_fake), 4),
            "real_probability": round(float(1-final_fake), 4),
            "modality": "video" if audio_score is None else "video+audio",
            "latency_ms": round((time.time()-t0)*1000, 1),
            "detail": {
                "video_score": round(float(video_fake), 4),
                "audio_score": None if audio_score is None else round(float(audio_score), 4),
                "frames_sampled": len(frame_scores),
                "peak_fake_frame": int(np.argmax(frame_scores)),
                "frame_scores": [round(float(s), 4) for s in frame_scores],
                "fusion": "temporal_image_plus_audio",
            },
        }
    except Exception as e:
        return {"error": str(e)}

def detect_audio(path):
    t0 = time.time()
    try:
        import librosa
        y, sr = librosa.load(path, sr=22050, mono=True)
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
        fake_p = float(probs[1])
        verdict    = "FAKE" if fake_p >= 0.5 else "REAL"
        confidence = "HIGH" if abs(fake_p-0.5)>0.35 else "MEDIUM" if abs(fake_p-0.5)>0.15 else "LOW"
        return {"verdict":verdict,"confidence":confidence,
                "fake_probability":round(fake_p,4),"real_probability":round(float(probs[0]),4),
                "modality":"audio","latency_ms":round((time.time()-t0)*1000,1)}
    except Exception as e: return {"error": str(e)}

if __name__ == "__main__":
    print("\n" + "="*50)
    print("🚀 DEEPFAKE DETECTOR API")
    print("   http://localhost:8000")
    print("   http://localhost:8000/docs")
    print("="*50 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
