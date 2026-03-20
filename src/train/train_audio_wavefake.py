"""
src/train/train_audio_wavefake.py — WaveFake dataset audio trainer
Supports csv mode (fast MLP) and raw mode (CNN on mel spectrograms)
"""

import os, argparse, time, random, pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import yaml
from loguru import logger
from tqdm import tqdm
from src.utils.helpers import set_seed, get_device, setup_logger, ensure_dir


class AudioMLP(nn.Module):
    def __init__(self, input_dim=26, num_classes=2, dropout=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(True), nn.Dropout(dropout),
            nn.Linear(256, 512), nn.BatchNorm1d(512), nn.ReLU(True), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(True), nn.Dropout(dropout/2),
            nn.Linear(256, 128), nn.ReLU(True), nn.Dropout(dropout/2),
            nn.Linear(128, num_classes),
        )
    def forward(self, x): return self.net(x)


class CSVAudioDataset(Dataset):
    def __init__(self, features, labels):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


class LightAudioCNN(nn.Module):
    def __init__(self, num_classes=2, dropout=0.4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.MaxPool2d(2), nn.Dropout2d(0.1),
            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2), nn.Dropout2d(0.2),
            nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d(2), nn.Dropout2d(0.2),
            nn.Conv2d(128,256,3,padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(256,128), nn.ReLU(True), nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )
    def forward(self, x): return self.classifier(self.features(x))


class RawAudioDataset(Dataset):
    def __init__(self, real_dir, fake_dir, sr=16000, max_dur=4.0, split="train", seed=42):
        self.sr = sr
        self.max_samples = int(sr * max_dur)
        self.split = split
        exts = {".wav", ".flac", ".mp3", ".ogg"}
        real_files = [(str(f), 0) for f in Path(real_dir).rglob("*") if f.suffix.lower() in exts]
        fake_files = [(str(f), 1) for f in Path(fake_dir).rglob("*") if f.suffix.lower() in exts]
        all_files = real_files + fake_files
        random.seed(seed)
        random.shuffle(all_files)
        n = len(all_files)
        n_train, n_val = int(n*0.8), int(n*0.1)
        self.samples = {"train": all_files[:n_train],
                        "val":   all_files[n_train:n_train+n_val],
                        "test":  all_files[n_train+n_val:]}[split]
        logger.info(f"RawAudioDataset {split}: {len(self.samples)} files")

    def _load(self, path):
        import librosa
        w, _ = librosa.load(path, sr=self.sr, mono=True)
        if len(w) < self.max_samples: w = np.pad(w, (0, self.max_samples-len(w)))
        else: w = w[:self.max_samples]
        return w.astype(np.float32)

    def _mel(self, w):
        import librosa
        m = librosa.feature.melspectrogram(y=w, sr=self.sr, n_fft=512, hop_length=160, n_mels=80)
        m = librosa.power_to_db(m, ref=np.max)
        return ((m - m.mean()) / (m.std() + 1e-8)).astype(np.float32)

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        try:
            w = self._load(path)
            if self.split == "train" and np.random.random() < 0.3:
                w += np.random.randn(*w.shape).astype(np.float32) * 0.005
            m = self._mel(w)
        except:
            w = np.zeros(self.max_samples, dtype=np.float32)
            m = np.zeros((80, self.max_samples//160), dtype=np.float32)
        return torch.tensor(w), torch.tensor(m).unsqueeze(0), torch.tensor(label, dtype=torch.long)


def train_csv_mode(csv_path, out_dir, cfg):
    set_seed(cfg["project"]["seed"])
    device = get_device()
    ensure_dir(out_dir)
    logger.info(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    df["label_int"] = df["LABEL"].map({"REAL": 0, "FAKE": 1})
    df = df.dropna(subset=["label_int"])
    feature_cols = [c for c in df.columns if c not in ("LABEL","label_int")]
    X = df[feature_cols].values.astype(np.float32)
    y = df["label_int"].values.astype(np.int64)
    logger.info(f"Samples: {len(X)} | Features: {X.shape[1]} | Real={sum(y==0)} Fake={sum(y==1)}")
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    with open(os.path.join(out_dir, "feature_scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    tr_dl = DataLoader(CSVAudioDataset(X_tr, y_tr), batch_size=256, shuffle=True)
    val_dl = DataLoader(CSVAudioDataset(X_val, y_val), batch_size=512)
    test_dl = DataLoader(CSVAudioDataset(X_test, y_test), batch_size=512)
    model = AudioMLP(X.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    best_auc, no_imp = 0.0, 0
    for epoch in range(1, 51):
        model.train()
        tr_loss = 0.0
        for xb, yb in tr_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            tr_loss += loss.item()
        sched.step()
        model.eval()
        preds, targets, probs = [], [], []
        with torch.no_grad():
            for xb, yb in val_dl:
                p = F.softmax(model(xb.to(device)), dim=-1).cpu().numpy()
                preds.extend(p.argmax(1)); targets.extend(yb.numpy()); probs.extend(p)
        acc = accuracy_score(targets, preds)
        auc = roc_auc_score(targets, np.array(probs)[:,1])
        logger.info(f"Epoch {epoch:03d}/050 | Loss={tr_loss/len(tr_dl):.4f} | ValAcc={acc:.4f} ValAUC={auc:.4f}")
        if auc > best_auc:
            best_auc = auc; no_imp = 0
            torch.save({"epoch":epoch,"model_state_dict":model.state_dict(),
                        "metrics":{"accuracy":acc,"auc_roc":auc},"feature_cols":feature_cols},
                       os.path.join(out_dir,"best_model.pt"))
            logger.info(f"  ✅ Best model saved — AUC={auc:.4f}")
        else:
            no_imp += 1
        if no_imp >= 10: logger.info("Early stopping"); break
    logger.info(f"\nBest Val AUC: {best_auc:.4f}")
    ckpt = torch.load(os.path.join(out_dir,"best_model.pt"), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"]); model.eval()
    preds, targets, probs = [], [], []
    with torch.no_grad():
        for xb, yb in test_dl:
            p = F.softmax(model(xb.to(device)), dim=-1).cpu().numpy()
            preds.extend(p.argmax(1)); targets.extend(yb.numpy()); probs.extend(p)
    logger.info(f"\nTEST  Accuracy={accuracy_score(targets,preds):.4f}  AUC={roc_auc_score(targets,np.array(probs)[:,1]):.4f}")
    logger.info("\n" + classification_report(targets, preds, target_names=["REAL","FAKE"]))


def train_raw_mode(audio_dir, out_dir, cfg):
    set_seed(cfg["project"]["seed"])
    device = get_device()
    ensure_dir(out_dir)
    real_dir = os.path.join(audio_dir, "REAL")
    fake_dir = os.path.join(audio_dir, "FAKE")
    tr_ds = RawAudioDataset(real_dir, fake_dir, split="train")
    val_ds = RawAudioDataset(real_dir, fake_dir, split="val")
    tr_dl = DataLoader(tr_ds, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)
    model = LightAudioCNN().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=15)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    best_auc, no_imp = 0.0, 0
    epochs = cfg["audio_model"]["train"]["epochs"]
    for epoch in range(1, epochs+1):
        t0 = time.time()
        model.train()
        tr_loss = 0.0
        for _, mel, labels in tqdm(tr_dl, desc=f"Epoch {epoch}", leave=False):
            mel, labels = mel.to(device), labels.to(device)
            opt.zero_grad()
            crit(model(mel), labels).backward()
            opt.step()
            tr_loss += crit(model(mel), labels).item()
        sched.step()
        model.eval()
        preds, targets, probs = [], [], []
        with torch.no_grad():
            for _, mel, labels in val_dl:
                p = F.softmax(model(mel.to(device)), dim=-1).cpu().numpy()
                preds.extend(p.argmax(1)); targets.extend(labels.numpy()); probs.extend(p)
        acc = accuracy_score(targets, preds)
        auc = roc_auc_score(targets, np.array(probs)[:,1])
        logger.info(f"Epoch {epoch:03d}/{epochs} | Loss={tr_loss/len(tr_dl):.4f} | ValAcc={acc:.4f} ValAUC={auc:.4f} | {time.time()-t0:.1f}s")
        if auc > best_auc:
            best_auc = auc; no_imp = 0
            torch.save({"epoch":epoch,"model_state_dict":model.state_dict(),"metrics":{"accuracy":acc,"auc_roc":auc}},
                       os.path.join(out_dir,"best_model.pt"))
            logger.info(f"  ✅ Best model saved — AUC={auc:.4f}")
        else:
            no_imp += 1
        if no_imp >= cfg["audio_model"]["train"]["early_stopping_patience"]:
            logger.info("Early stopping"); break
    logger.info(f"Done. Best AUC: {best_auc:.4f} → {out_dir}/best_model.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--mode", choices=["csv","raw"], default="csv")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    setup_logger(cfg["paths"]["logs"], "audio_wavefake")
    out_dir = os.path.join(cfg["paths"]["models"], "audio_model")
    ensure_dir(out_dir)
    if args.mode == "csv":
        train_csv_mode("data/raw/audio/KAGGLE/DATASET-balanced.csv", out_dir, cfg)
    else:
        train_raw_mode("data/raw/audio/KAGGLE/AUDIO", out_dir, cfg)