# ─────────────────────────────────────────────────────────────────────────────
#  Deepfake Detector — Production Dockerfile
#  Build: docker build -t deepfake-detector .
#  Run:   docker run -p 8000:8000 -p 8501:8501 deepfake-detector
# ─────────────────────────────────────────────────────────────────────────────

FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    cmake \
    build-essential \
    git \
    wget \
    curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy project ──────────────────────────────────────────────────────────────
COPY . .

# ── Ensure __init__.py files exist for module imports ─────────────────────────
RUN find src -type d -exec touch {}/__init__.py \; && \
    touch api/__init__.py ui/__init__.py

# ── Create required directories ───────────────────────────────────────────────
RUN mkdir -p data/{raw,processed/{frames,faces,audio}} \
             models/{image_model,video_model,audio_model} \
             logs results /tmp/deepfake_uploads

# ── Environment ───────────────────────────────────────────────────────────────
ENV PYTHONPATH=/app
ENV CONFIG_PATH=/app/configs/config.yaml
ENV MODELS_DIR=/app/models
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=4

# ── Expose ports ──────────────────────────────────────────────────────────────
EXPOSE 8000 8501

# ── Startup script ────────────────────────────────────────────────────────────
COPY scripts/docker_start.sh /docker_start.sh
RUN chmod +x /docker_start.sh

CMD ["/docker_start.sh"]
