#!/bin/bash
# scripts/docker_start.sh — Start API + UI simultaneously

echo "🔍 Starting Deepfake Detector..."

# Start FastAPI in background
uvicorn api.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 2 \
  --log-level info &

API_PID=$!
echo "✅ API started (PID=$API_PID) at http://0.0.0.0:8000"

# Wait for API to be ready
sleep 3

# Start Streamlit UI
streamlit run ui/app.py \
  --server.port=8501 \
  --server.address=0.0.0.0 \
  --server.headless=true \
  --browser.gatherUsageStats=false &

UI_PID=$!
echo "✅ UI started (PID=$UI_PID) at http://0.0.0.0:8501"

# Wait for either process to exit
wait -n $API_PID $UI_PID
