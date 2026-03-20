import subprocess, sys, os, threading, time

def run_api():
    os.system("python3 api_server.py &")

def run_ui():
    port = os.environ.get("PORT", "8501")
    os.system(f"streamlit run ui/app.py --server.port {port} --server.address 0.0.0.0 --server.headless true")

# Start API in background
api_thread = threading.Thread(target=run_api, daemon=True)
api_thread.start()
time.sleep(3)

# Run UI in foreground (Render needs this)
run_ui()
