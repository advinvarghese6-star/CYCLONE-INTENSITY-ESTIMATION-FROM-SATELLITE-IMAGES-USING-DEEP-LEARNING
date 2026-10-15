import os
from pathlib import Path

# Base Directory
BASE_DIR = Path(__file__).resolve().parent

# Google Drive Configuration
SERVICE_ACCOUNT_FILE = os.getenv("SERVICE_ACCOUNT_FILE", str(BASE_DIR / "credentials.json"))
# Copy the string from between /d/ and /view in your Google Drive URL:
DRIVE_FILE_ID = os.getenv("DRIVE_FILE_ID", "YOUR_COPIED_FILE_ID_HERE")

# Model Configuration
MODEL_LOCAL_PATH = os.getenv("MODEL_LOCAL_PATH", str(BASE_DIR / "baseline_cnn_v3.pt"))
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

# Server Configuration
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# CORS Configuration
CORS_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

# Additional origins from environment variable (comma-separated)
extra_origins = os.getenv("ADDITIONAL_CORS_ORIGINS", "")
if extra_origins:
    CORS_ORIGINS.extend([origin.strip() for origin in extra_origins.split(",") if origin.strip()])
