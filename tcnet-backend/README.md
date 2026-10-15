# TCNet — Cyclone Intensity Estimation Backend

A high-performance FastAPI backend for estimating Tropical Cyclone intensity from satellite Infrared (IR) imagery using the **TCNet** Deep Convolutional Neural Network with **Phase 3 Test-Time Augmentation (Rotation-Blending)** and **Grad-CAM sensitivity visualization**.

---

## 📁 Project Structure

```text
tcnet-backend/
├── app.py                 # FastAPI server, endpoints (/health, /predict), CORS, and lifecycle
├── model.py               # PyTorch TCNet 4-layer CNN architecture definition
├── drive_service.py       # Google Drive API v3 Service Account downloader
├── preprocessor.py        # 64x64 center crop, normalization, TTA rotation, and Grad-CAM overlay
├── config.py              # Configuration constants & environment variables
├── credentials.json       # Google Drive Service Account key
├── requirements.txt       # Python dependencies
├── .gitignore             # Git ignore for credentials, checkpoints, and venv
└── README.md              # Documentation and instructions
```

---

## ⚡ Prerequisites

- **Python 3.10+** (tested on Python 3.10 – 3.14)
- **Google Drive** model file (`baseline_cnn_v3.pt`)

---

## 🚀 Setup & Installation

### 1. Create a Python Virtual Environment
Open your terminal inside the `tcnet-backend` directory:

```bash
cd "tcnet-backend"
python -m venv venv
```

Activate the virtual environment:
- **Windows (PowerShell):**
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
- **Windows (Command Prompt):**
  ```cmd
  .\venv\Scripts\activate.bat
  ```
- **macOS / Linux:**
  ```bash
  source venv/bin/activate
  ```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🔑 Google Drive Service Account Authentication

The backend downloads the trained model weights (`baseline_cnn_v3.pt`) directly from Google Drive upon startup if the file does not already exist on disk.

### 1. Service Account Key
The `credentials.json` file in `tcnet-backend/` contains your Google Cloud Service Account credentials:
- **Service Account Email:** `website-downloader@balmy-curve-511703-v3.iam.gserviceaccount.com`

### 2. Share Your Model File on Google Drive
1. Go to your [Google Drive](https://drive.google.com).
2. Locate or upload your trained model file `baseline_cnn_v3.pt`.
3. Right-click the file and click **Share**.
4. In the "Add people and groups" box, paste:
   ```text
   website-downloader@balmy-curve-511703-v3.iam.gserviceaccount.com
   ```
5. Ensure the permission is set to **Viewer** and click **Send / Share**.

### 3. Extract the `FILE_ID`
In your browser address bar when viewing the file, the URL looks like:
```text
https://drive.google.com/file/d/1a2b3c4d5e6f7g8h9i0j/view
                                ^^^^^^^^^^^^^^^^^^^^
                                  THIS IS YOUR FILE ID
```

### 4. Insert Your `FILE_ID`
Open `config.py` and replace the placeholder:
```python
DRIVE_FILE_ID = os.getenv("DRIVE_FILE_ID", "PASTE_YOUR_COPIED_FILE_ID_HERE")
```
*Alternatively, you can set an environment variable without modifying code:*
```powershell
$env:DRIVE_FILE_ID="PASTE_YOUR_COPIED_FILE_ID_HERE"
```

---

## 🏃 Running the Server

Start the local development server with hot-reload enabled on port 8000:

```bash
uvicorn app:app --reload --port 8000
```

When started, FastAPI will:
1. Verify whether `baseline_cnn_v3.pt` exists locally.
2. If absent, authenticate with Google Drive using `credentials.json` and stream the file into `baseline_cnn_v3.pt`.
3. Load model weights into PyTorch (using CUDA GPU if available, else CPU).
4. Listen for requests at `http://localhost:8000`.

---

## 📡 API Endpoints

### 1. `GET /health`
Verifies backend connectivity, model loading status, and device info.
```json
{
  "status": "ok",
  "model_loaded": true,
  "device": "cpu",
  "drive_file_id_configured": true,
  "model_path": ".../baseline_cnn_v3.pt"
}
```

### 2. `POST /predict`
Estimates cyclone intensity from an uploaded satellite image or preset ID.

**Form Data Parameters:**
- `file`: (Optional) Satellite image file (`.png`, `.jpg`, `.jpeg`, or binary `.npy`).
- `preset_id`: (Optional) Preset cyclone ID (e.g., `fani_2019`, `amphan_2020`, `tauktae_2021`, `biparjoy_2023`, `yaas_2021`, `nisarga_2020`).
- `use_rotation_blending`: (Optional, `bool`) When `true`, runs 10-angle Test-Time Augmentation (TTA).

**Response Format:**
```json
{
  "status": "success",
  "intensity_knots": 84.5,
  "single_pass_knots": 82.1,
  "rotation_spread": 2.14,
  "rotation_blended": true,
  "imd_category": "Very Severe Cyclonic Storm",
  "raw_image_base64": "iVBORw0KGgo...",
  "gradcam_overlay_base64": "iVBORw0KGgo..."
}
```

### 3. Interactive Documentation
- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`

---

## 🌊 IMD Cyclone Intensity Scale Reference

| Wind Speed (knots) | IMD Classification Category |
| :--- | :--- |
| `< 28 kts` | Low Pressure Area |
| `28 – 33 kts` | Depression |
| `34 – 47 kts` | Cyclonic Storm |
| `48 – 63 kts` | Severe Cyclonic Storm |
| `64 – 89 kts` | Very Severe Cyclonic Storm |
| `90 – 119 kts` | Extremely Severe Cyclonic Storm |
| `≥ 120 kts` | Super Cyclonic Storm |
