# 🌀 Cyclone Intensity Estimation from Satellite Images using Deep Learning (TCNet)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-15-black.svg)](https://nextjs.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0-blue.svg)](https://www.typescriptlang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end operational Deep Learning system for estimating **Tropical Cyclone (TC) intensity** from single-channel Infrared (IR) satellite imagery. Powered by **TCNet** — a custom 4-layer Convolutional Neural Network trained on the TCIR dataset — featuring **Test-Time Augmentation (Rotation-Blending)**, **Regression Grad-CAM explainability**, and **India Meteorological Department (IMD) intensity scale categorization**.

---

## 🌟 Key Features

- **High-Accuracy CNN Architecture:** Custom 4-stage convolutional neural network (`389,057` parameters) achieving a **Validation MAE of 11.61 knots** on unseen test tracks.
- **Rotation-Blending (TTA):** 4-angle rotation averaging ($0^\circ, 90^\circ, 180^\circ, 270^\circ$) to neutralize rotational variance and compute prediction uncertainty.
- **Regression Grad-CAM (Explainable AI):** Visualizes which cyclone storm structures (eyewall, central dense overcast, spiral rainbands) contribute most to the predicted wind speed.
- **IMD Categorization:** Automatically maps estimated wind speed (knots and km/h) to official IMD categories:
  - *Low Pressure Area* (< 17 kn)
  - *Depression* (17–27 kn)
  - *Deep Depression* (28–33 kn)
  - *Cyclonic Storm* (34–47 kn)
  - *Severe Cyclonic Storm* (48–63 kn)
  - *Very Severe Cyclonic Storm* (64–89 kn)
  - *Extremely Severe Cyclonic Storm* (90–119 kn)
  - *Super Cyclonic Storm* ($\ge$ 120 kn)
- **Modern Interactive Dashboard:** Next.js application with full-screen infrared visualizer, opacity & sensitivity sliders, split-view mode, preset gallery of notable historical cyclones (Amphan, Fani, Tauktae, Biparjoy, Yaas, Nisarga), and real-time ML performance metrics.

---

## 🏗️ System Architecture

```text
CYCLONE-INTENSITY-ESTIMATION/
├── tcnet-backend/              # FastAPI Python Backend
│   ├── app.py                  # API endpoints (/health, /predict)
│   ├── model.py                # TCNet PyTorch CNN architecture (389,057 params)
│   ├── preprocessor.py         # 64x64 center crop, normalization, TTA, and Grad-CAM hooks
│   ├── baseline_cnn_v3.pt      # Pre-trained TCNet model checkpoint
│   ├── presets/                # Historical cyclone satellite IR frames (.npy)
│   ├── requirements.txt        # Python backend dependencies
│   └── README.md
│
├── tcnet-frontend/             # Next.js 15 Interactive Dashboard
│   ├── app/                    # App Router pages and layout
│   ├── components/             # Visualizer viewport, metric panels, preset gallery
│   ├── hooks/                  # Inference and health-check React hooks
│   ├── lib/                    # API client and IMD scale mappings
│   ├── types/                  # TypeScript interface definitions
│   └── package.json
│
├── context/                    # Reference research code & specifications
│   └── Phase3_Final_Code-1.md
│
├── .gitignore                  # Git ignore rules (protects credentials & build artifacts)
└── README.md                   # Project documentation
```

---

## 🚀 Quick Start Guide

### 1. Clone the Repository

```bash
git clone https://github.com/advinvarghese6-star/CYCLONE-INTENSITY-ESTIMATION-FROM-SATELLITE-IMAGES-USING-DEEP-LEARNING.git
cd CYCLONE-INTENSITY-ESTIMATION-FROM-SATELLITE-IMAGES-USING-DEEP-LEARNING
```

---

### 2. Run Backend (FastAPI + PyTorch)

```bash
cd tcnet-backend

# Create and activate virtual environment
python -m venv venv

# Windows
.\venv\Scripts\activate

# Linux / macOS
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start backend server on port 8000
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Backend will be available at:
- API Root: `http://127.0.0.1:8000`
- Swagger Interactive Docs: `http://127.0.0.1:8000/docs`
- Health Endpoint: `http://127.0.0.1:8000/health`

---

### 3. Run Frontend (Next.js Dashboard)

Open a new terminal window:

```bash
cd tcnet-frontend

# Install dependencies
npm install

# Start Next.js development server
npm run dev
```

Dashboard will be live at:
- Web Application: **[http://localhost:3000](http://localhost:3000)**

---

## 📊 Preprocessing & Model Details

| Parameter | Specification | Description |
| :--- | :--- | :--- |
| **Input Shape** | `(1, 64, 64)` | 64x64 center crop centered on storm eye |
| **Normalization** | `(X - 0.4350) / 0.2869` | Standardized on TCIR dataset distributions |
| **Backbone** | 4-Stage Conv2D | Channels: `32 -> 64 -> 128 -> 256` with BatchNorm & ReLU |
| **Pooling** | `AdaptiveAvgPool2d(1)` | Global Average Pooling |
| **Head** | `Linear(256, 1)` | Direct continuous intensity regression (knots) |
| **Parameters** | `389,057` | Lightweight, edge-deployable footprint |
| **Validation MAE**| `11.61 knots` | Verified on historical cyclone test tracks |

---

## 🔬 Explainable AI (Grad-CAM)

TCNet hooks into the final convolutional feature extractor (`features[11]`, ReLU output) to backpropagate intensity gradients with respect to spatial activations:

$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$

Where $\alpha_k^c$ captures the global sensitivity of wind speed regression to the $k$-th feature map, producing an intuitive Jet colormap overlay indicating the primary storm circulation drivers.

---

## 👤 Author

- **Advin Varghese** ([@advinvarghese6-star](https://github.com/advinvarghese6-star))
- Email: [advinvarghese6@gmail.com](mailto:advinvarghese6@gmail.com)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
