# AI-Optimized Yield Advisor

An explainable semiconductor chip-test yield intelligence application that automatically preloads the **AX7-SOC** chip-test dataset on startup and presents plain-English root causes, interactive wafer defect maps with AI Grad-CAM overlays, hidden burn-in risk screens, test-time savings, and prioritized engineering actions.

---

## What It Runs

| Engine | Model / Method | What It Explains |
|---|---|---|
| **Wafer Map Pattern AI** | PyTorch Multi-Label ResNet-CNN (`32×32×3`) + Grad-CAM + 2D Stepper Field Periodicity | Classifies `Edge-Ring`, `Center`, `Scratch`, `Loc`, `Reticle-Repeat`, and `Clean` wafers (`0.9908` hold-out Macro F1) and highlights where the AI focused |
| **Excursion Detector** | Chronological EWMA + CUSUM + Change-Point Segmentation + Isolation Forest | Catches small sustained lot yield shifts earlier than static thresholds |
| **Root Cause Engine** | LightGBM + SHAP + Benjamini-Hochberg Fisher/Mann-Whitney + Peer-Window Guardrails | Ranks fab equipment, chambers, reticles, handlers, and distinguishes probe-card site wear from silicon defects |
| **Hidden Risk Screener** | Nearest Neighbor Residual (NNR), Dynamic PAT (DPAT), Bad-Neighborhood (GDBN), Isolation Forest | Flags passing chips with latent reliability risk (validated against burn-in failures) |
| **Test Time Optimizer** | Unique-Fail Redundancy + Pearson Correlation + 95% Rule-of-Three DPPM Bound + Reordering | Quantifies safe audit-sampling candidates and stop-on-fail speedup |
| **Pre-Sort Yield Forecaster** | Quantile LightGBM (`P10`, `P50`, `P90`) | Predicts lot sort yield from electrical (WAT) and routing inputs before wafer testing |

---

## How to Run Locally

### Prerequisites
- Python **3.11+** (tested on 3.13)
- Node.js **18+** (only needed if you edit the React frontend; pre-built `frontend/dist/` is already included)

### 1. Clone the Repository
```bash
git clone https://github.com/naveensreekanth/y_adv.git
cd y_adv
```

### 2. Create a Virtual Environment & Install Python Packages
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
# source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Launch the Application
```bash
python run.py
```
`run.py` automatically picks an open port, preloads the dataset at startup, and opens the dashboard directly in your browser. You can also specify a port manually:
```bash
python run.py --port 8090
```

### (Optional) Verification & Frontend Editing
```bash
# Run end-to-end verification tests
python verify_e2e.py

# Edit & rebuild the React frontend
cd frontend
npm install
npm run build
```

---

## How to Deploy & Make It Live

Because the FastAPI backend automatically preloads the dataset on startup and serves the pre-built `frontend/dist` directly on `$PORT`, you can deploy it as one unified web service.

### Option A: Render (Recommended — Fastest via GitHub)
This repository includes a `render.yaml` blueprint.

1. Go to [render.com](https://render.com) and sign in with GitHub.
2. Click **New +** → **Web Service** (or **Blueprint** to auto-read `render.yaml`).
3. Connect the `naveensreekanth/y_adv` repository.
4. Confirm these settings:
   - **Runtime:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `cd backend && uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - **Health Check Path:** `/api/health`
   - **Instance Type:** Starter (`2 GB RAM` recommended for fast PyTorch + LightGBM preloading)
5. Click **Create Web Service** to get a live `https://....onrender.com` URL.

### Option B: Railway
1. Go to [railway.app](https://railway.app) → **New Project** → **Deploy from GitHub repo**.
2. Select `naveensreekanth/y_adv`.
3. Railway automatically detects the included `Dockerfile`, builds both the React UI and Python ML backend, and exposes a public domain under **Settings → Networking → Generate Domain**.

### Option C: Docker / Google Cloud Run
```bash
# Build & run locally
docker build -t ai-yield-advisor .
docker run -p 8000:8000 ai-yield-advisor

# Or deploy to Google Cloud Run
gcloud run deploy ai-yield-advisor \
  --source . \
  --platform managed \
  --region us-central1 \
  --memory 2Gi \
  --cpu 2 \
  --allow-unauthenticated
```

### Option D: Hugging Face Spaces
1. Create a new Space on [huggingface.co/spaces](https://huggingface.co/spaces) using SDK **Docker**.
2. Connect `naveensreekanth/y_adv` and set `PORT=7860` in Space Variables (`app_port: 7860`) to get a free live demo URL with 16 GB RAM.
