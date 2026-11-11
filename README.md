# AI-Optimized Yield Advisor

An explainable semiconductor chip-test yield intelligence application. Drop any CSV or Excel (`.xlsx`) wafer sort or yield summary file into the browser UI and get plain-English root causes, interactive wafer defect maps with AI Grad-CAM overlays, hidden burn-in risk screens, test-time savings, and prioritized engineering actions.

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

## Project Structure

```text
├── backend/
│   ├── app/
│   │   ├── engines/           # 6 ML engines + Advisor ranking & Q&A
│   │   ├── ingest.py          # Universal CSV / multi-sheet Excel parser + auto column mapper
│   │   ├── pipeline.py        # End-to-end orchestrator + session cache
│   │   └── main.py            # FastAPI routes + static UI serving
│   ├── models/                # Pre-trained PyTorch Wafer CNN (wafer_cnn.pt)
│   └── train_wafer_model.py   # Training & hold-out lot evaluation script
├── frontend/                  # Minimalist React + Vite daylight UI (built into frontend/dist)
├── sample_uploads/            # Ready-to-drop demo Excel & CSV datasets
├── create_samples.py          # Generates sample upload bundles from AX7-SOC data
├── verify_e2e.py              # Automated end-to-end test suite
├── run.py                     # One-command local launcher
├── requirements.txt
├── Dockerfile
└── render.yaml
```

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
This automatically selects an available local port (default **http://127.0.0.1:8080**; if busy, it picks the next free port) and opens the UI in your browser.

### 4. Try Uploading Test Data
Drag and drop any file from `sample_uploads/` directly onto the upload box:
- `sample_uploads/AX7_Complete_Demo_8_Lots.xlsx` — single multi-sheet Excel workbook (`sort_die_results`, `lot_context_mes`, `wat_pcm`, `test_limits`, `final_test_burnin`)
- `sample_uploads/AX7_Sort_Sample_14_Lots.csv` — standalone CSV test log
- `sample_uploads/AX7_Wafer_Summary.csv` — lightweight wafer-summary CSV

Or click **Run Built-in Demo Dataset** on the home screen.

### (Optional) Frontend Development Mode / Verification Tests
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

Because the FastAPI backend serves the pre-built `frontend/dist` directly on a single port (`$PORT`), you can deploy it as one unified web service with zero separate frontend hosting required.

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
   - **Instance Type:** Starter (`2 GB RAM` recommended if uploading large 100k+ row Excel/CSV files; free `512 MB` works for small CSVs/demo files)
5. Click **Create Web Service**. Once live, Render gives you a public `https://....onrender.com` URL.

---

### Option B: Railway
1. Go to [railway.app](https://railway.app) → **New Project** → **Deploy from GitHub repo**.
2. Select `naveensreekanth/y_adv`.
3. Railway automatically detects the included `Dockerfile`, builds both the React UI and Python ML backend, and exposes a public domain under **Settings → Networking → Generate Domain**.

---

### Option C: Docker (Any Cloud VM, AWS App Runner, or Google Cloud Run)
```bash
# Build image locally
docker build -t ai-yield-advisor .

# Run container
docker run -p 8000:8000 ai-yield-advisor
```

**Deploy directly to Google Cloud Run:**
```bash
gcloud run deploy ai-yield-advisor \
  --source . \
  --platform managed \
  --region us-central1 \
  --memory 2Gi \
  --cpu 2 \
  --allow-unauthenticated
```

---

### Option D: Hugging Face Spaces (Free Public Hosting with 16 GB RAM)
1. Create a new Space on [huggingface.co/spaces](https://huggingface.co/spaces) with SDK **Docker**.
2. Push this repo (or connect GitHub) and add a `README.md` frontmatter header setting `app_port: 8000` (or set `PORT=7860` in Space Variables).
3. Hugging Face builds the `Dockerfile` automatically and gives you a live shareable demo URL with plenty of RAM for large CSV/Excel uploads.
