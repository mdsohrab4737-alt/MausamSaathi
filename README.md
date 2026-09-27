# MAUSAMSAATHI Internal Hackathon Prototype

This is an offline-friendly local prototype for the SIH 26086 presentation/demo.

## What it demonstrates
- State → District → Block → Panchayat selection
- 7 / 14 / 21 / 30 day forecast horizon
- Probabilities for monsoon onset, break phase, heavy rain
- Rainfall anomaly + confidence
- Clickable colour-coded risk map
- Crop advisory (Rice / Maize)
- English / Hindi advisory view
- SMS / WhatsApp demo actions
- Browser voice advisory in English/Hindi
- FastAPI backend + React frontend

## Important
The forecast numbers and map are **illustrative prototype records**, not real operational predictions. Replace them later with trained-model outputs.

## Run backend
Open Terminal 1:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn main:app --reload --port 8000
```

If PowerShell blocks activation, run the backend with:

```powershell
python -m pip install -r requirements.txt
python -m uvicorn main:app --reload --port 8000
```

## Run frontend
Open Terminal 2:

```powershell
cd frontend
npm install
npm run dev
```

Then open the URL Vite shows, usually `http://localhost:5173`.

## Demo path
Use:
- State: Chhattisgarh
- District: Bilaspur
- Block: Kota
- Panchayat: Kendri
- Horizon: 14 Days
- Crop: Rice
- Language: हिंदी
- Click a map region
- Click Voice Call

## Production roadmap
1. Replace JSON with database-backed records.
2. Replace illustrative probabilities with trained/calibrated model outputs.
3. Replace illustrative map polygons with official Block/Panchayat boundaries.
4. Connect authenticated SMS/WhatsApp/voice providers.
5. Add monitoring, logging, model versioning and retraining.
