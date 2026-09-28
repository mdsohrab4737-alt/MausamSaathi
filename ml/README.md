# MAUSAMSAATHI ML Proof-of-Concept

This module demonstrates a real train → validate → calibrate → predict workflow for SIH 26086.

## What is real in this demo

- A real `XGBClassifier` is trained for three targets: onset, break and heavy rain.
- Data is split chronologically into 70% train, 15% validation and 15% test.
- Validation predictions are calibrated with `IsotonicRegression`.
- Test metrics include Brier score, log loss, ROC-AUC, PR-AUC and F1.
- Trained models and calibration artifacts are saved in `models/`.

## Important limitation

The bundled CSV is **synthetic demonstration data**. The event labels are also simple proxy rules created only to make the pipeline executable for an internal hackathon demo. They are not official meteorological definitions.

For production, replace the CSV and label-generation step with observed rainfall / atmospheric data and documented onset, break and heavy-rain event criteria.

## Run

From this `ml` directory:

```powershell
python train_model.py
```

Then use:

```powershell
python -c "from predict import predict_probabilities; print(predict_probabilities({'date':'2025-07-15','rainfall_1d':28,'rainfall_3d':62,'rainfall_7d':98,'rainy_days_7d':5,'temperature_c':28,'humidity_pct':78,'wind_ms':4.2,'pressure_hpa':1004,'enso_index':-0.4,'iod_index':0.3,'mjo_index':0.9,'rainfall_anomaly_pct':22}))"
```

## Judge explanation

`raw observations → engineered features → event labels → chronological train/validation/test → XGBoost → probability calibration → probabilistic output → API/dashboard`
