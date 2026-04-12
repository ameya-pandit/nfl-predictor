# NFL Predictor

A full-stack machine learning system that predicts NFL game outcomes using a Siamese neural network trained on historical play-by-play data. Achieved **65% accuracy** on the 2025 season (156W – 81L).

**Live site:** [nfl-predictor-mu.vercel.app](https://nfl-predictor-mu.vercel.app)

---

## How it works

The model takes two teams' rolling performance profiles and predicts the point margin of their matchup. A "team profile" is a vector of ~14 exponentially weighted stats — EPA, CPOE, redzone %, pressure rate, turnovers per drive, special teams EPA, and more — calculated from all games played so far that season.

The architecture is a **Siamese network**: both teams pass through the same encoder, producing a single power rating each. The difference between ratings, combined with contextual features (home/away, divisional game), feeds into a predictor head that outputs the expected margin.

---

## Architecture

```
Your machine          Railway (cloud)           Vercel (cloud)
─────────────         ───────────────           ──────────────
train.py          →   pipeline container    →   React frontend
(train model)         (inference + write)        (dashboard)
                           ↕
                      PostgreSQL
                      (predictions)
                           ↕
                      FastAPI
                      (REST API)
```

- **`train.py`** — fetches 2020–2024 NFL play-by-play data via `nfl_data_py`, engineers rolling features, trains the model, and saves versioned artifacts (`model_vX.pt`, `scaler_vX.pkl`)
- **`pipeline.py`** — loads a saved model, fetches 2025 data, runs inference on every matchup, and writes predictions + team profiles to PostgreSQL
- **FastAPI** — serves predictions and weekly performance via a REST API
- **React dashboard** — displays week-by-week matchup predictions, confidence tiers, and model performance

---

## Tech stack

| Layer | Technology |
|---|---|
| ML | Python, PyTorch, Pandas, NumPy, scikit-learn |
| Backend | FastAPI, PostgreSQL, Docker |
| Frontend | React, Tailwind CSS, Vite |
| Hosting | Railway (API + DB), Vercel (frontend) |
| Tooling | Git, nfl_data_py |

---

## Model details

- **Architecture:** Siamese neural network (`SiameseNFL_v2`)
- **Input:** 14 rolling features per team + 2 context features (home/away, divisional)
- **Output:** Predicted point margin (positive = home team wins)
- **Training data:** 2020–2023 NFL seasons
- **Validation:** 2024 season (68% val accuracy)
- **Test set:** 2025 season (65% accuracy, 156W–81L)
- **Confidence tiers:** High (>7 pts, 78% accurate), Moderate (3–7 pts, 64%), Toss-up (<3 pts, 53%)

---

## Project structure

```
nfl-predictor/
├── backend/
│   ├── pipeline/
│   │   ├── pipeline.py        # inference + database writes
│   │   ├── train.py           # model training
│   │   ├── model_v1_65pct.pt  # saved model weights
│   │   ├── scaler_v1_65pct.pkl
│   │   ├── requirements.txt
│   │   └── Dockerfile
│   └── api/
│       ├── main.py            # FastAPI server
│       ├── requirements.txt
│       └── Dockerfile
├── frontend/
│   └── src/
│       └── App.jsx            # React dashboard
├── docker-compose.yml
└── scheduler.ini              # weekly pipeline schedule (next season)
```

---

## Updating the model

1. Run `train.py` locally to produce new versioned artifacts
2. Update `MODEL_PATH` and `SCALER_PATH` in `pipeline.py`
3. Update `COPY` lines in `backend/pipeline/Dockerfile`
4. Push to GitHub — Railway automatically rebuilds and reruns the pipeline
5. New predictions are written to PostgreSQL and reflected on the live site

---

## API endpoints

| Endpoint | Description |
|---|---|
| `GET /predictions?week=5` | All matchups and predictions for a given week |
| `GET /performance?week=5` | Weekly and cumulative model accuracy |
| `GET /health` | Health check |
