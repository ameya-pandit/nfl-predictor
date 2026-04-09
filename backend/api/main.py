"""
main.py
-------
FastAPI server for NFL predictions.
Reads from PostgreSQL and serves predictions and performance data.

Endpoints:
    GET /predictions?week=5   — matchups and predictions for a given week
    GET /performance?week=5   — weekly and cumulative record for a given week
"""

import os
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import psycopg2
import psycopg2.extras

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

DB_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/nfl")

app = FastAPI(title="NFL Predictor API")

# Allow the React frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# DB CONNECTION
# ---------------------------------------------------------------------------

def get_conn():
    return psycopg2.connect(DB_URL, cursor_factory=psycopg2.extras.RealDictCursor)


# ---------------------------------------------------------------------------
# ENDPOINTS
# ---------------------------------------------------------------------------

@app.get("/predictions")
def get_predictions(week: int = Query(..., description="NFL week number")):
    """
    Returns all matchups and predictions for a given week.
    If the week is completed, actual_margin and correct_pick are included.
    """
    try:
        conn = get_conn()
        cur  = conn.cursor()
        cur.execute("""
            SELECT
                week,
                home_team,
                away_team,
                predicted_winner,
                predicted_margin,
                confidence_tier,
                actual_margin,
                correct_pick
            FROM predictions
            WHERE season = 2025 AND week = %s
            ORDER BY confidence_tier DESC, ABS(predicted_margin) DESC
        """, (week,))
        rows = cur.fetchall()
        cur.close()
        conn.close()

        if not rows:
            raise HTTPException(status_code=404, detail=f"No predictions found for week {week}")

        return [dict(row) for row in rows]

    except psycopg2.Error as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/performance")
def get_performance(week: int = Query(..., description="NFL week number")):
    """
    Returns weekly and cumulative performance for a given week.
    Only meaningful if actual_margin is populated (week is completed).
    """
    try:
        conn = get_conn()
        cur  = conn.cursor()

        # Weekly record
        cur.execute("""
            SELECT
                COUNT(*)                            AS total_games,
                SUM(correct_pick)                   AS weekly_wins,
                COUNT(*) - SUM(correct_pick)        AS weekly_losses,
                ROUND(AVG(correct_pick)::numeric, 4) AS weekly_accuracy
            FROM predictions
            WHERE season = 2025
              AND week = %s
              AND actual_margin IS NOT NULL
        """, (week,))
        weekly = dict(cur.fetchone())

        # Cumulative record through this week
        cur.execute("""
            SELECT
                COUNT(*)                            AS total_games,
                SUM(correct_pick)                   AS cumulative_wins,
                COUNT(*) - SUM(correct_pick)        AS cumulative_losses,
                ROUND(AVG(correct_pick)::numeric, 4) AS cumulative_accuracy
            FROM predictions
            WHERE season = 2025
              AND week <= %s
              AND actual_margin IS NOT NULL
        """, (week,))
        cumulative = dict(cur.fetchone())

        cur.close()
        conn.close()

        if weekly["total_games"] == 0:
            raise HTTPException(status_code=404, detail=f"No completed games found for week {week}")

        return {
            "week":                week,
            "weekly_wins":         int(weekly["weekly_wins"] or 0),
            "weekly_losses":       int(weekly["weekly_losses"] or 0),
            "weekly_accuracy":     float(weekly["weekly_accuracy"] or 0),
            "cumulative_wins":     int(cumulative["cumulative_wins"] or 0),
            "cumulative_losses":   int(cumulative["cumulative_losses"] or 0),
            "cumulative_accuracy": float(cumulative["cumulative_accuracy"] or 0),
        }

    except psycopg2.Error as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    """Simple health check."""
    return {"status": "ok"}