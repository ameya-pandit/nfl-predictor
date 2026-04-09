"""
train.py
--------
Trains the SiameseNFL_v2 model and saves versioned artifacts.
Run this manually when you want to produce a new model.

Usage:
    python train.py

Outputs:
    model_v{VERSION}.pt
    scaler_v{VERSION}.pkl
"""

import os
import joblib
import pandas as pd
import numpy as np
import nfl_data_py as nfl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# CONFIG — tweak these when experimenting
# ---------------------------------------------------------------------------

TRAIN_YEARS   = list(range(2020, 2024))  # 2020–2023
VAL_YEAR      = 2024
ALL_YEARS     = TRAIN_YEARS + [VAL_YEAR]

NUM_EPOCHS    = 50
BATCH_SIZE    = 32
LEARNING_RATE = 0.0005
EWM_SPAN      = 16
RANDOM_SEED   = 42                       # change this to experiment

VERSION       = "v2"               # update when saving a new model

FEATURE_COLUMNS = [
    "off_epa",
    "off_cpoe",
    "off_qb_pressure",
    "off_redzone_pct",
    "off_crucial_down_pct",
    "off_pen_yards_per_snap",
    "off_turnovers_per_drive",
    "def_epa_allowed",
    "def_turnovers_forced_per_drive",
    "def_redzone_pct_allowed",
    "def_pressure_rate_given",
    "def_crucial_down_pct_allowed",
    "st_epa",
    "st_success_rate",
]

CONTEXT_COLS = ["is_home", "is_divisional"]

RELOCATION_MAP = {"STL": "LA", "SD": "LAC", "OAK": "LV"}

TZ_MAP = {
    "SEA": -3, "SF": -3, "LA": -3, "LAC": -3,
    "ARI": -2, "LV": -2, "DEN": -2,
    "KC": -1, "DAL": -1, "HOU": -1, "NO": -1,
    "CHI": -1, "MIN": -1, "TEN": -1, "IND": -1, "GB": -1,
    "ATL": 0, "BUF": 0, "CAR": 0, "CIN": 0, "CLE": 0,
    "DET": 0, "JAX": 0, "MIA": 0, "NE": 0, "NYG": 0,
    "NYJ": 0, "PHI": 0, "PIT": 0, "TB": 0, "WAS": 0, "BAL": 0,
}

# ---------------------------------------------------------------------------
# MODEL
# ---------------------------------------------------------------------------

class SiameseNFL_v2(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 8),
            nn.LeakyReLU(),
            nn.Linear(8, 1),
        )
        self.predictor = nn.Sequential(
            nn.Linear(3, 8),
            nn.LeakyReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, x_team, x_opp, x_ctx):
        r_team   = self.encoder(x_team)
        r_opp    = self.encoder(x_opp)
        diff     = r_team - r_opp
        combined = torch.cat((diff, x_ctx), dim=1)
        return self.predictor(combined)


# ---------------------------------------------------------------------------
# FEATURE ENGINEERING (shared with pipeline.py via import)
# ---------------------------------------------------------------------------

def fetch_data(years):
    print("Fetching play-by-play data...")
    pbp = nfl.import_pbp_data(years)
    print("Fetching schedule data...")
    sched = nfl.import_schedules(years)
    sched["gameday"] = pd.to_datetime(sched["gameday"])
    return pbp, sched


def build_offensive_features(pbp):
    run_pass = pbp[pbp["play_type"].isin(["run", "pass"])].copy()
    run_pass = run_pass.dropna(subset=["epa"])
    pbp["turnover"]      = pbp["interception"].fillna(0) + pbp["fumble_lost"].fillna(0)
    run_pass["turnover"] = run_pass["interception"].fillna(0) + run_pass["fumble_lost"].fillna(0)

    epa_df = (
        run_pass.groupby(["season", "week", "posteam"])
        .agg(off_epa=("epa", "mean"))
        .reset_index().rename(columns={"posteam": "off_team"})
    )
    qb_df = (
        run_pass.groupby(["season", "week", "posteam"])
        .agg(
            off_cpoe=("cpoe", "mean"),
            off_qb_pressure=("qb_hit", lambda x: x.sum() / max(1,
                run_pass.loc[x.index, "pass_attempt"].sum() +
                run_pass.loc[x.index, "sack"].sum()
            )),
        )
        .reset_index().rename(columns={"posteam": "off_team"})
    )
    all_drives = (
        run_pass[run_pass["posteam"].notna()]
        .groupby(["season", "week", "posteam", "fixed_drive"], as_index=False)
        .agg(drive_inside20=("drive_inside20", "max"), drive_result=("fixed_drive_result", "last"))
    )
    rz = all_drives[all_drives["drive_inside20"] == 1].copy()
    rz["rz_td"] = rz["drive_result"].str.contains("Touchdown", na=False).astype(int)
    rz_df = rz.groupby(["season", "week", "posteam"], as_index=False).agg(
        redzone_drives=("fixed_drive", "count"), redzone_tds=("rz_td", "sum")
    )
    team_weeks = run_pass[run_pass["posteam"].notna()][["season", "week", "posteam"]].drop_duplicates()
    rz_df = team_weeks.merge(rz_df, on=["season", "week", "posteam"], how="left").fillna(0)
    rz_df["off_redzone_pct"] = rz_df["redzone_tds"] / rz_df["redzone_drives"].replace(0, 1)
    rz_df = rz_df.rename(columns={"posteam": "off_team"})

    pct_df = (
        run_pass.groupby(["season", "week", "posteam"])
        .agg(
            off_success=("success", "mean"),
            third_down_converted=("third_down_converted", "sum"),
            third_down_failed=("third_down_failed", "sum"),
            fourth_down_converted=("fourth_down_converted", "sum"),
            fourth_down_failed=("fourth_down_failed", "sum"),
        )
        .reset_index().rename(columns={"posteam": "off_team"})
    )
    pct_df["off_crucial_down_pct"] = (
        (pct_df["third_down_converted"] + pct_df["fourth_down_converted"]) /
        (pct_df["third_down_converted"] + pct_df["third_down_failed"] +
         pct_df["fourth_down_converted"] + pct_df["fourth_down_failed"])
    ).fillna(0)
    pct_df = pct_df.merge(rz_df[["season", "week", "off_team", "off_redzone_pct"]], on=["season", "week", "off_team"], how="left")

    off_plays = (
        pbp.groupby(["season", "week", "posteam"])
        .agg(off_pass_plays=("pass_attempt", "sum"), off_run_plays=("rush_attempt", "sum"))
        .reset_index().rename(columns={"posteam": "off_team"})
    )
    off_plays["off_total_plays"] = off_plays["off_pass_plays"] + off_plays["off_run_plays"]
    def_plays = (
        pbp.groupby(["season", "week", "defteam"])
        .agg(def_pass_plays=("pass_attempt", "sum"), def_run_plays=("rush_attempt", "sum"))
        .reset_index().rename(columns={"defteam": "team"})
    )
    def_plays["def_total_plays"] = def_plays["def_pass_plays"] + def_plays["def_run_plays"]
    total_plays = off_plays.rename(columns={"off_team": "team"}).merge(def_plays, on=["season", "week", "team"], how="left")
    total_plays["total_snaps"] = total_plays["off_total_plays"] + total_plays["def_total_plays"]

    valid_off = pbp[pbp["play_type"].isin(["run", "pass", "qb_kneel", "qb_spike"])]
    drive_df = valid_off[valid_off["posteam"].notna()].groupby(["season", "week", "posteam", "fixed_drive"], as_index=False).size()
    off_drives = drive_df.groupby(["season", "week", "posteam"]).agg(off_drives=("fixed_drive", "count")).reset_index().rename(columns={"posteam": "team"})

    turnover_df = run_pass.groupby(["season", "week", "posteam"]).agg(off_turnovers=("turnover", "sum")).reset_index().rename(columns={"posteam": "team"})
    turnover_df = turnover_df.merge(off_drives, on=["season", "week", "team"], how="left")
    turnover_df["off_turnovers_per_drive"] = turnover_df["off_turnovers"] / turnover_df["off_drives"].replace(0, 1)

    pen_df = (
        pbp[(pbp["penalty"] == 1) & (pbp["penalty_team"].notna())]
        .groupby(["season", "week", "penalty_team"]).agg(penalty_yards=("penalty_yards", "sum"))
        .reset_index().rename(columns={"penalty_team": "team"})
    )
    pen_df = pen_df.merge(total_plays[["season", "week", "team", "total_snaps"]], on=["season", "week", "team"], how="left")
    pen_df["off_pen_yards_per_snap"] = pen_df["penalty_yards"] / pen_df["total_snaps"].replace(0, 1)

    gamescript_df = (
        off_plays
        .merge(pen_df[["season", "week", "team", "off_pen_yards_per_snap"]], left_on=["season", "week", "off_team"], right_on=["season", "week", "team"], how="left").drop(columns="team")
        .merge(turnover_df[["season", "week", "team", "off_turnovers_per_drive"]], left_on=["season", "week", "off_team"], right_on=["season", "week", "team"], how="left")
        .drop(columns=["team", "off_total_plays", "off_run_plays", "off_pass_plays"])
    )

    off_df = (
        epa_df
        .merge(qb_df, on=["season", "week", "off_team"], how="left")
        .merge(pct_df[["season", "week", "off_team", "off_success", "off_crucial_down_pct", "off_redzone_pct"]], on=["season", "week", "off_team"], how="left")
        .merge(gamescript_df, on=["season", "week", "off_team"], how="left")
    )
    off_df = off_df.fillna({"off_pen_yards_per_snap": 0, "off_turnovers_per_drive": 0})
    return off_df, run_pass, pbp


def build_defensive_features(off_df, pbp):
    matchups = (
        pbp[pbp["posteam"].notna() & pbp["defteam"].notna()]
        [["season", "week", "posteam", "defteam"]].drop_duplicates()
        .rename(columns={"posteam": "off_team", "defteam": "def_team"})
    )
    def_df = matchups.merge(off_df, on=["season", "week", "off_team"], how="left").drop(columns="off_team")
    def_df = def_df.rename(columns={
        "off_epa": "def_epa_allowed", "off_cpoe": "def_cpoe_allowed",
        "off_qb_pressure": "def_pressure_rate_given", "off_success": "def_success_allowed",
        "off_crucial_down_pct": "def_crucial_down_pct_allowed", "off_redzone_pct": "def_redzone_pct_allowed",
        "off_pen_yards_per_snap": "def_pen_yards_per_snap", "off_turnovers_per_drive": "def_turnovers_forced_per_drive",
    })
    def_features = ["season", "week", "def_team", "def_epa_allowed", "def_pressure_rate_given",
                    "def_success_allowed", "def_crucial_down_pct_allowed", "def_redzone_pct_allowed", "def_turnovers_forced_per_drive"]
    return def_df[[c for c in def_features if c in def_df.columns]], matchups


def build_special_teams(pbp):
    pbp["calc_dist"] = pbp["yardline_100"] + 18
    st_plays = pbp[pbp["play_type"].isin(["kickoff", "punt", "field_goal", "extra_point"])].copy()
    st_plays["kicking_power"] = (st_plays["calc_dist"] / 10) * (st_plays["field_goal_result"] == "made")
    return (
        st_plays.groupby(["season", "week", "posteam"])
        .agg(st_epa=("epa", "sum"), st_success_rate=("success", "mean"),
             st_kicking_power=("kicking_power", "sum"), st_punt_inside_twenty=("punt_inside_twenty", "sum"))
        .reset_index().rename(columns={"posteam": "team"})
    )


def build_gamelogs(off_df, def_df, st_df, matchups, pbp):
    gamelogs = (
        off_df
        .merge(def_df, left_on=["season", "week", "off_team"], right_on=["season", "week", "def_team"], how="left")
        .drop(columns="def_team").rename(columns={"off_team": "team"})
    )
    gamelogs = gamelogs.merge(matchups.rename(columns={"off_team": "team", "def_team": "opponent"}), on=["season", "week", "team"], how="left")
    gamelogs = gamelogs.merge(st_df, on=["season", "week", "team"], how="left").fillna(0)

    final_scores = (
        pbp.sort_values(["season", "week", "game_id", "play_id"]).groupby("game_id").tail(1)
        [["season", "week", "game_id", "home_team", "away_team", "total_home_score", "total_away_score"]]
        .reset_index(drop=True)
    )
    home_rows = final_scores.copy()
    home_rows["team"] = home_rows["home_team"]; home_rows["opponent"] = home_rows["away_team"]
    home_rows["point_diff"] = home_rows["total_home_score"] - home_rows["total_away_score"]
    away_rows = final_scores.copy()
    away_rows["team"] = away_rows["away_team"]; away_rows["opponent"] = away_rows["home_team"]
    away_rows["point_diff"] = away_rows["total_away_score"] - away_rows["total_home_score"]
    point_diff = pd.concat([home_rows[["season", "week", "team", "opponent", "point_diff"]], away_rows[["season", "week", "team", "opponent", "point_diff"]]])

    gamelogs = gamelogs.drop(columns=["point_diff"], errors="ignore")
    return gamelogs.merge(point_diff, on=["season", "week", "team", "opponent"], how="left")


def build_rolling_features(gamelogs):
    gamelogs = gamelogs.sort_values(["team", "season", "week"])
    available = [c for c in FEATURE_COLUMNS if c in gamelogs.columns]
    rolling = (
        gamelogs.groupby(["team", "season"], group_keys=False)[available]
        .apply(lambda x: x.shift(1).ewm(span=EWM_SPAN, adjust=True).mean())
    )
    rolling = rolling.add_prefix("rolling_")
    team_state = pd.concat([gamelogs[["season", "week", "team", "opponent", "point_diff"]], rolling], axis=1)
    return team_state[team_state["week"] > 3].dropna()


def build_context(sched):
    home = sched[["season", "week", "home_team", "gameday"]].rename(columns={"home_team": "team"})
    away = sched[["season", "week", "away_team", "gameday"]].rename(columns={"away_team": "team"})
    combined = pd.concat([home, away]).sort_values(["team", "gameday"])
    combined["prev_gameday"] = combined.groupby("team")["gameday"].shift(1)
    combined["rest"] = (combined["gameday"] - combined["prev_gameday"]).dt.days.fillna(7).clip(upper=21)
    team_rest = combined[["season", "week", "team", "rest"]]

    sched_info = sched[["season", "week", "home_team", "away_team", "div_game"]].copy()
    sched_info = sched_info.merge(team_rest, left_on=["season", "week", "home_team"], right_on=["season", "week", "team"]).rename(columns={"rest": "rest_home"}).drop(columns="team")
    sched_info = sched_info.merge(team_rest, left_on=["season", "week", "away_team"], right_on=["season", "week", "team"]).rename(columns={"rest": "rest_away"}).drop(columns="team")

    home_view = sched_info.rename(columns={"home_team": "team", "away_team": "opp"})
    home_view["is_home"] = 1.0; home_view["rest_val"] = home_view["rest_home"]; home_view["opp_rest_val"] = home_view["rest_away"]; home_view["tz_diff"] = 0.0
    away_view = sched_info.rename(columns={"away_team": "team", "home_team": "opp"})
    away_view["is_home"] = 0.0; away_view["rest_val"] = away_view["rest_away"]; away_view["opp_rest_val"] = away_view["rest_home"]
    away_view["tz_diff"] = away_view["team"].map(TZ_MAP).fillna(0) - away_view["opp"].map(TZ_MAP).fillna(0)

    context = pd.concat([home_view, away_view])[["season", "week", "team", "opp", "is_home", "rest_val", "opp_rest_val", "div_game", "tz_diff"]]
    context["team"] = context["team"].replace(RELOCATION_MAP)
    context["opp"]  = context["opp"].replace(RELOCATION_MAP)
    return context


def build_game_model_df(team_state, context):
    game_model = team_state.merge(team_state, left_on=["season", "week", "opponent"], right_on=["season", "week", "team"], suffixes=("_team", "_opp"))
    game_model = game_model.drop(columns=["team", "opp", "is_home", "rest_val", "opp_rest_val", "div_game", "rest_diff", "is_divisional"], errors="ignore")
    game_model = game_model.merge(context, left_on=["season", "week", "team_team", "opponent_team"], right_on=["season", "week", "team", "opp"], how="left")
    game_model["rest_diff"]     = (game_model["rest_val"] - game_model["opp_rest_val"]).fillna(0)
    game_model["is_divisional"] = game_model["div_game"].fillna(0)
    game_model["is_home"]       = game_model["is_home"].fillna(0)
    return game_model


# ---------------------------------------------------------------------------
# TRAIN
# ---------------------------------------------------------------------------

def train(game_model):
    torch.manual_seed(RANDOM_SEED)

    team_cols = [c for c in game_model.columns if "rolling_" in c and "_team" in c]
    opp_cols  = [c for c in game_model.columns if "rolling_" in c and "_opp"  in c]

    train_df = game_model[game_model["season"] < VAL_YEAR]
    val_df   = game_model[game_model["season"] == VAL_YEAR]

    scaler = StandardScaler()
    scaler.fit(train_df[team_cols].values)

    def to_tensors(df):
        return (
            torch.tensor(scaler.transform(df[team_cols].values), dtype=torch.float32),
            torch.tensor(scaler.transform(df[opp_cols].values),  dtype=torch.float32),
            torch.tensor(df[CONTEXT_COLS].values,                dtype=torch.float32),
            torch.tensor(df["point_diff_team"].values,           dtype=torch.float32).unsqueeze(1),
        )

    X_train_t, X_train_o, X_train_ctx, y_train = to_tensors(train_df)
    X_val_t,   X_val_o,   X_val_ctx,   y_val   = to_tensors(val_df)
    train_loader = DataLoader(TensorDataset(X_train_t, X_train_o, X_train_ctx, y_train), batch_size=BATCH_SIZE, shuffle=True)

    model     = SiameseNFL_v2(input_dim=len(team_cols))
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    criterion = nn.HuberLoss(delta=1.0)

    print("Training...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        for b_t, b_o, b_ctx, b_y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(b_t, b_o, b_ctx), b_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_t, X_val_o, X_val_ctx)
            acc = ((val_preds > 0) == (y_val > 0)).float().mean()
        if epoch % 5 == 0 or epoch == NUM_EPOCHS - 1:
            print(f"  Epoch {epoch:>2} | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {acc:.2%}")

    print("Training complete.")
    return model, scaler, team_cols


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pbp, sched = fetch_data(ALL_YEARS)
    off_df, _, pbp = build_offensive_features(pbp)
    def_df, matchups = build_defensive_features(off_df, pbp)
    st_df = build_special_teams(pbp)
    gamelogs = build_gamelogs(off_df, def_df, st_df, matchups, pbp)
    team_state = build_rolling_features(gamelogs)
    context = build_context(sched)
    game_model = build_game_model_df(team_state, context)

    model, scaler, team_cols = train(game_model)

    model_path  = f"model_{VERSION}.pt"
    scaler_path = f"scaler_{VERSION}.pkl"
    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Saved {model_path} and {scaler_path}")