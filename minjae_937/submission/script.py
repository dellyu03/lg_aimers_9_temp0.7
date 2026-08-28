import os

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier


ID_COL = "row_id"
TARGET_COL = "control_success"
CAT_COLUMNS = [
    "top_bottom",
    "game_type",
    "base_state",
    "pitcher_id",
    "batter_id",
    "pitcher_hand",
    "batter_hand",
    "pitcher_team_id",
    "batter_team_id",
    "hand_matchup",
    "count_state",
    "recent_era",
    "game_type_era",
]
RECENT_RATE_COLUMNS = {
    "recent1_delta": "asof_pitcher_prev1_game_success_rate",
    "recent3_delta": "asof_pitcher_prev3_game_success_rate",
    "recent5_delta": "asof_pitcher_prev5_game_success_rate",
}

# XGBoost 다양성 후보: 2024 전체에서 가중치와 logit 보정을 공동 최적화했습니다.
# 순서: CatBoost depth8, HistGradientBoosting, decay0.45 CatBoost seed123,
#       팀 단위 target encoding을 사용하는 1-tree XGBoost
MODEL_WEIGHTS = np.array(
    [0.14395068, 0.19208417, 0.21364048, 0.45032467]
)
CALIBRATION_SLOPE = 2.0
CALIBRATION_INTERCEPT = -0.13282943247529952


def add_features(df, fix_cold_start=False):
    out = df.copy()
    long_rate = "asof_pitcher_success_rate"
    if long_rate in out.columns:
        for new_col, recent_col in RECENT_RATE_COLUMNS.items():
            if recent_col in out.columns:
                out[new_col] = out[recent_col] - out[long_rate]

    if "asof_pitcher_n" in out.columns:
        n = pd.to_numeric(out["asof_pitcher_n"], errors="coerce")
        out["pitcher_log_n"] = np.log1p(n.clip(lower=0))
        out["pitcher_cold_start"] = (n.fillna(0) < 100).astype("int8")
        if long_rate in out.columns:
            rate = pd.to_numeric(out[long_rate], errors="coerce")
            shrink_n = n.fillna(0).clip(lower=0) if fix_cold_start else n
            shrink_rate = rate.fillna(0.5) if fix_cold_start else rate
            for strength in (50.0, 200.0, 1000.0):
                out[f"pitcher_success_shrunk_{int(strength)}"] = (
                    shrink_n * shrink_rate + strength * 0.5
                ) / (shrink_n + strength)
        reliability = n / (n + 200.0)
        for delta_col in RECENT_RATE_COLUMNS:
            if delta_col in out.columns:
                out[f"{delta_col}_reliable"] = out[delta_col] * reliability

    bad_cols = [
        col
        for col in ["asof_pitcher_reverse_rate", "asof_pitcher_middle_rate"]
        if col in out.columns
    ]
    if bad_cols:
        out["bad_location_rate"] = out[bad_cols].sum(axis=1, min_count=1)

    if {"pitcher_hand", "batter_hand"}.issubset(out.columns):
        out["hand_matchup"] = (
            out["pitcher_hand"].astype("string")
            + "_"
            + out["batter_hand"].astype("string")
        )

    if "season" in out.columns:
        season = pd.to_numeric(out["season"], errors="coerce")
        out["recent_era"] = np.where(season >= 2023, "recent", "old")
        if "game_type" in out.columns:
            out["game_type_era"] = (
                out["recent_era"].astype("string")
                + "_"
                + out["game_type"].astype("string")
            )

    if {
        "asof_pitcher_success_rate",
        "asof_batter_success_rate",
    }.issubset(out.columns):
        out["pitcher_batter_success_gap"] = (
            out["asof_pitcher_success_rate"]
            - out["asof_batter_success_rate"]
        )

    if {
        "asof_pitcher_prev1_game_success_rate",
        "asof_pitcher_prev5_game_success_rate",
    }.issubset(out.columns):
        out["recent1_vs_recent5"] = (
            out["asof_pitcher_prev1_game_success_rate"]
            - out["asof_pitcher_prev5_game_success_rate"]
        )

    if {"balls_before", "strikes_before"}.issubset(out.columns):
        out["count_state"] = (
            out["balls_before"].astype("string")
            + "-"
            + out["strikes_before"].astype("string")
        )
    return out


def prepare_model_input(df, fix_cold_start=False):
    out = add_features(df, fix_cold_start=fix_cold_start)
    out = out.drop(
        columns=[col for col in [ID_COL, TARGET_COL] if col in out.columns]
    )
    for col in CAT_COLUMNS:
        if col in out.columns:
            out[col] = out[col].astype("string").fillna("__MISSING__").astype(str)
    return out


def add_target_encodings(df, payload):
    """공식 학습 데이터로만 만든 ID별 사전확률을 평가 행에 붙입니다."""
    out = df.copy()
    encoding = payload["target_encoding"]
    prior = float(encoding["prior"])
    for key, mapping in encoding["mappings"].items():
        out[f"te_{key}"] = out[key].map(mapping).fillna(prior)
    return out


def main():
    test = pd.read_csv("./data/test.csv", encoding="utf-8-sig")
    sample = pd.read_csv("./data/sample_submission.csv", encoding="utf-8-sig")
    if ID_COL not in test.columns:
        raise ValueError(f"test.csv에 {ID_COL}가 없습니다.")

    legacy_features = prepare_model_input(test, fix_cold_start=False)
    fixed_features = prepare_model_input(test, fix_cold_start=True)
    cat_d8 = CatBoostClassifier()
    cat_d8.load_model("./model/cat_d8.cbm")
    histboost = joblib.load("./model/histboost.joblib")
    cat_decay = CatBoostClassifier()
    cat_decay.load_model("./model/cat_decay045_seed123.cbm")
    xgboost_payload = joblib.load("./model/xgboost.joblib")

    p_d8 = cat_d8.predict_proba(
        legacy_features[cat_d8.feature_names_]
    )[:, 1]
    p_hist = histboost.predict_proba(legacy_features)[:, 1]
    p_decay = cat_decay.predict_proba(
        fixed_features[cat_decay.feature_names_]
    )[:, 1]
    xgboost_input = add_target_encodings(test, xgboost_payload)
    xgboost_features = prepare_model_input(
        xgboost_input, fix_cold_start=True
    )
    xgboost_array = xgboost_payload["preprocessor"].transform(
        xgboost_features
    ).astype("float32")
    p_xgboost = xgboost_payload["model"].predict_proba(
        xgboost_array
    )[:, 1]
    probability = (
        MODEL_WEIGHTS[0] * p_d8
        + MODEL_WEIGHTS[1] * p_hist
        + MODEL_WEIGHTS[2] * p_decay
        + MODEL_WEIGHTS[3] * p_xgboost
    )
    clipped = np.clip(probability, 1e-6, 1.0 - 1e-6)
    logit = np.log(clipped / (1.0 - clipped))
    probability = 1.0 / (
        1.0 + np.exp(-(CALIBRATION_SLOPE * logit + CALIBRATION_INTERCEPT))
    )

    prediction = pd.DataFrame(
        {ID_COL: test[ID_COL].to_numpy(), TARGET_COL: probability}
    )
    output = sample[[ID_COL]].merge(
        prediction, on=ID_COL, how="left", validate="one_to_one"
    )
    if output[TARGET_COL].isna().any():
        raise ValueError("예측이 누락된 row_id가 있습니다.")
    os.makedirs("./output", exist_ok=True)
    output.to_csv("./output/submission.csv", index=False, encoding="utf-8")
    print(
        f"Saved ./output/submission.csv rows={len(output)} "
        f"mean={output[TARGET_COL].mean():.6f}"
    )


if __name__ == "__main__":
    main()
