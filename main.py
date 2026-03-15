# ai_supreme_with_stacker.py
from flask import Flask, jsonify
import requests
import threading
import time
import os
import numpy as np
import random
import warnings

warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_predict

# Try optional boosters
try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:
    HAS_LGB = False

app = Flask(__name__)

API = "https://phanmemdudoan.fun/apisun.php"  # your data API

history_file = "history.txt"

window = 12
max_history = 2000

latest_period = None
latest_result = None
latest_total = None

prediction = None
confidence = None
dataset_size = 0

status = "BOT đang khởi động"

last_period = None

# engine predictions (binary) for introspection
engine_predictions = {}

# =====================
# NORMALIZE RESULT
# =====================
def normalize(r):
    r = str(r).lower()
    if "tai" in r:
        return 1
    if "xiu" in r:
        return 0
    # if it's numeric (total of dice), treat total >= 11 -> Tài (example) — but original uses labels
    try:
        v = int(r)
        # fallback heuristic: treat >=11 as TÀI for 3 dice (4..17), but better to rely on API labels
        return 1 if v >= 11 else 0
    except:
        return 0

# =====================
# HISTORY
# =====================
def read_history():
    data = []
    if not os.path.exists(history_file):
        return data
    with open(history_file) as f:
        for line in f:
            p = line.strip().split(",")
            if len(p) < 3:
                continue
            data.append(normalize(p[1]))
    # keep newest max_history entries
    return data[-max_history:]


def save_history(period, result, total):
    lines = []
    if os.path.exists(history_file):
        with open(history_file) as f:
            lines = f.readlines()
    lines.append(f"{period},{result},{total}\n")
    if len(lines) > max_history:
        lines = lines[-max_history:]
    with open(history_file, "w") as f:
        f.writelines(lines)


# =====================
# FEATURE ENGINEERING (expanded)
# =====================
def features(seq):
    """Return a list of numeric features computed from the window sequence (list of 0/1)."""
    arr = np.array(seq, dtype=float)
    # basic stats
    ssum = float(arr.sum())
    mean = float(arr.mean())
    std = float(arr.std())
    mx = float(arr.max())
    mn = float(arr.min())
    # moving averages / counts
    last3 = float(arr[-3:].sum()) if len(arr) >= 3 else ssum
    last6 = float(arr[-6:].sum()) if len(arr) >= 6 else ssum
    runs = run_lengths(arr)  # returns list of run lengths
    longest_run = float(max(runs)) if len(runs) > 0 else 0.0
    last_run = float(runs[-1]) if len(runs) > 0 else 0.0
    # transitions
    transitions = float(((arr[1:] - arr[:-1]) != 0).sum()) if len(arr) > 1 else 0.0
    # autocorrelation lag1
    ac1 = 0.0
    if len(arr) > 1:
        ac1 = float(np.corrcoef(arr[:-1], arr[1:])[0, 1]) if np.std(arr[:-1]) > 0 and np.std(arr[1:]) > 0 else 0.0
    # combine
    return [
        ssum,
        mean,
        std,
        mx,
        mn,
        last3,
        last6,
        longest_run,
        last_run,
        transitions,
        ac1,
    ]


def run_lengths(arr):
    """Return list of run lengths for consecutive identical values in arr."""
    if len(arr) == 0:
        return []
    runs = []
    cur = arr[0]
    length = 1
    for v in arr[1:]:
        if v == cur:
            length += 1
        else:
            runs.append(length)
            cur = v
            length = 1
    runs.append(length)
    return runs


# =====================
# BASE AI ENGINES
# (return probability of TÀI as float in [0,1])
# =====================

def ml_rf(history):
    X, y = build_dataset(history)
    if X is None:
        return 0.5
    model = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
    model.fit(X, y)
    last = np.array([make_last_feature(history)])
    return float(safe_proba_predict(model, last))


def ml_gb(history):
    X, y = build_dataset(history)
    if X is None:
        return 0.5
    model = GradientBoostingClassifier()
    model.fit(X, y)
    last = np.array([make_last_feature(history)])
    return float(safe_proba_predict(model, last))


def ml_lr(history):
    X, y = build_dataset(history)
    if X is None:
        return 0.5
    model = LogisticRegression(max_iter=500)
    model.fit(X, y)
    last = np.array([make_last_feature(history)])
    return float(safe_proba_predict(model, last))


def ml_et(history):
    X, y = build_dataset(history)
    if X is None:
        return 0.5
    model = ExtraTreesClassifier(n_estimators=200, n_jobs=-1, random_state=1)
    model.fit(X, y)
    last = np.array([make_last_feature(history)])
    return float(safe_proba_predict(model, last))


def ml_xgb(history):
    if not HAS_XGB:
        return 0.5
    X, y = build_dataset(history)
    if X is None:
        return 0.5
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", n_jobs=4)
    model.fit(X, y)
    last = np.array([make_last_feature(history)])
    return float(safe_proba_predict(model, last))


def ml_lgb(history):
    if not HAS_LGB:
        return 0.5
    X, y = build_dataset(history)
    if X is None:
        return 0.5
    model = lgb.LGBMClassifier(n_jobs=4)
    model.fit(X, y)
    last = np.array([make_last_feature(history)])
    return float(safe_proba_predict(model, last))


# =====================
# HELPERS for ML
# =====================
def build_dataset(history):
    """Return (X, y) suitable for training. X rows contain window sequence + engineered features."""
    n = len(history)
    if n < window + 2:
        return (None, None)
    X = []
    y = []
    # we want newest->oldest order to favor recent? but training uses chronological order
    for i in range(0, n - window):
        seq = history[i:i + window]
        feats = list(seq) + features(seq)
        X.append(feats)
        y.append(history[i + window])
    return (np.array(X), np.array(y))


def make_last_feature(history):
    seq = history[-window:]
    return list(seq) + features(seq)


def safe_proba_predict(model, X):
    """Return probability of positive class robustly."""
    try:
        proba = model.predict_proba(X)
        # ensure returning class 1 prob
        if proba.shape[1] == 2:
            return proba[:, 1][0]
        else:
            # fallback: if single column (rare), interpret as positive
            return float(proba[:, 0][0])
    except Exception:
        # fallback to model predict
        try:
            p = model.predict(X)[0]
            return 1.0 if p == 1 else 0.0
        except Exception:
            return 0.5


# =====================
# MARKOV / PATTERN / MISC engines
# =====================

def markov1(history):
    matrix = np.zeros((2, 2))
    for i in range(len(history) - 1):
        matrix[history[i]][history[i + 1]] += 1
    row = matrix[history[-1]]
    if row.sum() == 0:
        return 0.5
    return float(row[1] / row.sum())


def markov2(history):
    if len(history) < 3:
        return 0.5
    pair = history[-2:]
    count = 0
    win = 0
    for i in range(len(history) - 2):
        if history[i:i + 2] == pair:
            count += 1
            if history[i + 2] == 1:
                win += 1
    if count == 0:
        return 0.5
    return float(win / count)


def pattern_engine(history):
    wins = []
    weights = []
    n = len(history)
    for size in range(3, 11):
        if n < size + 1:
            continue
        seq = history[-size:]
        count = 0
        win = 0
        for i in range(n - size):
            if history[i:i + size] == seq:
                count += 1
                if i + size < n:
                    win += history[i + size]
        if count > 0:
            wins.append(win / count)
            weights.append(count)
    if len(wins) == 0:
        return 0.5
    return float(np.average(wins, weights=weights))


def monte_engine(history):
    p = float(sum(history) / len(history))
    trials = 20000
    wins = 0
    for _ in range(trials):
        if random.random() < p:
            wins += 1
    return float(wins / trials)


def market_engine(history):
    last50 = history[-50:] if len(history) >= 1 else history
    if len(last50) == 0:
        return 0.5
    return float(sum(last50) / len(last50))


def streak_engine(history):
    streak = 1
    for i in range(len(history) - 1, 0, -1):
        if history[i] == history[i - 1]:
            streak += 1
        else:
            break
    if streak >= 5:
        # conservative contrarian bias
        return 0.3 if history[-1] == 1 else 0.7
    return 0.5


def entropy_engine(history):
    p = float(sum(history) / len(history))
    entropy = -(p * np.log2(p + 1e-9) + (1 - p) * np.log2(1 - p + 1e-9))
    return float(p * (1 - entropy))


# =====================
# STACKER / META-LEARNER (new powerful engine)
# =====================
def stacker_engine(history):
    """
    Build out-of-fold predictions for base models and train a meta-learner (LogisticRegression).
    If dataset too small or an error occurs, fallback to simple average of base model predictions.
    """
    X, y = build_dataset(history)
    if X is None:
        return 0.5
    n_samples = X.shape[0]
    # Require at least some samples to do cv stacking
    if n_samples < 30:
        # fallback: average base models on last sample
        base_preds = []
        last = np.array([make_last_feature(history)])
        for fn in [ml_rf, ml_gb, ml_lr, ml_et, ml_xgb if HAS_XGB else None, ml_lgb if HAS_LGB else None]:
            if fn is None:
                continue
            try:
                base_preds.append(fn(history))
            except Exception:
                base_preds.append(0.5)
        if len(base_preds) == 0:
            return 0.5
        return float(np.mean(base_preds))
    try:
        # prepare simple base learners (scikit-learn compatible)
        base_models = []
        base_names = []
        rf = RandomForestClassifier(n_estimators=150, n_jobs=-1, random_state=1)
        gb = GradientBoostingClassifier()
        lr = LogisticRegression(max_iter=500)
        et = ExtraTreesClassifier(n_estimators=150, n_jobs=-1, random_state=2)
        base_models.extend([rf, gb, lr, et])
        base_names.extend(["rf", "gb", "lr", "et"])
        if HAS_XGB:
            xclf = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", n_jobs=4)
            base_models.append(xclf); base_names.append("xgb")
        if HAS_LGB:
            lclf = lgb.LGBMClassifier(n_jobs=4)
            base_models.append(lclf); base_names.append("lgb")

        # out-of-fold predictions for each base model
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        meta_features = np.zeros((n_samples, len(base_models)))
        for idx, m in enumerate(base_models):
            try:
                # get probabilities for class 1 via cross_val_predict with method='predict_proba'
                preds = cross_val_predict(m, X, y, cv=kf, method='predict_proba', n_jobs=1)
                # preds is (n_samples, n_classes)
                if preds.ndim == 2 and preds.shape[1] == 2:
                    meta_features[:, idx] = preds[:, 1]
                else:
                    # fallback if predict_proba returns one column
                    meta_features[:, idx] = preds[:, 0]
            except Exception:
                # fallback: train on full X and use predict_proba (less ideal)
                try:
                    m.fit(X, y)
                    p = safe_proba_predict(m, X).reshape(-1)
                    meta_features[:, idx] = p
                except Exception:
                    meta_features[:, idx] = 0.5

        # train meta learner on meta_features
        meta = LogisticRegression(max_iter=500)
        meta.fit(meta_features, y)
        # build last meta-feature row using base models trained on full X
        last_row = np.zeros((1, len(base_models)))
        for idx, m in enumerate(base_models):
            try:
                m.fit(X, y)
                last_row[0, idx] = safe_proba_predict(m, np.array([make_last_feature(history)]))
            except Exception:
                last_row[0, idx] = 0.5
        # predict
        proba = safe_proba_predict(meta, last_row)
        return float(proba)
    except Exception:
        # worst-case fallback
        base_preds = []
        for fn in [ml_rf, ml_gb, ml_lr, ml_et, ml_xgb if HAS_XGB else None, ml_lgb if HAS_LGB else None]:
            if fn is None:
                continue
            try:
                base_preds.append(fn(history))
            except Exception:
                base_preds.append(0.5)
        return float(np.mean(base_preds)) if base_preds else 0.5


# =====================
# AI PREDICT (main orchestration)
# =====================
def ai_predict():
    global prediction, confidence, status, dataset_size

    history = read_history()
    dataset_size = len(history)

    if dataset_size < window:
        status = "Cần ít nhất 12 phiên"
        return

    # progress & step messages
    status = "RF AI ........ 10%"
    rf = ml_rf(history)
    gb = ml_gb(history)
    lr = ml_lr(history)
    etp = ml_et(history)
    xgbp = ml_xgb(history) if HAS_XGB else 0.5
    lgbp = ml_lgb(history) if HAS_LGB else 0.5

    status = "Markov AI .... 30%"
    mk1 = markov1(history)
    mk2 = markov2(history)

    status = "Pattern AI ... 55%"
    pt = pattern_engine(history)

    status = "Stacker AI ... 70%"
    stkr = stacker_engine(history)

    status = "Monte Carlo .. 85%"
    mc = monte_engine(history)

    mk = market_engine(history)
    st = streak_engine(history)
    en = entropy_engine(history)

    # assemble engines — include stacker as extra engine
    engines_prob = [rf, gb, lr, etp, xgbp, lgbp, mk1, mk2, pt, stkr, mc, mk, st, en]
    # ensure all are floats and in [0,1]
    probs = [min(max(0.0, float(p)), 1.0) for p in engines_prob]

    # also fill engine_predictions simple interpretation (>=0.5 => 1)
    names = ["rf", "gb", "lr", "et", "xgb", "lgb", "markov1", "markov2", "pattern", "stacker", "monte", "market", "streak", "entropy"]
    engine_predictions.clear()
    for n, p in zip(names, probs):
        engine_predictions[n] = 1 if p >= 0.5 else 0

    status = "Final AI ..... 100%"

    # final decision: equal-weight average across all available engines
    prob = float(np.mean(probs))

    if prob > 0.5:
        prediction = "TÀI"
        confidence = round(prob * 100, 2)
    else:
        prediction = "XỈU"
        confidence = round((1 - prob) * 100, 2)

    status = "BOT đã phân tích xong"


# =====================
# DATA COLLECTOR
# =====================
def collector():
    global latest_period, latest_result, latest_total, last_period

    while True:
        try:
            r = requests.get(API, timeout=10)
            data = r.json()
            period = data.get("period")
            result = data.get("result")
            total = data.get("total")
            if period is None:
                time.sleep(3)
                continue

            if period != last_period:
                last_period = period
                latest_period = period
                latest_result = result
                latest_total = total
                save_history(period, result, total)
                # whenever we append a new period, run the predictor on the updated dataset
                ai_predict()
        except Exception as e:
            # keep the loop alive; print for debugging
            print("API error", e)
        time.sleep(3)


# =====================
# WEB
# =====================
@app.route("/")
def home():
    return """
<html>
<head>
<title>AI SUPREME (with Stacker)</title>
<style>
body{ background:#020617; color:white; font-family:Arial; text-align:center; margin-top:40px; }
.result{ font-size:70px; color:#00ffc8; }
.status{ font-size:18px; margin-top:14px; color:#00ffaa; white-space:pre-line; }
.small{ font-size:14px; color:#bfeee0; }
.box{ display:inline-block; padding:16px 26px; border-radius:12px; background:rgba(255,255,255,0.02); box-shadow:0 6px 18px rgba(0,0,0,0.6); }
</style>
</head>
<body>
<h2>AI SUPREME — Stacker Edition</h2>
<div class="box" id="data">Loading...</div>
<script>
async function load(){
  let r = await fetch("/api")
  let d = await r.json()
  let progressText = d.status || ""
  document.getElementById("data").innerHTML =
    "<div class='result'>"+(d.prediction||"-")+"</div>" +
    "<p class='small'>Phiên: "+(d.period||"-")+"</p>" +
    "<p class='small'>Kết quả: "+(d.result||"-")+"</p>" +
    "<p class='small'>Tổng xúc xắc: "+(d.total||"-")+"</p>" +
    "<p class='small'>Confidence: "+(d.confidence||"-")+"%</p>" +
    "<p class='small'>Dataset: "+(d.dataset||0)+" phiên</p>" +
    "<div class='status'>RF AI ........ 10%\\nMarkov AI .... 30%\\nPattern AI ... 55%\\nStacker AI ... 70%\\nMonte Carlo .. 85%\\nFinal AI ..... 100%\\n\\n"+progressText+"</div>"
}
setInterval(load,2000)
load()
</script>
</body>
</html>
"""


@app.route("/api")
def api():
    return jsonify({
        "period": latest_period,
        "result": latest_result,
        "total": latest_total,
        "prediction": prediction,
        "confidence": confidence,
        "dataset": dataset_size,
        "status": status,
        "engine_votes": engine_predictions
    })


# start collector thread
threading.Thread(target=collector, daemon=True).start()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
