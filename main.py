# ai_supreme_selflearning.py
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

# optional boosters
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

API = "https://phanmemdudoan.fun/apisun.php"  # replace if needed
history_file = "history.txt"

window = 12
max_history = 2000

latest_period = None
latest_result = None
latest_total = None

prediction = None
confidence = None
dataset_size = 0

status = "BOT khởi động..."

last_period = None

# store last engines' probs (so when real arrives we can learn)
last_engine_probs = None
last_engine_names = None

# engine adaptive scores (weights)
model_score = {
    "rf": 1.0, "gb": 1.0, "lr": 1.0, "et": 1.0, "xgb": 1.0, "lgb": 1.0,
    "markov1": 1.0, "markov2": 1.0, "pattern": 1.0, "stacker": 1.0,
    "monte": 1.0, "market": 1.0, "streak": 1.0, "entropy": 1.0
}

engine_predictions = {}  # binary votes for current prediction

# ===== normalize API result into 0/1 (xỉu:0, tài:1)
def normalize(r):
    r = str(r).lower()
    if "tai" in r:
        return 1
    if "xiu" in r:
        return 0
    try:
        v = int(r)
        # fallback: treat >=11 as tài for 3 dice
        return 1 if v >= 11 else 0
    except:
        return 0

# ===== history I/O
def read_history():
    data = []
    if not os.path.exists(history_file):
        return data
    with open(history_file, "r", encoding="utf-8") as f:
        for line in f:
            p = line.strip().split(",")
            if len(p) < 3:
                continue
            data.append(normalize(p[1]))
    return data[-max_history:]

def save_history(period, result, total):
    lines = []
    if os.path.exists(history_file):
        with open(history_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
    lines.append(f"{period},{result},{total}\n")
    if len(lines) > max_history:
        lines = lines[-max_history:]
    with open(history_file, "w", encoding="utf-8") as f:
        f.writelines(lines)

# ===== feature engineering (expanded)
def run_lengths(arr):
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

def features(seq):
    arr = np.array(seq, dtype=float)
    ssum = float(arr.sum())
    mean = float(arr.mean())
    std = float(arr.std())
    mx = float(arr.max())
    mn = float(arr.min())
    last3 = float(arr[-3:].sum()) if len(arr) >= 3 else ssum
    last6 = float(arr[-6:].sum()) if len(arr) >= 6 else ssum
    runs = run_lengths(arr)
    longest_run = float(max(runs)) if len(runs) > 0 else 0.0
    last_run = float(runs[-1]) if len(runs) > 0 else 0.0
    transitions = float(((arr[1:] - arr[:-1]) != 0).sum()) if len(arr) > 1 else 0.0
    ac1 = 0.0
    if len(arr) > 1:
        try:
            ac1 = float(np.corrcoef(arr[:-1], arr[1:])[0,1])
            if np.isnan(ac1):
                ac1 = 0.0
        except:
            ac1 = 0.0
    return [ssum, mean, std, mx, mn, last3, last6, longest_run, last_run, transitions, ac1]

# ===== dataset builder (IMPORTANT: we analyze newest -> oldest)
def build_dataset_from_history(history):
    """
    Build X,y using history ordered newest->oldest (history_rev[0] is newest).
    This honors "analyze from newest to oldest".
    """
    if len(history) < window + 2:
        return (None, None)
    # reverse: newest first
    hr = history[::-1]
    X = []
    y = []
    n = len(hr)
    for i in range(0, n - window):
        seq = hr[i:i+window]  # seq[0] is newest in window
        feats = list(seq) + features(seq)
        X.append(feats)
        y.append(hr[i+window])
    return (np.array(X), np.array(y))

def make_last_feature_from_history(history):
    hr = history[::-1]
    seq = hr[0:window] if len(hr) >= window else (hr + [0]* (window - len(hr)))
    return list(seq) + features(seq)

# ===== helpers for safe prediction
def safe_proba_predict(model, X):
    try:
        proba = model.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] == 2:
            return float(proba[:,1][0])
        else:
            return float(proba[:,0][0])
    except Exception:
        try:
            p = model.predict(X)[0]
            return 1.0 if p == 1 else 0.0
        except:
            return 0.5

# ===== base ML engines (train on newest->oldest)
def ml_rf(history):
    X,y = build_dataset_from_history(history)
    if X is None:
        return 0.5
    m = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
    m.fit(X,y)
    last = np.array([make_last_feature_from_history(history)])
    return safe_proba_predict(m, last)

def ml_gb(history):
    X,y = build_dataset_from_history(history)
    if X is None:
        return 0.5
    m = GradientBoostingClassifier()
    m.fit(X,y)
    last = np.array([make_last_feature_from_history(history)])
    return safe_proba_predict(m, last)

def ml_lr(history):
    X,y = build_dataset_from_history(history)
    if X is None:
        return 0.5
    m = LogisticRegression(max_iter=500)
    m.fit(X,y)
    last = np.array([make_last_feature_from_history(history)])
    return safe_proba_predict(m, last)

def ml_et(history):
    X,y = build_dataset_from_history(history)
    if X is None:
        return 0.5
    m = ExtraTreesClassifier(n_estimators=150, n_jobs=-1, random_state=1)
    m.fit(X,y)
    last = np.array([make_last_feature_from_history(history)])
    return safe_proba_predict(m, last)

def ml_xgb(history):
    if not HAS_XGB:
        return 0.5
    X,y = build_dataset_from_history(history)
    if X is None:
        return 0.5
    m = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", n_jobs=4)
    m.fit(X,y)
    last = np.array([make_last_feature_from_history(history)])
    return safe_proba_predict(m, last)

def ml_lgb(history):
    if not HAS_LGB:
        return 0.5
    X,y = build_dataset_from_history(history)
    if X is None:
        return 0.5
    m = lgb.LGBMClassifier(n_jobs=4)
    m.fit(X,y)
    last = np.array([make_last_feature_from_history(history)])
    return safe_proba_predict(m, last)

# ===== markov, pattern, monte, market, streak, entropy (use history_rev as newest-first)
def markov1(history):
    hr = history[::-1]
    # if only one sample, fallback
    if len(hr) < 2:
        return 0.5
    matrix = np.zeros((2,2))
    for i in range(len(hr)-1):
        matrix[hr[i]][hr[i+1]] += 1
    row = matrix[hr[0]]  # current newest -> next (older)
    if row.sum() == 0:
        return 0.5
    return float(row[1] / row.sum())

def markov2(history):
    hr = history[::-1]
    if len(hr) < 3:
        return 0.5
    pair = hr[0:2]
    count = 0
    win = 0
    n = len(hr)
    for i in range(n-2):
        if hr[i:i+2] == pair:
            count += 1
            if hr[i+2] == 1:
                win += 1
    if count == 0:
        return 0.5
    return float(win/count)

def pattern_engine(history):
    hr = history[::-1]
    wins = []
    weights = []
    n = len(hr)
    for size in range(3, 11):
        if n < size + 1:
            continue
        seq = hr[0:size]
        count = 0
        win = 0
        for i in range(n - size):
            if hr[i:i+size] == seq:
                count += 1
                if i + size < n:
                    win += hr[i+size]
        if count > 0:
            wins.append(win / count)
            weights.append(count)
    if len(wins) == 0:
        return 0.5
    return float(np.average(wins, weights=weights))

def monte_engine(history):
    p = float(sum(history)/len(history))
    trials = 20000
    wins = 0
    for _ in range(trials):
        if random.random() < p:
            wins += 1
    return float(wins/trials)

def market_engine(history):
    last50 = history[-50:] if len(history) >= 1 else history
    if len(last50) == 0:
        return 0.5
    return float(sum(last50)/len(last50))

def streak_engine(history):
    hr = history[::-1]
    streak = 1
    for i in range(len(hr)-1):
        if i+1 < len(hr) and hr[i] == hr[i+1]:
            streak += 1
        else:
            break
    if streak >= 5:
        # contrarian bias
        return 0.3 if hr[0] == 1 else 0.7
    return 0.5

def entropy_engine(history):
    p = float(sum(history)/len(history))
    entropy = -(p*np.log2(p+1e-9) + (1-p)*np.log2(1-p+1e-9))
    return float(p*(1-entropy))

# ===== stacker (meta-learner) — powerful engine
def stacker_engine(history):
    X, y = build_dataset_from_history(history)
    if X is None or X.shape[0] < 30:
        # fallback: average base models' last predictions
        base_preds = []
        for fn in [ml_rf, ml_gb, ml_lr, ml_et, ml_xgb if HAS_XGB else None, ml_lgb if HAS_LGB else None]:
            if fn is None:
                continue
            try:
                base_preds.append(fn(history))
            except:
                base_preds.append(0.5)
        return float(np.mean(base_preds)) if base_preds else 0.5
    try:
        # base models
        base_models = []
        base_names = []
        rf = RandomForestClassifier(n_estimators=120, n_jobs=-1, random_state=1)
        gb = GradientBoostingClassifier()
        lr = LogisticRegression(max_iter=500)
        et = ExtraTreesClassifier(n_estimators=120, n_jobs=-1, random_state=2)
        base_models.extend([rf, gb, lr, et])
        base_names.extend(["rf","gb","lr","et"])
        if HAS_XGB:
            xb = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", n_jobs=4)
            base_models.append(xb); base_names.append("xgb")
        if HAS_LGB:
            lb = lgb.LGBMClassifier(n_jobs=4)
            base_models.append(lb); base_names.append("lgb")
        # out-of-fold predictions
        n_samples = X.shape[0]
        meta_X = np.zeros((n_samples, len(base_models)))
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        for idx, m in enumerate(base_models):
            try:
                preds = cross_val_predict(m, X, y, cv=kf, method='predict_proba', n_jobs=1)
                if preds.ndim == 2 and preds.shape[1] == 2:
                    meta_X[:, idx] = preds[:,1]
                else:
                    meta_X[:, idx] = preds[:,0]
            except Exception:
                try:
                    m.fit(X,y)
                    p = m.predict_proba(X)
                    if p.shape[1] == 2:
                        meta_X[:, idx] = p[:,1]
                    else:
                        meta_X[:, idx] = p[:,0]
                except:
                    meta_X[:, idx] = 0.5
        # train meta
        meta = LogisticRegression(max_iter=500)
        meta.fit(meta_X, y)
        # build last meta row
        last_row = np.zeros((1, len(base_models)))
        for idx, m in enumerate(base_models):
            try:
                m.fit(X,y)
                last_row[0, idx] = safe_proba_predict(m, np.array([make_last_feature_from_history(history)]))
            except:
                last_row[0, idx] = 0.5
        return float(safe_proba_predict(meta, last_row))
    except Exception:
        # fallback to base average
        base_preds = []
        for fn in [ml_rf, ml_gb, ml_lr, ml_et, ml_xgb if HAS_XGB else None, ml_lgb if HAS_LGB else None]:
            if fn is None:
                continue
            try:
                base_preds.append(fn(history))
            except:
                base_preds.append(0.5)
        return float(np.mean(base_preds)) if base_preds else 0.5

# ===== self-learning adaptive update
def self_learning_from_last(real):
    """
    Update model_score using last_engine_probs and actual real outcome (0/1).
    - Increase weight when engine predicted correctly, decrease otherwise.
    - Apply small decay to all weights to forget old data slowly.
    - Keep weights bounded.
    """
    global last_engine_probs, last_engine_names, model_score

    if last_engine_probs is None or last_engine_names is None:
        return

    lr_pos = 0.06   # reward increment when correct
    lr_neg = 0.03   # penalty when wrong
    decay = 0.997   # multiplicative decay per update

    # apply decay to all
    for k in model_score:
        model_score[k] *= decay

    # update based on last prediction
    for name, prob in zip(last_engine_names, last_engine_probs):
        pred = 1 if prob >= 0.5 else 0
        if pred == real:
            model_score[name] += lr_pos
        else:
            model_score[name] -= lr_neg
        # clamp
        if model_score[name] < 0.2:
            model_score[name] = 0.2
        if model_score[name] > 6.0:
            model_score[name] = 6.0

# ===== main AI orchestration
def ai_predict():
    global prediction, confidence, status, dataset_size, last_engine_probs, last_engine_names, engine_predictions

    history = read_history()
    dataset_size = len(history)

    if dataset_size < window:
        status = "Cần ít nhất 12 phiên"
        return

    # We'll operate on history but build dataset newest->oldest inside engines
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

    # assemble engines to list with names (only include xgb/lgb if available)
    engine_names = ["rf","gb","lr","et","xgb","lgb","markov1","markov2","pattern","stacker","monte","market","streak","entropy"]
    engine_probs = [rf, gb, lr, etp, xgbp, lgbp, mk1, mk2, pt, stkr, mc, mk, st, en]

    # store last engine probs/names for self-learning when result comes
    last_engine_probs = engine_probs.copy()
    last_engine_names = engine_names.copy()

    # compute weighted average using adaptive model_score
    weights = [model_score.get(n, 1.0) for n in engine_names]
    # ensure lengths match (if xgb/lgb missing, names still present but prob=0.5)
    # normalize weights to avoid numeric issues (not strictly necessary)
    w_arr = np.array(weights, dtype=float)
    # weighted average:
    p_arr = np.array(engine_probs, dtype=float)
    prob = float(np.average(p_arr, weights=w_arr))

    # also compute simple average and majority (optional)
    simple_avg = float(np.mean(p_arr))
    majority_votes = sum([1 if p >= 0.5 else 0 for p in p_arr])

    status = "Final AI ..... 100%"

    if prob > 0.5:
        prediction = "TÀI"
        confidence = round(prob * 100, 2)
    else:
        prediction = "XỈU"
        confidence = round((1 - prob) * 100, 2)

    # store engine_predictions binary
    engine_predictions = {n: (1 if p >= 0.5 else 0) for n, p in zip(engine_names, engine_probs)}

    status = "BOT đã phân tích xong"

# ===== collector (fetch new data and trigger learning)
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

            # when new period appears
            if period != last_period:
                # If we had previous engine predictions (for previous upcoming), learn from actual result
                try:
                    real = normalize(result)
                    # Learn using last_engine_probs from previous ai_predict run (if exists)
                    self_learning_from_last(real)
                except Exception:
                    pass

                # update last period and save
                last_period = period
                latest_period = period
                latest_result = result
                latest_total = total
                save_history(period, result, total)

                # after saving, re-run prediction (will set last_engine_probs for next update)
                ai_predict()

        except Exception as e:
            print("API error", e)

        time.sleep(3)

# ===== Web UI
@app.route("/")
def home():
    return """
<html>
<head>
<title>AI SUPREME — Self-Learning Final</title>
<style>
body{ background:#020617; color:white; font-family:Arial; text-align:center; margin-top:40px; }
.result{ font-size:70px; color:#00ffc8; }
.status{ font-size:16px; margin-top:12px; color:#bfffe0; white-space:pre-line; text-align:left; display:inline-block; padding:12px 18px; border-radius:10px; background:rgba(255,255,255,0.02); }
.small{ font-size:14px; color:#cfeee7; }
.box{ display:inline-block; padding:16px 26px; border-radius:12px; background:rgba(255,255,255,0.02); box-shadow:0 10px 30px rgba(0,0,0,0.6); }
</style>
</head>
<body>
<h2>AI SUPREME — Self-Learning Final</h2>
<div class="box" id="data">Loading...</div>
<script>
async function load(){
  let r = await fetch("/api")
  let d = await r.json()
  let progressText = d.status || ""
  let progLines = "RF AI ........ 10%\\nMarkov AI .... 30%\\nPattern AI ... 55%\\nStacker AI ... 70%\\nMonte Carlo .. 85%\\nFinal AI ..... 100%\\n\\n"
  document.getElementById("data").innerHTML =
    "<div class='result'>"+(d.prediction||"-")+"</div>" +
    "<p class='small'>Phiên: "+(d.period||"-")+"</p>" +
    "<p class='small'>Kết quả: "+(d.result||"-")+"</p>" +
    "<p class='small'>Tổng xúc xắc: "+(d.total||"-")+"</p>" +
    "<p class='small'>Confidence: "+(d.confidence||"-")+"%</p>" +
    "<p class='small'>Dataset: "+(d.dataset||0)+" phiên</p>" +
    "<div class='status'>"+progLines+progressText+"</div>"
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
        "engine_votes": engine_predictions,
        "model_score": model_score
    })

# start collector thread
threading.Thread(target=collector, daemon=True).start()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
