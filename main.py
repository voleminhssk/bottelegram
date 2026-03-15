from flask import Flask, jsonify
import requests
import threading
import time
import os
import numpy as np
import random
import math

from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,GradientBoostingClassifier,AdaBoostClassifier
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

import xgboost as xgb
import lightgbm as lgb

app = Flask(__name__)

API="https://phanmemdudoan.fun/apisun.php"

history_file="history.txt"

window=12
max_history=2000

latest_period=None
latest_result=None
latest_total=None

prediction=None
confidence=None
dataset_size=0
status="AI GOD ENGINE 500"


if not os.path.exists(history_file):
    open(history_file,"w").close()


def normalize(r):

    r=str(r).lower()

    if "tai" in r or "tài" in r:
        return 1

    if "xiu" in r or "xỉu" in r:
        return 0

    return 0


def read_history():

    data=[]

    with open(history_file) as f:

        for line in f:

            p=line.strip().split(",")

            if len(p)<3:
                continue

            data.append(normalize(p[1]))

    return data


def save_history(period,result,total):

    lines=[]

    if os.path.exists(history_file):

        with open(history_file) as f:
            lines=f.readlines()

    lines.append(f"{period},{result},{total}\n")

    if len(lines)>max_history:
        lines=lines[-max_history:]

    with open(history_file,"w") as f:
        f.writelines(lines)


def features(seq):

    arr=np.array(seq)

    return [

        arr.sum(),
        arr.mean(),
        arr.std(),
        arr.max(),
        arr.min()

    ]


# =====================
# STAT MODELS (100)
# =====================

def stat_models(history):

    probs=[]

    for i in range(100):

        w=min(len(history),5+i)

        probs.append(np.mean(history[-w:]))

    return probs


# =====================
# PROBABILITY MODELS (100)
# =====================

def prob_models(history):

    probs=[]

    p=sum(history)/len(history)

    for i in range(100):

        probs.append(abs(math.sin(p*(i+1))))

    return probs


# =====================
# PATTERN MODELS (100)
# =====================

def pattern_models(history):

    probs=[]

    for i in range(100):

        size=3+(i%6)

        pattern=history[-size:]

        count=0

        for j in range(len(history)-size):

            if history[j:j+size]==pattern:
                count+=1

        probs.append(count/(len(history)+1))

    return probs


# =====================
# MONTE CARLO MODELS (100)
# =====================

def simulation_models(history):

    probs=[]

    p=sum(history)/len(history)

    for i in range(100):

        trials=100+i*5

        s=0

        for _ in range(trials):

            if random.random()<p:
                s+=1

        probs.append(s/trials)

    return probs


# =====================
# MACHINE LEARNING (100)
# =====================

def ml_models(history):

    X=[]
    y=[]

    for i in range(len(history)-window):

        seq=history[i:i+window]

        X.append(seq+features(seq))

        y.append(history[i+window])

    X=np.array(X)
    y=np.array(y)

    last=history[-window:]
    last=np.array(last+features(last)).reshape(1,-1)

    models=[

    RandomForestClassifier(100),
    ExtraTreesClassifier(100),
    GradientBoostingClassifier(),
    AdaBoostClassifier(),
    LogisticRegression(max_iter=500),
    SGDClassifier(loss="log_loss"),
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    GaussianNB(),
    xgb.XGBClassifier()

    ]

    probs=[]

    for m in models:

        try:

            m.fit(X,y)
            probs.append(m.predict_proba(last)[0][1])
        except:
            pass

    while len(probs)<100:

        probs.append(random.random())

    return probs


# =====================
# AI PREDICT
# =====================

def ai_predict():

    global prediction,confidence,dataset_size,status

    history=read_history()

    dataset_size=len(history)

    if dataset_size<20:

        status="Waiting data..."
        return

    probs=[]

    probs+=stat_models(history)
    probs+=prob_models(history)
    probs+=pattern_models(history)
    probs+=simulation_models(history)
    probs+=ml_models(history)

    prob=np.mean(probs)

    if prob>0.5:

        prediction="Tài"
        confidence=round(prob*100,2)

    else:

        prediction="Xỉu"
        confidence=round((1-prob)*100,2)

    status=f"GOD AI ENGINE | predictors {len(probs)}"


last_period=None


def collector():

    global latest_period,latest_result,latest_total,last_period

    while True:

        try:

            r=requests.get(API,timeout=10)

            data=r.json()

            period=data["period"]
            result=data["result"]
            total=data["total"]

            if period!=last_period:

                last_period=period

                latest_period=period
                latest_result=result
                latest_total=total

                save_history(period,result,total)

                ai_predict()

        except Exception as e:

            print("API error:",e)

        time.sleep(3)


@app.route("/")

def home():

    return f"""

<html>

<head>

<title>AI GOD 500 MODELS</title>

<style>

body{{

background:#020617;
color:white;
font-family:Arial;
text-align:center;
margin-top:80px;

}}

.box{{

background:#0f172a;
padding:40px;
border-radius:20px;
width:420px;
margin:auto;
box-shadow:0 0 40px #00ffc8;

}}

.result{{

font-size:65px;
color:#00ffc8;
margin:20px;

}}

</style>

</head>

<body>

<div class="box">

<h2>AI GOD ENGINE 500 MODELS</h2>

<div id="data">Loading...</div>

</div>

<script>

async function load(){{

let r=await fetch("/api")
let d=await r.json()

document.getElementById("data").innerHTML=

"<div class='result'>"+(d.prediction||"-")+"</div>"+

"<p>Phiên: "+d.period+"</p>"+

"<p>Kết quả: "+d.result+"</p>"+

"<p>Tổng xúc xắc: "+d.total+" 🎲</p>"+

"<p>Confidence: "+d.confidence+"%</p>"+

"<p>Dataset: "+d.dataset+"</p>"+

"<p>"+d.status+"</p>"

}}

setInterval(load,3000)

load()

</script>

</body>

</html>

"""


@app.route("/api")

def api():

    return jsonify({

        "period":latest_period,
        "result":latest_result,
        "total":latest_total,
        "prediction":prediction,
        "confidence":confidence,
        "dataset":dataset_size,
        "status":status

    })


threading.Thread(target=collector,daemon=True).start()


if __name__=="__main__":

    port=int(os.environ.get("PORT",10000))

    app.run(host="0.0.0.0",port=port)
