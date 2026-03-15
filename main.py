from flask import Flask,jsonify
import requests
import threading
import time
import os
import numpy as np
import random

from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

import xgboost as xgb
import lightgbm as lgb

app=Flask(__name__)

API="https://phanmemdudoan.fun/apisun.php"

history_file="history.txt"

window=12
max_history=10000

latest_period=None
latest_result=None
latest_total=None

prediction=None
confidence=None
dataset_size=0

status="AI SUPREME READY"
progress=0

last_period=None

model_score={

"ml":1,
"markov":1,
"pattern":1,
"monte":1,
"market":1

}

if not os.path.exists(history_file):
    open(history_file,"w").close()

# =====================
# NORMALIZE
# =====================

def normalize(r):

    r=str(r).lower()

    if "tai" in r:
        return 1

    if "xiu" in r:
        return 0

    return 0

# =====================
# HISTORY
# =====================

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

# =====================
# FEATURES
# =====================

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
# MACHINE LEARNING
# =====================

def ml_engine(history):

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

    RandomForestClassifier(200),
    ExtraTreesClassifier(200),
    LogisticRegression(max_iter=500),
    DecisionTreeClassifier(),
    GaussianNB(),
    xgb.XGBClassifier(),
    lgb.LGBMClassifier()

    ]

    probs=[]

    for m in models:

        try:

            m.fit(X,y)
            probs.append(m.predict_proba(last)[0][1])
        except:
            pass

    return np.mean(probs)

# =====================
# MARKOV LEVEL 2
# =====================

def markov_engine(history):

    matrix=np.zeros((2,2))

    for i in range(len(history)-1):

        matrix[history[i]][history[i+1]]+=1

    row=matrix[history[-1]]

    if row.sum()==0:
        return 0.5

    return row[1]/row.sum()

# =====================
# PATTERN MEMORY
# =====================

def pattern_engine(history):

    seq=history[-6:]

    count=0
    win=0

    for i in range(len(history)-6):

        if history[i:i+6]==seq:

            count+=1

            if i+6<len(history):

                win+=history[i+6]

    if count==0:
        return 0.5

    return win/count

# =====================
# MONTE CARLO
# =====================

def monte_engine(history):

    p=sum(history)/len(history)

    trials=2000

    win=0

    for _ in range(trials):

        if random.random()<p:
            win+=1

    return win/trials

# =====================
# MARKET PROBABILITY
# =====================

def market_engine(history):

    last50=history[-50:]

    return sum(last50)/len(last50)

# =====================
# AI ENGINE
# =====================

def ai_predict():

    global prediction,confidence,dataset_size,status,progress

    status="AI đang phân tích..."
    progress=10

    history=read_history()

    dataset_size=len(history)

    if dataset_size<window:

        status="Cần ít nhất 12 phiên"
        return

    p1=ml_engine(history)
    progress=30

    p2=markov_engine(history)
    progress=50

    p3=pattern_engine(history)
    progress=70

    p4=monte_engine(history)
    progress=85

    p5=market_engine(history)

    weights=list(model_score.values())

    probs=[p1,p2,p3,p4,p5]

    prob=np.average(probs,weights=weights)

    progress=100

    if prob>0.5:

        prediction="TÀI"
        confidence=round(prob*100,2)

    else:

        prediction="XỈU"
        confidence=round((1-prob)*100,2)

    status="AI phân tích xong"

# =====================
# COLLECTOR
# =====================

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

            print("API error",e)

        time.sleep(3)

# =====================
# WEB
# =====================

@app.route("/")

def home():

    return """

<html>

<head>

<title>AI GOD SUPREME</title>

<style>

body{
background:#020617;
color:white;
font-family:Arial;
text-align:center;
margin-top:80px;
}

.result{
font-size:70px;
color:#00ffc8;
}

.bar{
height:10px;
background:#00ffc8;
}

.progress{
height:10px;
background:#1e293b;
width:300px;
margin:auto;
}

</style>

</head>

<body>

<h2>AI GOD SUPREME ENGINE</h2>

<div id="data">Loading...</div>

<script>

async function load(){

let r=await fetch("/api")
let d=await r.json()

document.getElementById("data").innerHTML=

"<div class='result'>"+(d.prediction||"-")+"</div>"+

"<p>Phiên: "+d.period+"</p>"+

"<p>Kết quả: "+d.result+"</p>"+

"<p>Tổng xúc xắc: "+d.total+"</p>"+

"<p>Confidence: "+d.confidence+"%</p>"+

"<p>"+d.status+"</p>"+

"<div class='progress'><div class='bar' style='width:"+d.progress+"%'></div></div>"

}

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
        "status":status,
        "progress":progress

    })

threading.Thread(target=collector,daemon=True).start()

if __name__=="__main__":

    port=int(os.environ.get("PORT",10000))

    app.run(host="0.0.0.0",port=port)
