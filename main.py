from flask import Flask,jsonify
import requests
import threading
import time
import os
import numpy as np
import random

from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

app=Flask(__name__)

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

status="BOT đang khởi động"

last_period=None

# =====================
# NORMALIZE RESULT
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

    if not os.path.exists(history_file):
        return data

    with open(history_file) as f:

        for line in f:

            p=line.strip().split(",")

            if len(p)<3:
                continue

            data.append(normalize(p[1]))

    return data[-max_history:]


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
# FEATURE ENGINEERING
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
# AI 1 RANDOM FOREST
# =====================

def ml_rf(history):

    X=[]
    y=[]

    for i in range(len(history)-window):

        seq=history[i:i+window]

        X.append(seq+features(seq))
        y.append(history[i+window])

    if len(X)<10:
        return 0.5

    model=RandomForestClassifier(300)

    model.fit(X,y)

    last=history[-window:]

    last=np.array(last+features(last)).reshape(1,-1)

    return model.predict_proba(last)[0][1]

# =====================
# AI 2 GRADIENT BOOST
# =====================

def ml_gb(history):

    X=[]
    y=[]

    for i in range(len(history)-window):

        seq=history[i:i+window]

        X.append(seq+features(seq))
        y.append(history[i+window])

    if len(X)<10:
        return 0.5

    model=GradientBoostingClassifier()

    model.fit(X,y)

    last=history[-window:]

    last=np.array(last+features(last)).reshape(1,-1)

    return model.predict_proba(last)[0][1]

# =====================
# AI 3 LOGISTIC
# =====================

def ml_lr(history):

    X=[]
    y=[]

    for i in range(len(history)-window):

        seq=history[i:i+window]

        X.append(seq+features(seq))
        y.append(history[i+window])

    if len(X)<10:
        return 0.5

    model=LogisticRegression(max_iter=500)

    model.fit(X,y)

    last=history[-window:]

    last=np.array(last+features(last)).reshape(1,-1)

    return model.predict_proba(last)[0][1]

# =====================
# AI 4 MARKOV 1
# =====================

def markov1(history):

    matrix=np.zeros((2,2))

    for i in range(len(history)-1):

        matrix[history[i]][history[i+1]]+=1

    row=matrix[history[-1]]

    if row.sum()==0:
        return 0.5

    return row[1]/row.sum()

# =====================
# AI 5 MARKOV 2
# =====================

def markov2(history):

    if len(history)<3:
        return 0.5

    pair=history[-2:]

    count=0
    win=0

    for i in range(len(history)-2):

        if history[i:i+2]==pair:

            count+=1

            if history[i+2]==1:
                win+=1

    if count==0:
        return 0.5

    return win/count

# =====================
# AI 6 PATTERN SCAN
# =====================

def pattern_engine(history):

    wins=[]
    weights=[]

    for size in range(3,11):

        seq=history[-size:]

        count=0
        win=0

        for i in range(len(history)-size):

            if history[i:i+size]==seq:

                count+=1

                if i+size<len(history):
                    win+=history[i+size]

        if count>0:

            wins.append(win/count)
            weights.append(count)

    if len(wins)==0:
        return 0.5

    return np.average(wins,weights=weights)

# =====================
# AI 7 MONTE CARLO
# =====================

def monte_engine(history):

    p=sum(history)/len(history)

    trials=20000

    wins=0

    for _ in range(trials):

        if random.random()<p:
            wins+=1

    return wins/trials

# =====================
# AI 8 MARKET TREND
# =====================

def market_engine(history):

    last50=history[-50:]

    return sum(last50)/len(last50)

# =====================
# AI 9 STREAK
# =====================

def streak_engine(history):

    streak=1

    for i in range(len(history)-1,0,-1):

        if history[i]==history[i-1]:
            streak+=1
        else:
            break

    if streak>=5:

        if history[-1]==1:
            return 0.3
        else:
            return 0.7

    return 0.5

# =====================
# AI 10 ENTROPY
# =====================

def entropy_engine(history):

    p=sum(history)/len(history)

    entropy=-(p*np.log2(p+1e-9)+(1-p)*np.log2(1-p+1e-9))

    return p*(1-entropy)

# =====================
# AI PREDICT
# =====================

def ai_predict():

    global prediction,confidence,status,dataset_size

    history=read_history()

    dataset_size=len(history)

    if dataset_size<window:
        status="Cần ít nhất 12 phiên"
        return

    status="RF AI ........ 10%"

    rf=ml_rf(history)
    gb=ml_gb(history)
    lr=ml_lr(history)

    status="Markov AI .... 30%"

    mk1=markov1(history)
    mk2=markov2(history)

    status="Pattern AI ... 55%"

    pt=pattern_engine(history)

    status="Monte Carlo .. 80%"

    mc=monte_engine(history)

    mk=market_engine(history)
    st=streak_engine(history)
    en=entropy_engine(history)

    probs=[rf,gb,lr,mk1,mk2,pt,mc,mk,st,en]

    status="Final AI ..... 100%"

    prob=np.mean(probs)

    if prob>0.5:

        prediction="TÀI"
        confidence=round(prob*100,2)

    else:

        prediction="XỈU"
        confidence=round((1-prob)*100,2)

    status="BOT đã phân tích xong"

# =====================
# DATA COLLECTOR
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

<title>AI SUPREME</title>

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

.status{
font-size:20px;
margin-top:20px;
color:#00ffaa;
}

</style>

</head>

<body>

<h2>AI SUPREME</h2>

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

"<p>Dataset: "+d.dataset+" phiên</p>"+

"<div class='status'>"+d.status+"</div>"

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
