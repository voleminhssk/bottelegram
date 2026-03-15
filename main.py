from flask import Flask, jsonify
import requests
import threading
import time
import os
import unicodedata
import numpy as np
import random
import math

from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,GradientBoostingClassifier,HistGradientBoostingClassifier,AdaBoostClassifier,BaggingClassifier
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

app = Flask(__name__)

API="https://phanmemdudoan.fun/apisun.php"

history_file="history.txt"

window=12
max_history=1200

last_period=None

latest_period=None
latest_result=None
latest_total=None

prediction=None
confidence=None
status="AI GOD MODE loading..."

history_count=0
dataset_size=0


if not os.path.exists(history_file):
    open(history_file,"w").close()


def normalize(text):

    text=str(text)
    text=unicodedata.normalize("NFC",text).lower()

    if "tai" in text or "tài" in text:
        return "Tài"

    if "xiu" in text or "xỉu" in text:
        return "Xỉu"

    return text


def read_history():

    global history_count

    data=[]

    with open(history_file,encoding="utf-8") as f:
        lines=f.readlines()

    history_count=len(lines)

    for line in lines:

        p=line.strip().split(",")

        if len(p)<3:
            continue

        r=normalize(p[1])

        if r=="Tài":
            data.append(1)

        elif r=="Xỉu":
            data.append(0)

    return data


def save_history(period,result,total):

    lines=[]

    if os.path.exists(history_file):

        with open(history_file,encoding="utf-8") as f:
            lines=f.readlines()

    lines.append(f"{period},{result},{total}\n")

    if len(lines)>max_history:
        lines=lines[-max_history:]

    with open(history_file,"w",encoding="utf-8") as f:
        f.writelines(lines)


def sequence_features(seq):

    arr=np.array(seq)

    return [

        sum(arr),
        np.mean(arr),
        np.std(arr),
        np.max(arr),
        np.min(arr)

    ]


def markov_predict(history):

    trans=[[1,1],[1,1]]

    for i in range(len(history)-1):

        a=history[i]
        b=history[i+1]

        trans[a][b]+=1

    last=history[-1]

    prob=trans[last][1]/(trans[last][0]+trans[last][1])

    return prob


def entropy_predict(history):

    p=sum(history)/len(history)

    if p==0 or p==1:
        return 0.5

    entropy=-(p*math.log2(p)+(1-p)*math.log2(1-p))

    return p*(1-entropy)


def montecarlo(history):

    trials=1000

    count=0

    p=sum(history)/len(history)

    for _ in range(trials):

        if random.random()<p:
            count+=1

    return count/trials


def ai_predict():

    global prediction,confidence,status,dataset_size

    history=read_history()

    dataset_size=len(history)

    if dataset_size<window+10:

        status=f"Cần {window+10} phiên ({dataset_size})"
        return

    X=[]
    y=[]

    for i in range(dataset_size-window):

        seq=history[i:i+window]

        feats=sequence_features(seq)

        X.append(seq+feats)

        y.append(history[i+window])

    X=np.array(X)
    y=np.array(y)

    last_seq=history[-window:]

    last_feats=sequence_features(last_seq)

    last=np.array(last_seq+last_feats).reshape(1,-1)

    models=[

    RandomForestClassifier(200),
    ExtraTreesClassifier(200),
    GradientBoostingClassifier(),
    HistGradientBoostingClassifier(),
    AdaBoostClassifier(),
    BaggingClassifier(),
    LogisticRegression(max_iter=1000),
    SGDClassifier(loss="log_loss"),
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    GaussianNB(),
    MLPClassifier(max_iter=500),
    xgb.XGBClassifier(),
    lgb.LGBMClassifier(),
    CatBoostClassifier(verbose=0)

    ]

    probs=[]

    for m in models:

        try:

            m.fit(X,y)

            p=m.predict_proba(last)[0][1]

            probs.append(p)

        except:
            pass

    probs.append(markov_predict(history))
    probs.append(entropy_predict(history))
    probs.append(montecarlo(history))

    prob=np.mean(probs)

    if prob>0.5:

        prediction="Tài"
        confidence=round(prob*100,2)

    else:

        prediction="Xỉu"
        confidence=round((1-prob)*100,2)

    status=f"AI GOD MODE | {len(probs)} models | Dataset {dataset_size}"


def collector():

    global last_period,latest_period,latest_result,latest_total

    while True:

        try:

            r=requests.get(API,timeout=10)

            data=r.json()

            period=data["period"]
            result=normalize(data["result"])
            total=data["total"]

            if period!=last_period:

                last_period=period

                latest_period=period
                latest_result=result
                latest_total=total

                save_history(period,result,total)

                ai_predict()

                print("Round:",period,result,total)

        except Exception as e:

            print("API error:",e)

        time.sleep(15)


@app.route("/")
def home():

    return """

<html>

<head>

<title>AI Tài Xỉu GOD MODE</title>

<style>

body{
background:#0f172a;
color:white;
font-family:Arial;
text-align:center;
margin-top:80px;
}

.box{
background:#1e293b;
padding:35px;
border-radius:15px;
width:360px;
margin:auto;
box-shadow:0 0 20px rgba(0,255,200,0.3);
}

.result{
font-size:45px;
margin:20px;
color:#00ffc8;
}

.info{
margin:6px;
font-size:15px;
}

</style>

</head>

<body>

<div class="box">

<h2>AI TÀI XỈU GOD MODE</h2>

<div id="data">Loading...</div>

</div>

<script>

async function load(){

let r=await fetch("/api")
let d=await r.json()

document.getElementById("data").innerHTML=

"<div class='result'>"+(d.prediction||"-")+"</div>"+

"<div class='info'>Phiên trước: "+(d.period||"-")+"</div>"+

"<div class='info'>Kết quả: "+(d.result||"-")+"</div>"+

"<div class='info'>Tổng xúc xắc: "+(d.total||"-")+" 🎲</div>"+

"<div class='info'>Confidence: "+(d.confidence||"-")+"%</div>"+

"<div class='info'>History: "+(d.history)+" / 1200</div>"+

"<div class='info'>Dataset AI: "+(d.dataset)+"</div>"+

"<div class='info'>"+d.status+"</div>"

}

setInterval(load,5000)

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
        "status":status,
        "history":history_count,
        "dataset":dataset_size

    })


threading.Thread(target=collector,daemon=True).start()

if __name__=="__main__":

    port=int(os.environ.get("PORT",10000))

    app.run(host="0.0.0.0",port=port)

