from flask import Flask, jsonify, render_template_string
import requests
import threading
import time
import os
import numpy as np
import unicodedata
import random

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


app = Flask(__name__)

API = "https://phanmemdudoan.fun/apisun.php"

history_file = "history.txt"

window = 12
max_history = 1000

last_period=None

latest_period=None
latest_result=None
latest_total=None

prediction=None
confidence=None
status="Waiting data"

wins=0
loss=0
total_predict=0

if not os.path.exists(history_file):
    open(history_file,"w").close()


# normalize text
def normalize(text):

    text=str(text)
    text=unicodedata.normalize("NFC",text).lower()

    if "tai" in text or "tài" in text:
        return "Tài"

    if "xiu" in text or "xỉu" in text:
        return "Xỉu"

    return text


# read history
def read_history():

    data=[]

    with open(history_file,encoding="utf-8") as f:
        lines=f.readlines()

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


# save history
def save_history(period,result,total):

    lines=[]

    if os.path.exists(history_file):
        with open(history_file,encoding="utf-8") as f:
            lines=f.readlines()

    if len(lines)>=max_history:
        lines=[]

    lines.append(f"{period},{result},{total}\n")

    with open(history_file,"w",encoding="utf-8") as f:
        f.writelines(lines)


# pattern engines
def entropy(seq):

    p=sum(seq)/len(seq)

    if p==0 or p==1:
        return 0

    return -(p*np.log2(p)+(1-p)*np.log2(1-p))


def streak(seq):

    c=1
    for i in range(len(seq)-1,0,-1):

        if seq[i]==seq[i-1]:
            c+=1
        else:
            break

    return c


def momentum(seq):
    return sum(seq[-5:])/5


# transformer attention
def attention(seq):

    seq=np.array(seq)

    pos=np.arange(1,len(seq)+1)

    weights=np.exp(seq*pos)

    weights=weights/np.sum(weights)

    return float(np.sum(seq*weights))


# feature engine
def features(seq):

    return [

        sum(seq),
        np.mean(seq),
        np.std(seq),
        entropy(seq),
        streak(seq),
        momentum(seq),
        attention(seq)

    ]


# dataset
def dataset(history):

    X=[]
    y=[]

    for i in range(len(history)-window):

        seq=history[i:i+window]

        X.append(seq+features(seq))

        y.append(history[i+window])

    return np.array(X),np.array(y)


# probability engines
def bayesian(history):

    t=history.count(1)
    x=history.count(0)

    if t+x==0:
        return 0.5

    return t/(t+x)


def markov(history):

    if len(history)<2:
        return 0.5

    trans=[[0,0],[0,0]]

    for i in range(len(history)-1):

        a=history[i]
        b=history[i+1]

        trans[a][b]+=1

    last=history[-1]

    total=trans[last][0]+trans[last][1]

    if total==0:
        return 0.5

    return trans[last][1]/total


# monte carlo
def monte_carlo(history,sim=200):

    score=0

    p=sum(history[-50:])/len(history[-50:])

    for _ in range(sim):

        if random.random()<p:
            score+=1

    return score/sim


# AI predict
def ai_predict():

    global prediction,confidence,status

    history=read_history()

    if len(history)<window+5:

        status=f"Cần ít nhất {window} phiên"
        return

    X,y=dataset(history)

    seq=history[-window:]

    last=np.array(seq+features(seq)).reshape(1,-1)

    models=[

        RandomForestClassifier(n_estimators=120),
        GradientBoostingClassifier(),
        LogisticRegression(max_iter=300),
        MLPClassifier(hidden_layer_sizes=(20,),max_iter=200),
        XGBClassifier(n_estimators=120),
        LGBMClassifier(n_estimators=120)

    ]

    probs=[]

    for m in models:

        try:

            m.fit(X,y)

            p=m.predict_proba(last)[0][1]

            probs.append(p)

        except:
            pass

    ml=sum(probs)/len(probs)

    engines=[

        bayesian(history),
        markov(history),
        monte_carlo(history)

    ]

    final=(ml+sum(engines))/(1+len(engines))

    if final>0.5:

        prediction="Tài"
        confidence=round(final*100,2)

    else:

        prediction="Xỉu"
        confidence=round((1-final)*100,2)

    status="AI Ready"


# accuracy
def update_accuracy(real):

    global wins,loss,total_predict,prediction

    if prediction is None:
        return

    total_predict+=1

    if prediction==real:
        wins+=1
    else:
        loss+=1


def accuracy():

    if total_predict==0:
        return 0

    return round(wins/total_predict*100,2)


# collector
def collector():

    global last_period
    global latest_period
    global latest_result
    global latest_total

    while True:

        try:

            r=requests.get(API,timeout=10)

            data=r.json()

            period=data["period"]
            result=normalize(data["result"])
            total=data["total"]

            if period!=last_period:

                update_accuracy(result)

                last_period=period

                latest_period=period
                latest_result=result
                latest_total=total

                save_history(period,result,total)

                ai_predict()

        except Exception as e:
            print("API error:",e)

        time.sleep(12)


# HTML UI
@app.route("/")
def home():

    return render_template_string("""

<html>

<head>

<title>AI Tai Xiu GOD Engine</title>

<style>

body{
background:#0f0f0f;
color:white;
font-family:Arial;
text-align:center;
margin-top:60px;
}

.box{
background:#1c1c1c;
padding:30px;
border-radius:12px;
width:380px;
margin:auto;
}

.result{
font-size:42px;
color:#00ffcc;
margin:20px;
}

.stat{
margin:6px;
}

</style>

</head>

<body>

<div class="box">

<h2>AI Tai Xiu GOD Engine</h2>

<div id="data">Loading...</div>

</div>

<script>

async function load(){

let r=await fetch("/api")
let d=await r.json()

document.getElementById("data").innerHTML=

"<div class='result'>"+(d.prediction||"-")+"</div>"+

"<div class='stat'>Phiên: "+(d.period||"-")+"</div>"+

"<div class='stat'>Kết quả: "+(d.result||"-")+"</div>"+

"<div class='stat'>Tổng xúc xắc: "+(d.total||"-")+" 🎲</div>"+

"<div class='stat'>Confidence: "+(d.confidence||"-")+"%</div>"+

"<div class='stat'>Wins: "+d.wins+"</div>"+

"<div class='stat'>Loss: "+d.loss+"</div>"+

"<div class='stat'>Accuracy: "+d.accuracy+"%</div>"+

"<div class='stat'>"+d.status+"</div>"

}

setInterval(load,5000)

load()

</script>

</body>

</html>

""")


@app.route("/api")
def api():

    return jsonify({

        "period":latest_period,
        "result":latest_result,
        "total":latest_total,
        "prediction":prediction,
        "confidence":confidence,
        "wins":wins,
        "loss":loss,
        "accuracy":accuracy(),
        "status":status

    })


threading.Thread(target=collector,daemon=True).start()


if __name__=="__main__":

    port=int(os.environ.get("PORT",10000))

    app.run(host="0.0.0.0",port=port)
