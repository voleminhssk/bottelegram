from flask import Flask, jsonify
import requests
import threading
import time
import os
import unicodedata
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.neural_network import MLPClassifier

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import StackingClassifier

app = Flask(__name__)

API="https://phanmemdudoan.fun/apisun.php"

history_file="history.txt"

window=12
max_history=5000

last_period=None

latest_period=None
latest_result=None
latest_total=None

prediction=None
confidence=None
status="Đang chờ dữ liệu..."

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


def build_dataset(history):

    X=[]
    y=[]

    for i in range(len(history)-window):

        X.append(history[i:i+window])
        y.append(history[i+window])

    return np.array(X),np.array(y)


def sequence_features(seq):

    arr=np.array(seq)

    return [

        sum(arr),

        np.mean(arr),

        np.std(arr),

        np.max(arr),

        np.min(arr)

    ]


def ai_predict():

    global prediction,confidence,status

    history=read_history()

    if len(history)<window+10:

        status=f"Cần ít nhất {window+10} phiên ({len(history)}/{window+10})"
        prediction=None
        confidence=None
        return

    status="AI đang phân tích..."

    X,y=build_dataset(history)

    X2=[]

    for row in X:

        feats=sequence_features(row)

        X2.append(list(row)+feats)

    X=np.array(X2)

    last=np.array(history[-window:])

    last_feats=sequence_features(last)

    last=np.array(list(last)+last_feats).reshape(1,-1)

    models=[

        ("rf",RandomForestClassifier(n_estimators=200)),

        ("et",ExtraTreesClassifier(n_estimators=200)),

        ("gb",GradientBoostingClassifier()),

        ("hgb",HistGradientBoostingClassifier()),

        ("ada",AdaBoostClassifier(n_estimators=200)),

        ("bag",BaggingClassifier()),

        ("lr",LogisticRegression(max_iter=1000)),

        ("sgd",SGDClassifier(loss="log_loss")),

        ("knn",KNeighborsClassifier()),

        ("dt",DecisionTreeClassifier()),

        ("nb",GaussianNB()),

        ("mlp",MLPClassifier(max_iter=500))

    ]

    calibrated=[]

    for name,m in models:

        try:

            c=CalibratedClassifierCV(m)

            calibrated.append((name,c))

        except:
            pass

    meta=LogisticRegression()

    stack=StackingClassifier(

        estimators=calibrated,

        final_estimator=meta,

        passthrough=True

    )

    try:

        stack.fit(X,y)

        prob=stack.predict_proba(last)[0][1]

    except:

        prediction=None
        confidence=None
        status="AI training error"
        return

    if prob>0.5:

        prediction="Tài"
        confidence=round(prob*100,2)

    else:

        prediction="Xỉu"
        confidence=round((1-prob)*100,2)

    status="AI PRO 12 Models + Stacking đã phân tích"


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

<title>AI Tai Xiu PRO</title>

<style>

body{
background:#111;
color:white;
font-family:Arial;
text-align:center;
margin-top:80px;
}

.box{
background:#222;
padding:30px;
border-radius:10px;
width:360px;
margin:auto;
}

.result{
font-size:40px;
margin:20px;
color:#00ffcc;
}

</style>

</head>

<body>

<div class="box">

<h2>AI Tai Xiu PRO</h2>

<div id="data">Loading...</div>

</div>

<script>

async function load(){

let r=await fetch("/api")
let d=await r.json()

document.getElementById("data").innerHTML=

"<div class='result'>"+(d.prediction||"-")+"</div>"+

"<p>Phiên trước: "+(d.period||"-")+"</p>"+

"<p>Kết quả: "+(d.result||"-")+"</p>"+

"<p>Tổng xúc xắc: "+(d.total||"-")+" 🎲</p>"+

"<p>Confidence: "+(d.confidence||"-")+"%</p>"+

"<p>"+d.status+"</p>"

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
        "status":status

    })


threading.Thread(target=collector,daemon=True).start()


if __name__=="__main__":

    port=int(os.environ.get("PORT",10000))

    app.run(host="0.0.0.0",port=port)
