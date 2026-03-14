from flask import Flask, jsonify
import requests
import threading
import time
import os
import unicodedata
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


app = Flask(__name__)

API="https://phanmemdudoan.fun/apisun.php"

history_file="history.txt"

window=12
max_history=1000

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

    lines.reverse()

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


def ai_predict():

    global prediction,confidence,status

    history=read_history()

    if len(history)<10:

        status=f"Cần ít nhất 10 phiên để AI phân tích ({len(history)}/10)"
        prediction=None
        confidence=None
        return

    status="AI đang phân tích..."

    X,y=build_dataset(history)

    last=np.array(history[:window]).reshape(1,-1)

    models=[]

    models.append(XGBClassifier(n_estimators=150))
    models.append(LGBMClassifier(n_estimators=150))
    models.append(CatBoostClassifier(iterations=150,verbose=0))
    models.append(RandomForestClassifier(n_estimators=200))
    models.append(GradientBoostingClassifier())
    models.append(ExtraTreesClassifier(n_estimators=200))

    probs=[]

    for m in models:

        try:

            m.fit(X,y)

            p=m.predict_proba(last)[0][1]

            probs.append(p)

        except:
            continue


    if len(probs)==0:

        status="AI training error"
        prediction=None
        confidence=None
        return


    prob=sum(probs)/len(probs)

    if prob>0.5:

        prediction="Tài"
        confidence=round(prob*100,2)

    else:

        prediction="Xỉu"
        confidence=round((1-prob)*100,2)

    status="AI Ensemble 6 Model đã phân tích"


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

<title>AI Tai Xiu Analyzer</title>

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

<h2>AI Tai Xiu Analyzer</h2>

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
