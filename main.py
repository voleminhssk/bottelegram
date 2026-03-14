from flask import Flask, jsonify
import requests
import threading
import time
import os
import unicodedata
import numpy as np

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

app = Flask(__name__)

API="https://phanmemdudoan.fun/apisun.php"

history_file="history.txt"

window=10
max_history=1000

last_period=None

latest_result=None
latest_prediction=None
latest_confidence=None
status_message="Đang chờ dữ liệu..."


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

    global status_message

    history=read_history()

    if len(history)<10:

        status_message=f"Chưa đủ dữ liệu AI ({len(history)}/10)"

        return None

    status_message="AI đang phân tích..."

    X,y=build_dataset(history)

    last=np.array(history[:window]).reshape(1,-1)

    # XGBoost

    xgb=XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1
    )

    xgb.fit(X,y)

    p1=xgb.predict_proba(last)[0][1]


    # LightGBM

    lgb=LGBMClassifier(
        n_estimators=200
    )

    lgb.fit(X,y)

    p2=lgb.predict_proba(last)[0][1]


    # CatBoost

    cat=CatBoostClassifier(
        iterations=200,
        verbose=0
    )

    cat.fit(X,y)

    p3=cat.predict_proba(last)[0][1]


    prob=(p1+p2+p3)/3


    if prob>0.5:

        pred="Tài"
        conf=prob*100

    else:

        pred="Xỉu"
        conf=(1-prob)*100


    status_message="AI đã phân tích xong"

    return pred,round(conf,2)


def collector():

    global last_period
    global latest_result
    global latest_prediction
    global latest_confidence

    while True:

        try:

            r=requests.get(API,timeout=10)

            data=r.json()

            period=data["period"]
            result=normalize(data["result"])
            total=data["total"]

            if period!=last_period:

                last_period=period

                save_history(period,result,total)

                latest_result=result

                ai=ai_predict()

                if ai:

                    p,c=ai

                    latest_prediction=p
                    latest_confidence=c

                print("Round:",period,result)

        except Exception as e:

            print(e)

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
margin-top:100px;
}

.box{
background:#222;
padding:30px;
border-radius:10px;
width:350px;
margin:auto;
}

.result{
font-size:40px;
margin:20px;
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

"<p>Kết quả trước: "+(d.result||"-")+"</p>"+

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

        "result":latest_result,
        "prediction":latest_prediction,
        "confidence":latest_confidence,
        "status":status_message

    })


threading.Thread(target=collector,daemon=True).start()


if __name__=="__main__":

    port=int(os.environ.get("PORT",10000))

    app.run(host="0.0.0.0",port=port)
