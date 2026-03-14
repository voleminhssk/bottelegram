from flask import Flask, jsonify
import requests
import threading
import time
import os
import pickle
import unicodedata
import numpy as np
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False

API = "https://phanmemdudoan.fun/apisun.php"

last_period = None
current_day = datetime.now().day


# tạo history
if not os.path.exists("history.txt"):
    open("history.txt","w",encoding="utf-8").close()


# chuẩn hóa chữ tài xỉu
def normalize_result(text):

    text=str(text)
    text=unicodedata.normalize("NFC",text)
    text=text.lower()

    if "tai" in text or "tài" in text:
        return "Tài"

    if "xiu" in text or "xỉu" in text:
        return "Xỉu"

    return text


# reset mỗi ngày
def reset_daily():

    global current_day

    while True:

        if datetime.now().day != current_day:

            current_day = datetime.now().day

            open("history.txt","w").close()

            print("History reset for new day")

        time.sleep(60)


# đọc history newest → oldest
def read_history():

    history=[]

    with open("history.txt",encoding="utf-8") as f:

        lines=f.readlines()

    lines.reverse()

    for line in lines:

        p=line.strip().split(",")

        if len(p)<3:
            continue

        r=normalize_result(p[1])

        if r=="Tài":
            history.append(1)

        elif r=="Xỉu":
            history.append(0)

    return history


# lấy API
def collector():

    global last_period

    while True:

        try:

            r=requests.get(API,timeout=10)

            data=r.json()

            period=data["period"]

            result=normalize_result(data["result"])

            total=data["total"]

            if period!=last_period:

                last_period=period

                line=f"{period},{result},{total}\n"

                with open("history.txt","a",encoding="utf-8") as f:
                    f.write(line)

                print("Saved:",period,result)

        except Exception as e:

            print("API error:",e)

        time.sleep(10)


# train AI
def train_ai():

    while True:

        try:

            history = read_history()

            if len(history) < 50:

                time.sleep(60)
                continue

            window = 20

            X=[]
            y=[]

            for i in range(window,len(history)):

                X.append(history[i-window:i])
                y.append(history[i])

            X=np.array(X)
            y=np.array(y)

            model = RandomForestClassifier(
                n_estimators=300
            )

            model.fit(X,y)

            pickle.dump((model,window),open("model.pkl","wb"))

            print("AI trained:",len(history),"rounds")

        except Exception as e:

            print("Train error:",e)

        time.sleep(120)


# dự đoán
def predict():

    try:

        if not os.path.exists("model.pkl"):

            return {"prediction":"Loading","confidence":0}

        history=read_history()

        model,window = pickle.load(open("model.pkl","rb"))

        if len(history) < window:

            return {"prediction":"Loading","confidence":0}

        last = history[:window]

        prob=model.predict_proba([last])[0]

        if prob[1] > prob[0]:

            return {
                "prediction":"Tài",
                "confidence":round(prob[1]*100,2)
            }

        else:

            return {
                "prediction":"Xỉu",
                "confidence":round(prob[0]*100,2)
            }

    except:

        return {"prediction":"Error","confidence":0}


# WEB
@app.route("/")
def home():

    return """

<h1>AI Tai Xiu Predictor</h1>

<div id="result">Loading...</div>

<script>

async function load(){

let r=await fetch("/api/predict")

let d=await r.json()

document.getElementById("result").innerHTML=

"<h2>"+d.prediction+"</h2><p>"+d.confidence+"%</p>"

}

setInterval(load,5000)

load()

</script>

"""


# API predict
@app.route("/api/predict")
def api_predict():

    return jsonify(predict())


# API history
@app.route("/api/history")
def api_history():

    data=[]

    try:

        with open("history.txt",encoding="utf-8") as f:

            for line in f.readlines()[-100:]:

                p=line.strip().split(",")

                data.append({

                    "period":int(p[0]),
                    "result":p[1],
                    "total":int(p[2])

                })

    except:
        pass

    return jsonify(data)


# threads
threading.Thread(target=collector,daemon=True).start()
threading.Thread(target=train_ai,daemon=True).start()
threading.Thread(target=reset_daily,daemon=True).start()


if __name__=="__main__":

    port=int(os.environ.get("PORT",10000))

    app.run(host="0.0.0.0",port=port)
