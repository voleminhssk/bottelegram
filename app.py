from flask import Flask, jsonify
import requests
import threading
import time
import os
import unicodedata
import numpy as np

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False

API="https://phanmemdudoan.fun/apisun.php"

history_file="history.txt"
model_file="model.h5"

window=10
last_period=None
model=None


# tạo history nếu chưa có
if not os.path.exists(history_file):
    open(history_file,"w",encoding="utf-8").close()


# chuẩn hóa chữ
def normalize(text):

    text=str(text)
    text=unicodedata.normalize("NFC",text)
    text=text.lower()

    if "tai" in text or "tài" in text:
        return "Tài"

    if "xiu" in text or "xỉu" in text:
        return "Xỉu"

    return text


# đọc history
def read_history():

    history=[]

    with open(history_file,encoding="utf-8") as f:
        lines=f.readlines()

    lines.reverse()

    for line in lines:

        p=line.strip().split(",")

        if len(p)<3:
            continue

        r=normalize(p[1])

        if r=="Tài":
            history.append(1)

        elif r=="Xỉu":
            history.append(0)

    return history


# tạo model
def build_model():

    m=Sequential()

    m.add(LSTM(32,input_shape=(window,1)))

    m.add(Dense(1,activation="sigmoid"))

    m.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return m


# train AI
def train_ai():

    global model

    history=read_history()

    if len(history) < window+1:
        return

    X=[]
    y=[]

    for i in range(window,len(history)):

        X.append(history[i-window:i])
        y.append(history[i])

    X=np.array(X)
    y=np.array(y)

    X=X.reshape((X.shape[0],X.shape[1],1))

    if model is None:

        if os.path.exists(model_file):
            model=load_model(model_file)
        else:
            model=build_model()

    model.fit(X,y,epochs=3,verbose=0)

    model.save(model_file)

    print("AI trained:",len(history))


# predict
def predict():

    global model

    history=read_history()

    if len(history) < window:

        return {"prediction":"Loading","confidence":0}

    if model is None:

        if os.path.exists(model_file):
            model=load_model(model_file)
        else:
            return {"prediction":"Loading","confidence":0}

    last=history[:window]

    X=np.array(last).reshape((1,window,1))

    prob=float(model.predict(X,verbose=0)[0][0])

    if prob > 0.5:

        return {
            "prediction":"Tài",
            "confidence":round(prob*100,2)
        }

    else:

        return {
            "prediction":"Xỉu",
            "confidence":round((1-prob)*100,2)
        }


# lấy API
def collector():

    global last_period

    while True:

        try:

            r=requests.get(API,timeout=5)

            data=r.json()

            period=data["period"]
            result=normalize(data["result"])
            total=data["total"]

            if period != last_period:

                last_period=period

                line=f"{period},{result},{total}\n"

                with open(history_file,"a",encoding="utf-8") as f:
                    f.write(line)

                print("Saved:",period)

                train_ai()

        except Exception as e:

            print("API error:",e)

        time.sleep(2)


# UI
@app.route("/")
def home():

    return """

<!DOCTYPE html>
<html>

<head>

<title>AI Tài Xỉu</title>

<style>

body{
background:#0f172a;
color:white;
font-family:sans-serif;
text-align:center;
}

.card{
background:#1e293b;
padding:40px;
margin:120px auto;
width:320px;
border-radius:15px;
box-shadow:0 0 25px rgba(0,0,0,0.6);
}

.result{
font-size:48px;
margin-top:20px;
}

.conf{
color:#38bdf8;
font-size:18px;
}

</style>

</head>

<body>

<div class="card">

<h1>AI Tài Xỉu</h1>

<div id="res" class="result">Loading...</div>

<div id="conf" class="conf"></div>

</div>

<script>

async function load(){

let r=await fetch("/api/predict")

let d=await r.json()

document.getElementById("res").innerHTML=d.prediction

document.getElementById("conf").innerHTML="Confidence "+d.confidence+"%"

}

setInterval(load,3000)

load()

</script>

</body>

</html>

"""


@app.route("/api/predict")
def api_predict():

    return jsonify(predict())


@app.route("/api/history")
def api_history():

    data=[]

    with open(history_file,encoding="utf-8") as f:

        for line in f.readlines()[-100:]:

            p=line.strip().split(",")

            data.append({

                "period":int(p[0]),
                "result":p[1],
                "total":int(p[2])

            })

    return jsonify(data)


threading.Thread(target=collector,daemon=True).start()


if __name__=="__main__":

    port=int(os.environ.get("PORT",10000))

    app.run(host="0.0.0.0",port=port)

