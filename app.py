from flask import Flask, jsonify
import requests
import threading
import time
import os
import pickle
import unicodedata
import numpy as np

from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False

API="https://phanmemdudoan.fun/apisun.php"

history_file="history.txt"
model_file="model.pkl"

last_period=None
model=None
window=10


# tạo history
if not os.path.exists(history_file):
    open(history_file,"w",encoding="utf-8").close()


# sửa unicode tài xỉu
def normalize(text):

    text=str(text)
    text=unicodedata.normalize("NFC",text)
    text=text.lower()

    if "tai" in text or "tài" in text:
        return "Tài"

    if "xiu" in text or "xỉu" in text:
        return "Xỉu"

    return text


# đọc history newest -> oldest
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

    model=RandomForestClassifier(
        n_estimators=400,
        max_depth=10
    )

    model.fit(X,y)

    pickle.dump(model,open(model_file,"wb"))

    print("AI trained:",len(history),"rounds")


# dự đoán
def predict():

    global model

    history=read_history()

    if len(history) < window:
        return {"prediction":"Loading","confidence":0}

    if model is None and os.path.exists(model_file):

        model=pickle.load(open(model_file,"rb"))

    if model is None:

        return {"prediction":"Loading","confidence":0}

    last=history[:window]

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


# collector API nhanh
def collector():

    global last_period

    while True:

        try:

            r=requests.get(API,timeout=5)

            data=r.json()

            period=data["period"]
            result=normalize(data["result"])
            total=data["total"]

            if period!=last_period:

                last_period=period

                line=f"{period},{result},{total}\n"

                with open(history_file,"a",encoding="utf-8") as f:
                    f.write(line)

                print("Saved:",period)

                train_ai()

        except Exception as e:

            print("API error",e)

        time.sleep(2)


# web
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

"<h2>"+d.prediction+"</h2>"+
"<p>"+d.confidence+"%</p>"

}

setInterval(load,3000)

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

        with open(history_file,encoding="utf-8") as f:

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


threading.Thread(target=collector,daemon=True).start()


if __name__=="__main__":

    port=int(os.environ.get("PORT",10000))

    app.run(host="0.0.0.0",port=port)
