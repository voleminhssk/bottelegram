from flask import Flask, jsonify
import requests
import threading
import time
import pickle
import os
from sklearn.neural_network import MLPClassifier

API="https://phanmemdudoan.fun/apisun.php"

app = Flask(__name__)

# tạo file history nếu chưa có
if not os.path.exists("history.txt"):
    open("history.txt","w").close()

last_period=None


def collector():

    global last_period

    while True:

        try:

            r=requests.get(API,timeout=10)

            data=r.json()

            period=data["period"]

            if period!=last_period:

                last_period=period

                line=f'{period},{data["result"]},{data["total"]}\n'

                with open("history.txt","a") as f:
                    f.write(line)

                print("Saved",period)

        except:
            pass

        time.sleep(10)


def train_ai():

    while True:

        try:

            history=[]

            with open("history.txt") as f:

                for line in f:

                    p=line.strip().split(",")

                    if len(p)<3:
                        continue

                    history.append(1 if p[1]=="Tài" else 0)

            if len(history)<50:

                time.sleep(60)
                continue

            X=[]
            y=[]

            for i in range(10,len(history)):

                X.append(history[i-10:i])
                y.append(history[i])

            model=MLPClassifier(
                hidden_layer_sizes=(20,20),
                max_iter=500
            )

            model.fit(X,y)

            pickle.dump(model,open("model.pkl","wb"))

            print("AI trained")

        except:
            pass

        time.sleep(300)


def predict():

    try:

        if not os.path.exists("model.pkl"):
            return {"prediction":"loading","confidence":0}

        history=[]

        with open("history.txt") as f:

            for line in f:

                p=line.strip().split(",")

                if len(p)<3:
                    continue

                history.append(1 if p[1]=="Tài" else 0)

        if len(history)<10:
            return {"prediction":"loading","confidence":0}

        last10=history[-10:]

        model=pickle.load(open("model.pkl","rb"))

        prob=model.predict_proba([last10])[0]

        if prob[1]>prob[0]:

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

        return {"prediction":"loading","confidence":0}


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
    text-align:center;
    font-family:sans-serif;
    margin-top:120px;
    }

    .box{
    background:#1e293b;
    padding:40px;
    width:300px;
    margin:auto;
    border-radius:15px;
    font-size:30px;
    }

    </style>

    </head>

    <body>

    <h1>AI Dự Đoán Tài Xỉu</h1>

    <div class="box" id="predict">
    Loading...
    </div>

    <script>

    async function load(){

    let r=await fetch("/api/predict")

    let d=await r.json()

    document.getElementById("predict").innerHTML=

    `<h2>${d.prediction}</h2>
    <p>${d.confidence}%</p>`

    }

    setInterval(load,5000)

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

    try:

        with open("history.txt") as f:
            data=f.readlines()[-50:]

    except:
        data=[]

    return jsonify(data)


# chạy bot nền
threading.Thread(target=collector).start()
threading.Thread(target=train_ai).start()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
