from flask import Flask, jsonify
import requests
import threading
import time
import os
import unicodedata
from datetime import datetime

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False

API = "https://phanmemdudoan.fun/apisun.php"

history_file = "history.txt"

last_period = None
current_day = datetime.now().day


# tạo history nếu chưa có
if not os.path.exists(history_file):
    open(history_file,"w",encoding="utf-8").close()


# sửa lỗi chữ tài xỉu
def normalize(text):

    text=str(text)
    text=unicodedata.normalize("NFC",text)
    text=text.lower()

    if "tai" in text or "tài" in text:
        return "Tài"

    if "xiu" in text or "xỉu" in text:
        return "Xỉu"

    return text


# đọc history mới -> cũ
def read_history():

    data=[]

    try:

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

    except:
        pass

    return data


# AI phân tích
def ai_analyze():

    history = read_history()

    if len(history)==0:

        return {
            "prediction":"Loading",
            "confidence":0
        }

    tai = history.count(1)
    xiu = history.count(0)

    total = tai + xiu

    if total == 0:
        return {"prediction":"Loading","confidence":0}

    tai_rate = tai/total
    xiu_rate = xiu/total

    if tai_rate > xiu_rate:

        return {

            "prediction":"Tài",
            "confidence":round(tai_rate*100,2),
            "rounds":total

        }

    else:

        return {

            "prediction":"Xỉu",
            "confidence":round(xiu_rate*100,2),
            "rounds":total

        }


# lấy API
def collector():

    global last_period

    while True:

        try:

            r=requests.get(API,timeout=10)

            data=r.json()

            period=data["period"]
            result=normalize(data["result"])
            total=data["total"]

            if period!=last_period:

                last_period=period

                line=f"{period},{result},{total}\n"

                with open(history_file,"a",encoding="utf-8") as f:
                    f.write(line)

                print("Saved:",period,result)

        except Exception as e:

            print("API error:",e)

        time.sleep(5)


# reset mỗi ngày
def reset_daily():

    global current_day

    while True:

        if datetime.now().day != current_day:

            current_day = datetime.now().day

            open(history_file,"w").close()

            print("History reset")

        time.sleep(60)


# web
@app.route("/")
def home():

    return """

<h1>AI Tai Xiu Analyzer</h1>

<div id="result">Loading...</div>

<script>

async function load(){

let r=await fetch("/api/predict")
let d=await r.json()

document.getElementById("result").innerHTML=

"<h2>"+d.prediction+"</h2>"+
"<p>Confidence: "+d.confidence+"%</p>"+
"<p>Rounds analyzed: "+d.rounds+"</p>"

}

setInterval(load,5000)
load()

</script>

"""


# api predict
@app.route("/api/predict")
def api_predict():

    return jsonify(ai_analyze())


# api history
@app.route("/api/history")
def api_history():

    data=[]

    try:

        with open(history_file,encoding="utf-8") as f:

            for line in f.readlines():

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
threading.Thread(target=reset_daily,daemon=True).start()


if __name__=="__main__":

    port=int(os.environ.get("PORT",10000))

    app.run(host="0.0.0.0",port=port)
