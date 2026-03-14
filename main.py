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


# tạo history
if not os.path.exists(history_file):
    open(history_file,"w",encoding="utf-8").close()


# chuẩn hóa tài xỉu
def normalize(text):

    text=str(text)
    text=unicodedata.normalize("NFC",text)
    text=text.lower()

    if "tai" in text or "tài" in text:
        return "Tài"

    if "xiu" in text or "xỉu" in text:
        return "Xỉu"

    return text


# đọc tổng xúc xắc
def read_totals():

    totals=[]

    try:

        with open(history_file,encoding="utf-8") as f:

            for line in f.readlines():

                p=line.strip().split(",")

                if len(p)<3:
                    continue

                try:
                    totals.append(int(p[2]))
                except:
                    pass

    except:
        pass

    return totals


# AI phân tích
def ai_analyze():

    totals = read_totals()

    n=len(totals)

    if n<6:

        return {
            "prediction":"Loading",
            "confidence":0,
            "rounds":n
        }


    # chuỗi 3 phiên gần nhất
    pattern = totals[-3:]


    tai=0
    xiu=0
    matches=0


    # tìm pattern trong history
    for i in range(n-4):

        if totals[i:i+3]==pattern:

            next_total=totals[i+3]

            matches+=1

            if next_total>=11:
                tai+=1
            else:
                xiu+=1


    # nếu trùng >=3 lần
    if matches>=3:

        if tai>xiu:

            conf=tai/(tai+xiu)*100

            return {
                "prediction":"Tài",
                "confidence":round(conf,2),
                "rounds":n,
                "pattern":pattern,
                "matches":matches
            }

        else:

            conf=xiu/(tai+xiu)*100

            return {
                "prediction":"Xỉu",
                "confidence":round(conf,2),
                "rounds":n,
                "pattern":pattern,
                "matches":matches
            }


    # fallback AI nếu không đủ pattern
    tai_score=0
    xiu_score=0


    for t in totals:

        if t>=11:
            tai_score+=1
        else:
            xiu_score+=1


    total=tai_score+xiu_score


    if tai_score>xiu_score:

        return {
            "prediction":"Tài",
            "confidence":round(tai_score/total*100,2),
            "rounds":n,
            "pattern":"AI fallback"
        }

    else:

        return {
            "prediction":"Xỉu",
            "confidence":round(xiu_score/total*100,2),
            "rounds":n,
            "pattern":"AI fallback"
        }


# lấy API
def collector():

    global last_period

    while True:

        try:

            r=requests.get(API,timeout=10)

            data=r.json()

            period=data.get("period")
            result=normalize(data.get("result"))
            total=data.get("total")

            if not period or not total:
                time.sleep(3)
                continue


            if period!=last_period:

                last_period=period

                line=f"{period},{result},{total}\n"

                with open(history_file,"a",encoding="utf-8") as f:
                    f.write(line)

                print("Saved:",period,result,total)

        except Exception as e:

            print("API error:",e)

        time.sleep(3)


# reset history mỗi ngày
def reset_daily():

    global current_day

    while True:

        if datetime.now().day!=current_day:

            current_day=datetime.now().day

            open(history_file,"w",encoding="utf-8").close()

            print("History reset")

        time.sleep(60)


# web
@app.route("/")
def home():

    return """

<h1>AI Tai Xiu Pattern Analyzer</h1>

<div id="result">Loading...</div>

<script>

async function load(){

let r=await fetch("/api/predict")
let d=await r.json()

document.getElementById("result").innerHTML=

"<h2>"+d.prediction+"</h2>"+
"<p>Confidence: "+d.confidence+"%</p>"+
"<p>Rounds analyzed: "+d.rounds+"</p>"+
"<p>Pattern: "+d.pattern+"</p>"

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

                if len(p)<3:
                    continue

                try:

                    data.append({

                        "period":int(p[0]),
                        "result":p[1],
                        "total":int(p[2])

                    })

                except:
                    pass

    except:
        pass

    return jsonify(data)


# threads
threading.Thread(target=collector,daemon=True).start()
threading.Thread(target=reset_daily,daemon=True).start()


if __name__=="__main__":

    port=int(os.environ.get("PORT",10000))

    app.run(host="0.0.0.0",port=port)
