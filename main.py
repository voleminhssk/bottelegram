import os
import time
import threading
import requests
import numpy as np
import unicodedata

from datetime import datetime
from flask import Flask

from telegram import Bot

from sklearn.ensemble import RandomForestClassifier


# ==============================
# TELEGRAM
# ==============================

TOKEN="7789180148:AAHzAdGMxWS3IWXkk-VoVpP8zoAsGkITALQ"
CHAT_ID="-1002260844789"

bot=Bot(token=TOKEN)


# ==============================
# CONFIG
# ==============================

API="https://phanmemdudoan.fun/apisun.php"

history_file="history.txt"

window=12
max_history=500

last_period=None
current_day=datetime.now().day


app = Flask(__name__)


# ==============================
# NORMALIZE
# ==============================

def normalize(text):

    text=str(text)
    text=unicodedata.normalize("NFC",text).lower()

    if "tai" in text or "tài" in text:
        return "Tài"

    if "xiu" in text or "xỉu" in text:
        return "Xỉu"

    return text


# ==============================
# READ HISTORY
# ==============================

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

    return data[:max_history]


# ==============================
# DATASET
# ==============================

def build_dataset(history):

    X=[]
    y=[]

    for i in range(len(history)-window):

        X.append(history[i:i+window])
        y.append(history[i+window])

    return np.array(X),np.array(y)


# ==============================
# AI PREDICT
# ==============================

def ai_predict():

    history=read_history()

    if len(history)<50:
        return None

    X,y=build_dataset(history)

    model=RandomForestClassifier(
        n_estimators=200
    )

    model.fit(X,y)

    last=np.array(history[:window]).reshape(1,-1)

    prob=model.predict_proba(last)[0][1]

    if prob>0.5:

        pred="Tài"
        conf=prob*100

    else:

        pred="Xỉu"
        conf=(1-prob)*100

    return pred,round(conf,2)


# ==============================
# RESET DAILY
# ==============================

def reset_daily():

    global current_day

    if datetime.now().day!=current_day:

        current_day=datetime.now().day

        open(history_file,"w").close()

        bot.send_message(
            CHAT_ID,
            "History reset"
        )


# ==============================
# COLLECT DATA
# ==============================

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

                with open(history_file,"a",encoding="utf-8") as f:

                    f.write(f"{period},{result},{total}\n")

                ai=ai_predict()

                if ai:

                    p,c=ai

                    bot.send_message(

                        CHAT_ID,

                        f"Phiên {period}\n"
                        f"KQ: {result}\n\n"
                        f"AI dự đoán: {p}\n"
                        f"Confidence: {c}%"

                    )

        except Exception as e:

            print("Error:",e)

        reset_daily()

        time.sleep(15)


# ==============================
# WEB SERVER (GIỮ RENDER KHÔNG SLEEP)
# ==============================

@app.route("/")
def home():
    return "AI BOT RUNNING 24/24"


# ==============================
# START THREAD
# ==============================

threading.Thread(
    target=collector,
    daemon=True
).start()


# ==============================
# RUN SERVER
# ==============================

if __name__ == "__main__":

    port=int(os.environ.get("PORT",10000))

    app.run(
        host="0.0.0.0",
        port=port
    )
