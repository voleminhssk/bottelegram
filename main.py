import os
import time
import requests
import numpy as np
import unicodedata

from datetime import datetime

from telegram import Bot

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, load_model

from sklearn.ensemble import RandomForestClassifier


TOKEN="7789180148:AAHzAdGMxWS3IWXkk-VoVpP8zoAsGkITALQ"
CHAT_ID="-1002260844789"

bot=Bot(token=TOKEN)


API="https://phanmemdudoan.fun/apisun.php"

history_file="history.txt"

window=12
max_history=500
train_interval=20

last_period=None
round_count=0
current_day=datetime.now().day


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

    return data[:max_history]


def build_dataset(history):

    X=[]
    y=[]

    for i in range(len(history)-window):

        X.append(history[i:i+window])
        y.append(history[i+window])

    return np.array(X),np.array(y)


# Transformer nhẹ
def transformer_model():

    inp=Input(shape=(window,1))

    x=MultiHeadAttention(num_heads=2,key_dim=16)(inp,inp)

    x=LayerNormalization()(x)

    x=Dense(32,activation="relu")(x)

    x=GlobalAveragePooling1D()(x)

    out=Dense(1,activation="sigmoid")(x)

    model=Model(inp,out)

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy"
    )

    return model


# LSTM
def lstm_model():

    inp=Input(shape=(window,1))

    x=LSTM(32)(inp)

    out=Dense(1,activation="sigmoid")(x)

    model=Model(inp,out)

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy"
    )

    return model


# Train AI
def train_models(history):

    X,y=build_dataset(history)

    X=X.reshape((X.shape[0],X.shape[1],1))

    # transformer

    if os.path.exists("transformer.h5"):
        model=load_model("transformer.h5")
    else:
        model=transformer_model()

    model.fit(X,y,epochs=2,verbose=0)
    model.save("transformer.h5")

    # lstm

    if os.path.exists("lstm.h5"):
        model=load_model("lstm.h5")
    else:
        model=lstm_model()

    model.fit(X,y,epochs=2,verbose=0)
    model.save("lstm.h5")


# AI Predict
def ai_predict():

    history=read_history()

    if len(history)<50:
        return None

    last=np.array(history[:window]).reshape((1,window,1))

    # transformer

    tf_model=load_model("transformer.h5")

    p1=tf_model.predict(last,verbose=0)[0][0]

    # lstm

    lstm=load_model("lstm.h5")

    p2=lstm.predict(last,verbose=0)[0][0]

    # random forest

    X,y=build_dataset(history)

    rf=RandomForestClassifier(n_estimators=100)

    rf.fit(X,y)

    p3=rf.predict_proba(last.reshape(1,-1))[0][1]

    # markov

    markov={0:{0:1,1:1},1:{0:1,1:1}}

    for i in range(len(history)-1):
        markov[history[i]][history[i+1]]+=1

    last_state=history[0]

    p4=markov[last_state][1]/(
        markov[last_state][0]+markov[last_state][1]
    )

    prob=(p1+p2+p3+p4)/4

    if prob>0.5:
        pred="Tài"
        conf=prob*100
    else:
        pred="Xỉu"
        conf=(1-prob)*100

    return pred,round(conf,2)


def reset_daily():

    global current_day

    if datetime.now().day!=current_day:

        current_day=datetime.now().day

        open(history_file,"w").close()

        bot.send_message(CHAT_ID,"History reset")


def collector():

    global last_period
    global round_count

    while True:

        try:

            r=requests.get(API,timeout=10)

            data=r.json()

            period=data["period"]
            result=normalize(data["result"])
            total=data["total"]

            if period!=last_period:

                last_period=period

                round_count+=1

                with open(history_file,"a",encoding="utf-8") as f:
                    f.write(f"{period},{result},{total}\n")

                history=read_history()

                if round_count % train_interval == 0:

                    train_models(history)

                ai=ai_predict()

                if ai:

                    p,c=ai

                    bot.send_message(

                        CHAT_ID,

                        f"Phiên {period}\n"
                        f"KQ {result}\n\n"
                        f"AI dự đoán: {p}\n"
                        f"Confidence: {c}%"

                    )

        except Exception as e:

            print(e)

        reset_daily()

        time.sleep(15)


print("RENDER AI BOT RUNNING")

collector()
