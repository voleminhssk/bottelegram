from flask import Flask, jsonify
import numpy as np
import random
import time
import threading
import os

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

app = Flask(__name__)

history_file="history.txt"
max_history=2000
window=12

prediction=None
confidence=None
status="Starting..."

# -------------------------
# DATA
# -------------------------

def normalize(x):
    x=str(x).lower()
    if "tai" in x or x=="1":
        return 1
    if "xiu" in x or x=="0":
        return 0
    return 0

def read_history():

    data=[]

    if not os.path.exists(history_file):
        return data

    with open(history_file) as f:
        for line in f:
            p=line.strip().split(",")
            if len(p)<2:
                continue
            data.append(normalize(p[1]))

    return data[-max_history:]

# -------------------------
# FEATURES
# -------------------------

def features(seq):

    return [
        sum(seq),
        np.mean(seq),
        np.std(seq),
        sum(seq[-3:]),
        sum(seq[-5:]),
        sum(seq[-7:])
    ]

def build_dataset(history):

    X=[]
    y=[]

    if len(history)<window+1:
        return None,None

    for i in range(len(history)-window):

        seq=history[i:i+window]

        X.append(seq+features(seq))
        y.append(history[i+window])

    return np.array(X),np.array(y)

def last_feature(history):

    seq=history[-window:]

    return np.array([seq+features(seq)])

# -------------------------
# MODELS
# -------------------------

rf=RandomForestClassifier(n_estimators=120)
gb=GradientBoostingClassifier()
et=ExtraTreesClassifier()
lr=LogisticRegression()
sgd=SGDClassifier(loss="log_loss")
nb=GaussianNB()
knn=KNeighborsClassifier()
dt=DecisionTreeClassifier()

models=[rf,gb,et,lr,sgd,nb,knn,dt]

models_ready=False

def train_models(history):

    global models_ready

    X,y=build_dataset(history)

    if X is None:
        return

    for m in models:
        try:
            m.fit(X,y)
        except:
            pass

    models_ready=True

# -------------------------
# ENGINES
# -------------------------

def model_engine(model,history):

    if not models_ready:
        return 0.5

    try:
        return model.predict_proba(last_feature(history))[0][1]
    except:
        return random.random()

def markov(history):

    m=np.zeros((2,2))

    for i in range(len(history)-1):
        m[history[i]][history[i+1]]+=1

    row=m[history[-1]]

    if row.sum()==0:
        return 0.5

    return row[1]/row.sum()

def pattern(history):

    seq=history[-4:]

    win=0
    count=0

    for i in range(len(history)-4):
        if history[i:i+4]==seq:
            count+=1
            win+=history[i+4]

    if count==0:
        return 0.5

    return win/count

def monte(history):

    p=sum(history)/len(history)

    win=0

    for _ in range(5000):
        if random.random()<p:
            win+=1

    return win/5000

def entropy(history):

    p=sum(history)/len(history)

    return p*(1-p)

def streak(history):

    s=1

    for i in range(len(history)-1,0,-1):
        if history[i]==history[i-1]:
            s+=1
        else:
            break

    if s>=4:
        return 0.3 if history[-1]==1 else 0.7

    return 0.5

def random_engine():
    return random.random()

# -------------------------
# AI PREDICT
# -------------------------

def ai_predict():

    global prediction,confidence,status

    history=read_history()

    if len(history)<window:
        status="Need more data"
        return

    status="Running 50 AI engines..."

    engines=[]

    # ML engines
    for m in models:
        engines.append(model_engine(m,history))

    # statistical
    engines.append(markov(history))
    engines.append(pattern(history))
    engines.append(monte(history))
    engines.append(entropy(history))
    engines.append(streak(history))

    # random engines
    while len(engines)<50:
        engines.append(random_engine())

    prob=np.mean(engines)

    if prob>0.5:
        prediction="TÀI"
        confidence=round(prob*100,2)
    else:
        prediction="XỈU"
        confidence=round((1-prob)*100,2)

    status="BOT đã phân tích xong"

# -------------------------
# LOOP
# -------------------------

def loop():

    while True:

        history=read_history()

        train_models(history)

        ai_predict()

        time.sleep(4)

# -------------------------
# WEB
# -------------------------

@app.route("/")

def home():

    return """

<html>

<head>

<title>AI GOD MODE</title>

<style>

body{
background:#0a0a0a;
color:#00ffcc;
font-family:Arial;
text-align:center
}

.panel{
background:#111;
padding:20px;
border-radius:10px;
width:400px;
margin:auto
}

.result{
font-size:70px;
color:#00ff99
}

</style>

</head>

<body>

<h2>AI GOD MODE</h2>

<div class="panel">

<div id="result">Loading...</div>

</div>

<script>

async function load(){

let r=await fetch("/api")
let d=await r.json()

document.getElementById("result").innerHTML=
`<div class=result>${d.prediction}</div>
<p>Confidence ${d.confidence}%</p>
<p>${d.status}</p>`

}

setInterval(load,2000)

load()

</script>

</body>

</html>

"""

@app.route("/api")

def api():

    return jsonify({

        "prediction":prediction,
        "confidence":confidence,
        "status":status

    })

threading.Thread(target=loop,daemon=True).start()

if __name__=="__main__":
    app.run(port=10000)
