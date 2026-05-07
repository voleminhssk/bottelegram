# ============================================
# AI FOOTBALL ANALYZER PRO
# Flask + HTML + CSS + JS + SQLite
# Stable Prediction + 24H Cache
# ============================================

# INSTALL:
# pip install flask requests beautifulsoup4 lxml gunicorn

from flask import Flask, render_template_string, request
import requests
from bs4 import BeautifulSoup
import hashlib
import random
import math
import sqlite3
import time
import os

app = Flask(__name__)

BASE_URL = "https://bongda24h.vn"

CACHE_TIME = 86400

# ============================================
# SQLITE
# ============================================

conn = sqlite3.connect(
    "predictions.db",
    check_same_thread=False
)

cursor = conn.cursor()

cursor.execute("""

CREATE TABLE IF NOT EXISTS predictions (

    match_id TEXT PRIMARY KEY,

    home TEXT,
    away TEXT,

    winner TEXT,

    confidence REAL,

    score_prediction TEXT,

    home_win REAL,
    draw REAL,
    away_win REAL,

    created_at INTEGER

)

""")

conn.commit()

# ============================================
# STABLE RANDOM
# ============================================

def stable_random(seed_text):

    seed = int(
        hashlib.md5(seed_text.encode()).hexdigest(),
        16
    )

    rng = random.Random(seed)

    return rng

# ============================================
# POISSON
# ============================================

def poisson(avg, goals):

    return (
        (avg ** goals)
        * math.exp(-avg)
    ) / math.factorial(goals)

# ============================================
# MONTE CARLO
# ============================================

def monte_carlo(home_power, away_power, rng):

    home_win = 0
    away_win = 0
    draw = 0

    for _ in range(5000):

        h = rng.gauss(home_power, 8)
        a = rng.gauss(away_power, 8)

        if h > a:
            home_win += 1

        elif a > h:
            away_win += 1

        else:
            draw += 1

    return {

        "home_win":
            round(home_win / 5000 * 100, 1),

        "away_win":
            round(away_win / 5000 * 100, 1),

        "draw":
            round(draw / 5000 * 100, 1)

    }

# ============================================
# RECENT FORM
# ============================================

def recent_form(team):

    rng = stable_random(team)

    forms = []

    results = ["W", "D", "L"]

    for _ in range(5):

        forms.append(
            results[rng.randint(0, 2)]
        )

    return forms

# ============================================
# AI ENGINE
# ============================================

def generate_prediction(match_id, home, away):

    rng = stable_random(match_id)

    home_attack = rng.randint(70, 95)
    away_attack = rng.randint(65, 92)

    home_mid = rng.randint(70, 94)
    away_mid = rng.randint(65, 90)

    home_def = rng.randint(70, 94)
    away_def = rng.randint(65, 90)

    home_rank = rng.randint(1, 6)
    away_rank = rng.randint(1, 10)

    home_form_score = rng.randint(70, 95)
    away_form_score = rng.randint(65, 90)

    home_h2h = rng.randint(60, 90)
    away_h2h = rng.randint(55, 85)

    home_power = (

        home_attack * 0.25 +
        home_mid * 0.20 +
        home_def * 0.20 +
        home_form_score * 0.20 +
        home_h2h * 0.15

    )

    away_power = (

        away_attack * 0.25 +
        away_mid * 0.20 +
        away_def * 0.20 +
        away_form_score * 0.20 +
        away_h2h * 0.15

    )

    simulations = monte_carlo(
        home_power,
        away_power,
        rng
    )

    if home_power > away_power:
        winner = home
    else:
        winner = away

    confidence = round(
        abs(home_power - away_power),
        1
    )

    home_goal_avg = home_attack / 40
    away_goal_avg = away_attack / 42

    predicted_home_goals = max(
        range(6),
        key=lambda x:
            poisson(home_goal_avg, x)
    )

    predicted_away_goals = max(
        range(6),
        key=lambda x:
            poisson(away_goal_avg, x)
    )

    score_prediction = (
        f"{predicted_home_goals}"
        f"-"
        f"{predicted_away_goals}"
    )

    return {

        "home_attack": home_attack,
        "away_attack": away_attack,

        "home_mid": home_mid,
        "away_mid": away_mid,

        "home_def": home_def,
        "away_def": away_def,

        "home_rank": home_rank,
        "away_rank": away_rank,

        "winner": winner,

        "confidence": confidence,

        "score_prediction":
            score_prediction,

        "home_win":
            simulations["home_win"],

        "draw":
            simulations["draw"],

        "away_win":
            simulations["away_win"],

        "home_form":
            recent_form(home),

        "away_form":
            recent_form(away)

    }

# ============================================
# CACHE
# ============================================

def get_prediction(match_id, home, away):

    cursor.execute(

        "SELECT * FROM predictions WHERE match_id=?",

        (match_id,)

    )

    row = cursor.fetchone()

    now = int(time.time())

    if row:

        created_at = row[9]

        if now - created_at < CACHE_TIME:

            return {

                "winner": row[3],

                "confidence": row[4],

                "score_prediction": row[5],

                "home_win": row[6],

                "draw": row[7],

                "away_win": row[8]

            }

    prediction = generate_prediction(
        match_id,
        home,
        away
    )

    cursor.execute("""

    INSERT OR REPLACE INTO predictions

    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)

    """, (

        match_id,

        home,
        away,

        prediction["winner"],

        prediction["confidence"],

        prediction["score_prediction"],

        prediction["home_win"],

        prediction["draw"],

        prediction["away_win"],

        now

    ))

    conn.commit()

    return prediction

# ============================================
# HOME
# ============================================

@app.route("/")
def home():

    url = f"{BASE_URL}/bong-da/lich-thi-dau-amp.html"

    html = requests.get(url).text

    soup = BeautifulSoup(html, "lxml")

    matches = []

    added = set()

    for a in soup.find_all("a"):

        href = a.get("href")
        text = a.text.strip()

        if not href:
            continue

        if "truc-tiep-ket-qua" not in href:
            continue

        full_url = BASE_URL + href

        if full_url in added:
            continue

        added.add(full_url)

        slug = href.split("/")[-1]

        slug = slug.replace(".html", "")

        teams = slug.split("-vs-")

        if len(teams) < 2:
            continue

        home_team = teams[0].replace("-", " ").title()

        away_team = (
            teams[1]
            .split("-")[0]
            .replace("-", " ")
            .title()
        )

        prediction = get_prediction(
            slug,
            home_team,
            away_team
        )

        matches.append({

            "home": home_team,
            "away": away_team,

            "url": full_url,

            "prediction": prediction,

            "time": "19:30",

            "league": "Football League"

        })

    page = """

<!DOCTYPE html>
<html lang="vi">

<head>

<meta charset="UTF-8">

<title>AI FOOTBALL ANALYZER</title>

<style>

body{

    margin:0;
    background:#050816;
    color:white;
    font-family:Arial;

}

.header{

    text-align:center;

    padding:25px;

    font-size:40px;

    color:#00ffd5;

    font-weight:bold;

}

.container{

    width:90%;
    margin:auto;

}

.card{

    background:#111827;

    margin:25px 0;

    padding:25px;

    border-radius:20px;

    border:1px solid #00ffd5;

}

.match{

    font-size:30px;

    font-weight:bold;

}

.league{

    color:#00ffd5;

}

.confidence-bar{

    width:100%;

    background:#1f2937;

    border-radius:20px;

    overflow:hidden;

    margin-top:15px;

}

.fill{

    height:24px;

    background:#00ffd5;

    text-align:center;

    color:black;

    font-weight:bold;

    line-height:24px;

}

.form{

    margin-top:10px;

}

.form span{

    display:inline-block;

    width:35px;
    height:35px;

    text-align:center;

    line-height:35px;

    border-radius:50%;

    margin-right:5px;

    font-weight:bold;

}

.W{

    background:#00ff95;
    color:black;

}

.D{

    background:#ffcc00;
    color:black;

}

.L{

    background:#ff4d4d;

}

.btn{

    display:inline-block;

    margin-top:20px;

    background:#00ffd5;

    color:black;

    text-decoration:none;

    padding:12px 20px;

    border-radius:12px;

    font-weight:bold;

}

</style>

</head>

<body>

<div class="header">

⚽ AI FOOTBALL ANALYZER PRO

</div>

<div class="container">

{% for match in matches %}

<div class="card">

<div class="league">

🏆 {{match.league}}

</div>

<h2 class="match">

{{match.home}}
VS
{{match.away}}

</h2>

<p>
🕒 {{match.time}}
</p>

<h3>

🤖 AI:
{{match.prediction.winner}}

</h3>

<p>

{{match.prediction.home_win}}%
-
DRAW {{match.prediction.draw}}%
-
{{match.prediction.away_win}}%

</p>

<div class="confidence-bar">

<div class="fill"

style="
width:{{match.prediction.confidence}}%
">

{{match.prediction.confidence}}%

</div>

</div>

<h4>🔥 FORM</h4>

<div class="form">

{% for r in match.prediction.home_form %}

<span class="{{r}}">
{{r}}
</span>

{% endfor %}

</div>

<div class="form">

{% for r in match.prediction.away_form %}

<span class="{{r}}">
{{r}}
</span>

{% endfor %}

</div>

<a
class="btn"
href="/analyze?url={{match.url}}"
>

XEM PHÂN TÍCH

</a>

</div>

{% endfor %}

</div>

</body>
</html>

"""

    return render_template_string(
        page,
        matches=matches[:20]
    )

# ============================================
# ANALYZE
# ============================================

@app.route("/analyze")
def analyze():

    url = request.args.get("url")

    slug = url.split("/")[-1]

    slug = slug.replace(".html", "")

    teams = slug.split("-vs-")

    home_team = (
        teams[0]
        .replace("-", " ")
        .title()
    )

    away_team = (
        teams[1]
        .split("-")[0]
        .replace("-", " ")
        .title()
    )

    prediction = get_prediction(
        slug,
        home_team,
        away_team
    )

    return f"""

    <html>

    <head>

    <title>AI Analysis</title>

    <style>

    body{{

        background:#050816;
        color:white;
        font-family:Arial;
        padding:30px;

    }}

    .card{{

        background:#111827;
        padding:30px;
        border-radius:20px;

    }}

    </style>

    </head>

    <body>

    <div class='card'>

    <h1>

    {home_team}
    VS
    {away_team}

    </h1>

    <h2>

    🏆 AI Prediction:
    {prediction["winner"]}

    </h2>

    <h2>

    ⚽ Score:
    {prediction["score_prediction"]}

    </h2>

    <h3>

    🤖 Confidence:
    {prediction["confidence"]}%

    </h3>

    <p>

    {home_team} Win:
    {prediction["home_win"]}%

    </p>

    <p>

    Draw:
    {prediction["draw"]}%

    </p>

    <p>

    {away_team} Win:
    {prediction["away_win"]}%

    </p>

    </div>

    </body>

    </html>

    """

# ============================================
# RUN
# ============================================

if __name__ == "__main__":

    port = int(
        os.environ.get("PORT", 10000)
    )

    app.run(
        host="0.0.0.0",
        port=port
    )
