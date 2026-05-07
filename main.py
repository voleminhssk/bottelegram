# =========================================================
# AI FOOTBALL ANALYZER PRO
# FULL SOURCE CODE
# Flask + SQLite + Stable AI + 24H Cache
# =========================================================

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

# =========================================================
# DATABASE
# =========================================================

conn = sqlite3.connect(
    "football_ai.db",
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

# =========================================================
# STABLE AI RANDOM
# =========================================================

def stable_random(seed_text):

    seed = int(
        hashlib.md5(seed_text.encode()).hexdigest(),
        16
    )

    return random.Random(seed)

# =========================================================
# POISSON
# =========================================================

def poisson(avg, goals):

    return (
        (avg ** goals)
        * math.exp(-avg)
    ) / math.factorial(goals)

# =========================================================
# RECENT FORM
# =========================================================

def generate_recent_form(team):

    rng = stable_random(team)

    form = []

    results = ["W", "D", "L"]

    for _ in range(5):

        form.append(
            results[rng.randint(0, 2)]
        )

    return form

# =========================================================
# TEAM RANK
# =========================================================

def get_team_rank(team):

    rng = stable_random(team + "_rank")

    return rng.randint(1, 20)

# =========================================================
# RECENT MATCHES
# =========================================================

def recent_matches(team):

    rng = stable_random(team + "_recent")

    clubs = [

        "Liverpool",
        "Barcelona",
        "Arsenal",
        "Chelsea",
        "Bayern",
        "Juventus",
        "PSG",
        "Milan"

    ]

    data = []

    for _ in range(5):

        opponent = clubs[
            rng.randint(0, len(clubs)-1)
        ]

        home_goals = rng.randint(0, 4)
        away_goals = rng.randint(0, 4)

        if home_goals > away_goals:
            result = "W"

        elif home_goals < away_goals:
            result = "L"

        else:
            result = "D"

        data.append({

            "opponent": opponent,

            "score":
                f"{home_goals}-{away_goals}",

            "result": result

        })

    return data

# =========================================================
# MONTE CARLO
# =========================================================

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

# =========================================================
# AI ENGINE
# =========================================================

def generate_prediction(match_id, home, away):

    rng = stable_random(match_id)

    home_attack = rng.randint(70, 95)
    away_attack = rng.randint(65, 92)

    home_mid = rng.randint(70, 94)
    away_mid = rng.randint(65, 90)

    home_def = rng.randint(70, 94)
    away_def = rng.randint(65, 90)

    home_form_score = rng.randint(70, 95)
    away_form_score = rng.randint(65, 90)

    home_h2h = rng.randint(65, 92)
    away_h2h = rng.randint(60, 88)

    home_rank = get_team_rank(home)
    away_rank = get_team_rank(away)

    home_rank_score = 100 - (home_rank * 3)
    away_rank_score = 100 - (away_rank * 3)

    home_power = (

        home_attack * 0.25 +
        home_mid * 0.20 +
        home_def * 0.20 +
        home_form_score * 0.15 +
        home_h2h * 0.10 +
        home_rank_score * 0.10

    )

    away_power = (

        away_attack * 0.25 +
        away_mid * 0.20 +
        away_def * 0.20 +
        away_form_score * 0.15 +
        away_h2h * 0.10 +
        away_rank_score * 0.10

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

    return {

        "winner": winner,

        "confidence": confidence,

        "score_prediction":
            f"{predicted_home_goals}-{predicted_away_goals}",

        "home_win":
            simulations["home_win"],

        "draw":
            simulations["draw"],

        "away_win":
            simulations["away_win"],

        "home_attack": home_attack,
        "away_attack": away_attack,

        "home_mid": home_mid,
        "away_mid": away_mid,

        "home_def": home_def,
        "away_def": away_def,

        "home_rank": home_rank,
        "away_rank": away_rank,

        "home_form":
            generate_recent_form(home),

        "away_form":
            generate_recent_form(away),

        "home_recent":
            recent_matches(home),

        "away_recent":
            recent_matches(away)

    }

# =========================================================
# CACHE
# =========================================================

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

# =========================================================
# HOME
# =========================================================

@app.route("/")
def home():

    url = f"{BASE_URL}/bong-da/lich-thi-dau-amp.html"

    html = requests.get(url).text

    soup = BeautifulSoup(html, "lxml")

    matches = []

    added = set()

    for a in soup.find_all("a"):

        href = a.get("href")

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

        matches.append({

            "home": home_team,
            "away": away_team,

            "url": full_url,

            "time": "Updating",

            "league": "Football League",

            "prediction": prediction

        })

    return render_template_string("""

<!DOCTYPE html>
<html>

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

    font-size:28px;
    font-weight:bold;

}

.rank{

    color:#00ffd5;

}

.form span{

    display:inline-block;

    width:35px;
    height:35px;

    line-height:35px;

    text-align:center;

    border-radius:50%;

    margin-right:5px;

}

.W{background:#00ff95;color:black;}
.D{background:#ffcc00;color:black;}
.L{background:#ff4d4d;}

.bar{

    width:100%;
    background:#1f2937;

    border-radius:20px;

    overflow:hidden;

}

.fill{

    height:24px;

    background:
    linear-gradient(
        90deg,
        #00ff95,
        #00ffd5
    );

    text-align:center;

    color:black;

    line-height:24px;

    font-weight:bold;

}

.btn{

    display:inline-block;

    margin-top:20px;

    padding:12px 20px;

    background:#00ffd5;

    color:black;

    text-decoration:none;

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

<h2 class="match">

{{match.home}}
(#{{match.prediction.home_rank}})

VS

{{match.away}}
(#{{match.prediction.away_rank}})

</h2>

<p>
🕒 {{match.time}}
</p>

<h3>

🏆 AI:
{{match.prediction.winner}}

</h3>

<p>

{{match.prediction.home_win}}%
-
DRAW {{match.prediction.draw}}%
-
{{match.prediction.away_win}}%

</p>

<div class="bar">

<div class="fill"

style="
width:{{match.prediction.confidence}}%
">

{{match.prediction.confidence}}%

</div>

</div>

<h3>🔥 FORM</h3>

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

    """, matches=matches[:20])

# =========================================================
# ANALYZE
# =========================================================

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

    prediction = generate_prediction(
        slug,
        home_team,
        away_team
    )

    return render_template_string("""

<html>

<head>

<title>AI Analysis</title>

<style>

body{

    background:#050816;
    color:white;
    font-family:Arial;
    padding:30px;

}

.card{

    background:#111827;
    padding:30px;
    border-radius:20px;

}

.form span{

    display:inline-block;

    width:35px;
    height:35px;

    line-height:35px;

    text-align:center;

    border-radius:50%;

    margin-right:5px;

}

.W{background:#00ff95;color:black;}
.D{background:#ffcc00;color:black;}
.L{background:#ff4d4d;}

</style>

</head>

<body>

<div class="card">

<h1>

{{home}}
VS
{{away}}

</h1>

<h2>

🏆 AI:
{{prediction.winner}}

</h2>

<h2>

⚽ Score:
{{prediction.score_prediction}}

</h2>

<h3>

🤖 Confidence:
{{prediction.confidence}}%

</h3>

<hr>

<h2>📊 BXH</h2>

<p>

{{home}}:
#{{prediction.home_rank}}

</p>

<p>

{{away}}:
#{{prediction.away_rank}}

</p>

<hr>

<h2>🔥 FORM</h2>

<div class="form">

{% for r in prediction.home_form %}

<span class="{{r}}">
{{r}}
</span>

{% endfor %}

</div>

<div class="form">

{% for r in prediction.away_form %}

<span class="{{r}}">
{{r}}
</span>

{% endfor %}

</div>

<hr>

<h2>⚽ RECENT MATCHES</h2>

<h3>{{home}}</h3>

{% for m in prediction.home_recent %}

<p>

{{m.opponent}}
-
{{m.score}}
-
{{m.result}}

</p>

{% endfor %}

<h3>{{away}}</h3>

{% for m in prediction.away_recent %}

<p>

{{m.opponent}}
-
{{m.score}}
-
{{m.result}}

</p>

{% endfor %}

</div>

</body>

</html>

    """,

    home=home_team,
    away=away_team,

    prediction=prediction

    )

# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    port = int(
        os.environ.get("PORT", 10000)
    )

    app.run(
        host="0.0.0.0",
        port=port
    )
