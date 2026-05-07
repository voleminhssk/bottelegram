# ================================
# AI FOOTBALL ANALYZER FULL APP
# Python + HTML + CSS + JS
# ================================

# CÀI:
# pip install flask requests beautifulsoup4 lxml

from flask import Flask, render_template_string, request
import requests
from bs4 import BeautifulSoup
import random
import math

app = Flask(__name__)

BASE_URL = "https://bongda24h.vn"

# ==========================================
# AI ENGINE
# ==========================================

def poisson(avg, goals):

    return ((avg ** goals) * math.exp(-avg)) / math.factorial(goals)

def monte_carlo(home_power, away_power, simulations=10000):

    home_wins = 0
    away_wins = 0
    draws = 0

    for _ in range(simulations):

        home = random.gauss(home_power, 10)
        away = random.gauss(away_power, 10)

        if home > away:
            home_wins += 1

        elif away > home:
            away_wins += 1

        else:
            draws += 1

    return {

        "home_win":
            round(home_wins / simulations * 100, 1),

        "away_win":
            round(away_wins / simulations * 100, 1),

        "draw":
            round(draws / simulations * 100, 1)

    }

def analyze_match(home, away):

    # ==========================
    # FIFA STYLE AI STATS
    # ==========================

    home_attack = random.randint(70, 95)
    away_attack = random.randint(65, 92)

    home_mid = random.randint(70, 94)
    away_mid = random.randint(65, 91)

    home_def = random.randint(68, 93)
    away_def = random.randint(60, 90)

    home_form = random.randint(65, 95)
    away_form = random.randint(60, 92)

    home_h2h = random.randint(50, 90)
    away_h2h = random.randint(45, 85)

    home_power = (

        home_attack * 0.25 +
        home_mid * 0.20 +
        home_def * 0.20 +
        home_form * 0.20 +
        home_h2h * 0.15

    )

    away_power = (

        away_attack * 0.25 +
        away_mid * 0.20 +
        away_def * 0.20 +
        away_form * 0.20 +
        away_h2h * 0.15

    )

    simulations = monte_carlo(
        home_power,
        away_power
    )

    if home_power > away_power:
        winner = home
    else:
        winner = away

    home_goal_avg = home_attack / 40
    away_goal_avg = away_attack / 42

    predicted_home_goals = max(
        range(6),
        key=lambda x: poisson(home_goal_avg, x)
    )

    predicted_away_goals = max(
        range(6),
        key=lambda x: poisson(away_goal_avg, x)
    )

    confidence = round(
        abs(home_power - away_power),
        1
    )

    return {

        "winner": winner,

        "confidence": confidence,

        "home_attack": home_attack,
        "away_attack": away_attack,

        "home_mid": home_mid,
        "away_mid": away_mid,

        "home_def": home_def,
        "away_def": away_def,

        "home_form": home_form,
        "away_form": away_form,

        "home_h2h": home_h2h,
        "away_h2h": away_h2h,

        "home_power": round(home_power, 1),
        "away_power": round(away_power, 1),

        "home_win": simulations["home_win"],
        "away_win": simulations["away_win"],
        "draw": simulations["draw"],

        "score_prediction":
            f"{predicted_home_goals}-{predicted_away_goals}"

    }

# ==========================================
# HOME PAGE
# ==========================================

@app.route("/")
def home():

    url = f"{BASE_URL}/bong-da/lich-thi-dau-amp.html"

    html = requests.get(url).text

    soup = BeautifulSoup(html, "lxml")

    matches = []

    links = soup.find_all("a")

    added = set()

    for link in links:

        href = link.get("href")
        text = link.text.strip()

        if not href:
            continue

        if "truc-tiep-ket-qua" in href:

            full_url = BASE_URL + href

            if full_url in added:
                continue

            added.add(full_url)

            matches.append({

                "name": text if text else "Football Match",
                "url": full_url

            })

    html_page = """

<!DOCTYPE html>
<html lang="vi">

<head>

<meta charset="UTF-8">

<title>AI FIFA FOOTBALL ANALYZER</title>

<style>

body{

    margin:0;
    background:#050816;
    color:white;
    font-family:Arial;

}

.header{

    text-align:center;
    padding:30px;

    font-size:40px;
    font-weight:bold;

    color:#00ffd5;

    text-shadow:0 0 20px #00ffd5;

}

.container{

    width:90%;
    margin:auto;

}

.card{

    background:#111827;

    margin:20px 0;

    padding:25px;

    border-radius:20px;

    border:1px solid #00ffd5;

    box-shadow:0 0 20px rgba(0,255,213,0.2);

}

.card h2{

    margin:0 0 15px 0;

}

.btn{

    background:#00ffd5;
    color:black;

    padding:12px 25px;

    border-radius:12px;

    text-decoration:none;

    font-weight:bold;

    display:inline-block;

}

.btn:hover{

    transform:scale(1.05);

}

</style>

</head>

<body>

<div class="header">

⚽ AI FIFA FOOTBALL ANALYZER

</div>

<div class="container">

{% for match in matches %}

<div class="card">

<h2>{{match.name}}</h2>

<a class="btn"

href="/analyze?url={{match.url}}">

XEM AI PHÂN TÍCH

</a>

</div>

{% endfor %}

</div>

</body>
</html>

"""

    return render_template_string(
        html_page,
        matches=matches[:30]
    )

# ==========================================
# ANALYZE PAGE
# ==========================================

@app.route("/analyze")
def analyze():

    match_url = request.args.get("url")

    # ==================================
    # TỰ PARSE TEAM TỪ URL
    # ==================================

    slug = match_url.split("/")[-1]

    slug = slug.replace(".html", "")

    teams = slug.split("-vs-")

    if len(teams) >= 2:

        home_team = teams[0].replace("-", " ").title()
        away_team = teams[1].split("-")[0].replace("-", " ").title()

    else:

        home_team = "Home Team"
        away_team = "Away Team"

    analysis = analyze_match(
        home_team,
        away_team
    )

    page = """

<!DOCTYPE html>
<html lang="vi">

<head>

<meta charset="UTF-8">

<title>AI Match Analysis</title>

<style>

body{

    margin:0;
    background:#050816;
    color:white;
    font-family:Arial;

}

.container{

    width:90%;
    margin:auto;

    padding:30px;

}

.card{

    background:#111827;

    padding:30px;

    border-radius:20px;

    border:1px solid #00ffd5;

    box-shadow:0 0 25px rgba(0,255,213,0.2);

}

.title{

    text-align:center;

    font-size:45px;

    color:#00ffd5;

}

.winner{

    font-size:35px;

    color:#00ff95;

}

.bar{

    width:100%;
    background:#1f2937;

    border-radius:20px;

    overflow:hidden;

    margin-bottom:15px;

}

.fill{

    height:22px;

    background:#00ffd5;

}

.stat{

    margin-top:25px;

}

.probability{

    font-size:22px;

}

.score{

    font-size:40px;

    color:#ffcc00;

}

</style>

</head>

<body>

<div class="container">

<div class="card">

<h1 class="title">

{{home}} VS {{away}}

</h1>

<h2 class="winner">

🏆 AI DỰ ĐOÁN:
{{analysis.winner}}

</h2>

<h2 class="score">

⚽ {{analysis.score_prediction}}

</h2>

<h3>

🤖 AI CONFIDENCE:
{{analysis.confidence}}%

</h3>

<hr>

<div class="probability">

<p>
{{home}} WIN:
{{analysis.home_win}}%
</p>

<p>
DRAW:
{{analysis.draw}}%
</p>

<p>
{{away}} WIN:
{{analysis.away_win}}%
</p>

</div>

<div class="stat">

<h2>ATTACK</h2>

<p>{{home}}: {{analysis.home_attack}}</p>

<div class="bar">
<div class="fill"
style="width:{{analysis.home_attack}}%">
</div>
</div>

<p>{{away}}: {{analysis.away_attack}}</p>

<div class="bar">
<div class="fill"
style="width:{{analysis.away_attack}}%">
</div>
</div>

</div>

<div class="stat">

<h2>MIDFIELD</h2>

<p>{{home}}: {{analysis.home_mid}}</p>

<div class="bar">
<div class="fill"
style="width:{{analysis.home_mid}}%">
</div>
</div>

<p>{{away}}: {{analysis.away_mid}}</p>

<div class="bar">
<div class="fill"
style="width:{{analysis.away_mid}}%">
</div>
</div>

</div>

<div class="stat">

<h2>DEFENSE</h2>

<p>{{home}}: {{analysis.home_def}}</p>

<div class="bar">
<div class="fill"
style="width:{{analysis.home_def}}%">
</div>
</div>

<p>{{away}}: {{analysis.away_def}}</p>

<div class="bar">
<div class="fill"
style="width:{{analysis.away_def}}%">
</div>
</div>

</div>

<div class="stat">

<h2>FORM</h2>

<p>{{home}}: {{analysis.home_form}}</p>

<div class="bar">
<div class="fill"
style="width:{{analysis.home_form}}%">
</div>
</div>

<p>{{away}}: {{analysis.away_form}}</p>

<div class="bar">
<div class="fill"
style="width:{{analysis.away_form}}%">
</div>
</div>

</div>

<div class="stat">

<h2>HEAD TO HEAD</h2>

<p>{{home}}: {{analysis.home_h2h}}</p>

<div class="bar">
<div class="fill"
style="width:{{analysis.home_h2h}}%">
</div>
</div>

<p>{{away}}: {{analysis.away_h2h}}</p>

<div class="bar">
<div class="fill"
style="width:{{analysis.away_h2h}}%">
</div>
</div>

</div>

</div>

</div>

</body>

</html>

"""

    return render_template_string(

        page,

        home=home_team,
        away=away_team,

        analysis=analysis

    )

# ==========================================
# RUN
# ==========================================

if __name__ == "__main__":

    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000
    )
