from flask import Flask, render_template, jsonify
import requests
from bs4 import BeautifulSoup
import random
import math

app = Flask(__name__)

BASE_URL = "https://bongda24h.vn"

# =========================
# AI ENGINE
# =========================

def poisson(avg, goals):

    return (avg ** goals) * math.exp(-avg) / math.factorial(goals)

def analyze_match(home, away):

    # Demo stats AI
    # Sau này có thể thay bằng dữ liệu thật

    home_form = random.randint(60, 95)
    away_form = random.randint(50, 90)

    home_attack = random.randint(60, 95)
    away_attack = random.randint(50, 90)

    home_defense = random.randint(55, 90)
    away_defense = random.randint(50, 85)

    home_rank = random.randint(1, 10)
    away_rank = random.randint(1, 10)

    home_score = (
        home_form * 0.35 +
        home_attack * 0.25 +
        home_defense * 0.20 +
        (100 - home_rank) * 0.20
    )

    away_score = (
        away_form * 0.35 +
        away_attack * 0.25 +
        away_defense * 0.20 +
        (100 - away_rank) * 0.20
    )

    if home_score > away_score:
        winner = home
    else:
        winner = away

    confidence = round(
        abs(home_score - away_score),
        1
    )

    home_goal_avg = round(home_attack / 40, 2)
    away_goal_avg = round(away_attack / 45, 2)

    predicted_home_goals = max(
        range(5),
        key=lambda x: poisson(home_goal_avg, x)
    )

    predicted_away_goals = max(
        range(5),
        key=lambda x: poisson(away_goal_avg, x)
    )

    return {

        "winner": winner,

        "confidence": confidence,

        "home_score": round(home_score, 1),
        "away_score": round(away_score, 1),

        "home_form": home_form,
        "away_form": away_form,

        "home_attack": home_attack,
        "away_attack": away_attack,

        "home_defense": home_defense,
        "away_defense": away_defense,

        "score_prediction":
            f"{predicted_home_goals}-{predicted_away_goals}"

    }

# =========================
# HOME
# =========================

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

    return render_template(
        "index.html",
        matches=matches[:30]
    )

# =========================
# MATCH ANALYSIS
# =========================

@app.route("/match")
def match():

    # Demo đọc URL
    # Sau này parse thật từ URL

    home_team = "Home Team"
    away_team = "Away Team"

    analysis = analyze_match(
        home_team,
        away_team
    )

    return render_template(

        "match.html",

        home=home_team,
        away=away_team,

        analysis=analysis

    )

if __name__ == "__main__":
    app.run(debug=True)
