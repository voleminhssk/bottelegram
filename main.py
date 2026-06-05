from flask import Flask, request, render_template_string
from PIL import Image
import numpy as np

app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Baccarat Road Analyzer</title>

<style>
body{
    background:#111827;
    color:white;
    font-family:Arial;
    margin:0;
    padding:20px;
}

.container{
    max-width:1000px;
    margin:auto;
}

.card{
    background:#1f2937;
    padding:20px;
    border-radius:15px;
    margin-bottom:20px;
}

h1{
    text-align:center;
}

input[type=file]{
    width:100%;
    padding:10px;
}

button{
    background:#2563eb;
    color:white;
    border:none;
    padding:12px 20px;
    border-radius:8px;
    cursor:pointer;
    margin-top:10px;
}

.stats{
    display:grid;
    grid-template-columns:repeat(auto-fit,minmax(200px,1fr));
    gap:15px;
}

.box{
    background:#111827;
    padding:15px;
    border-radius:10px;
    text-align:center;
}

.red{color:#ff4d4d}
.blue{color:#55aaff}
.green{color:#55ff88}

.history{
    word-wrap:break-word;
    line-height:1.8;
}
</style>
</head>

<body>

<div class="container">

<div class="card">
<h1>Baccarat Road Analyzer</h1>

<form method="post" enctype="multipart/form-data">
<input type="file" name="image" accept="image/*" required>
<br>
<button type="submit">Phân Tích</button>
</form>
</div>

{% if result %}

<div class="card">

<div class="stats">

<div class="box">
<h2 class="red">{{ result.red }}</h2>
<p>Banker (Đỏ)</p>
</div>

<div class="box">
<h2 class="blue">{{ result.blue }}</h2>
<p>Player (Xanh)</p>
</div>

<div class="box">
<h2 class="green">{{ result.green }}</h2>
<p>Tie (Xanh Lá)</p>
</div>

</div>

<br>

<div class="box">
<h3>Tổng Kết</h3>
<p>Tổng Banker: {{ result.red }}</p>
<p>Tổng Player: {{ result.blue }}</p>
<p>Tổng Tie: {{ result.green }}</p>
</div>

<br>

<div class="box">
<h3>Chuỗi Nhận Diện</h3>
<div class="history">
{{ result.history }}
</div>
</div>

</div>

{% endif %}

</div>

</body>
</html>
"""


def analyze_image(img):
    img = img.convert("RGB")

    arr = np.array(img)

    h, w = arr.shape[:2]

    red = 0
    blue = 0
    green = 0

    history = []

    step = max(10, min(h, w) // 40)

    for y in range(0, h, step):
        for x in range(0, w, step):

            r, g, b = arr[y, x]

            if r > 170 and g < 140 and b < 140:
                red += 1
                history.append("B")

            elif b > 170 and r < 150:
                blue += 1
                history.append("P")

            elif g > 170 and r < 170:
                green += 1
                history.append("T")

    history = history[-200:]

    return {
        "red": red,
        "blue": blue,
        "green": green,
        "history": " ".join(history)
    }


@app.route("/", methods=["GET", "POST"])
def index():

    result = None

    if request.method == "POST":

        file = request.files.get("image")

        if file and file.filename:

            img = Image.open(file.stream)

            result = analyze_image(img)

    return render_template_string(
        HTML,
        result=result
    )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
