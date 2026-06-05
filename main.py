
from flask import Flask, request, render_template_string
from PIL import Image
import numpy as np
import base64
import io

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

HTML = """
<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Image Analyzer</title>

<style>
body{
    background:#111827;
    color:white;
    font-family:Arial,sans-serif;
    margin:0;
    padding:20px;
}
.container{
    max-width:1000px;
    margin:auto;
}
.card{
    background:#1f2937;
    border-radius:15px;
    padding:20px;
    margin-bottom:20px;
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
img{
    max-width:100%;
    border-radius:10px;
}
.stats{
    display:grid;
    grid-template-columns:repeat(auto-fit,minmax(200px,1fr));
    gap:15px;
}
.box{
    background:#111827;
    border-radius:10px;
    padding:15px;
    text-align:center;
}
</style>
</head>
<body>

<div class="container">

<div class="card">
<h1>Upload Ảnh</h1>

<form method="post" enctype="multipart/form-data">
<input
    type="file"
    name="image"
    accept=".jpg,.jpeg,.png,.webp"
    required
>
<br>
<button type="submit">Phân Tích</button>
</form>
</div>

{% if image_data %}
<div class="card">
<h2>Ảnh Đã Tải Lên</h2>
<img src="data:image/png;base64,{{ image_data }}">
</div>
{% endif %}

{% if result %}
<div class="card">

<div class="stats">

<div class="box">
<h2>{{ result.red }}</h2>
<p>Điểm Đỏ</p>
</div>

<div class="box">
<h2>{{ result.blue }}</h2>
<p>Điểm Xanh</p>
</div>

<div class="box">
<h2>{{ result.green }}</h2>
<p>Điểm Xanh Lá</p>
</div>

</div>

<br>

<div class="box">
<h3>Chuỗi Nhận Diện</h3>
<p>{{ result.history }}</p>
</div>

</div>
{% endif %}

{% if error %}
<div class="card">
<p>{{ error }}</p>
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
                history.append("R")

            elif b > 170 and r < 150:
                blue += 1
                history.append("B")

            elif g > 170 and r < 170:
                green += 1
                history.append("G")

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
    image_data = None
    error = None

    if request.method == "POST":

        file = request.files.get("image")

        if not file:
            error = "Không nhận được ảnh."

        else:
            try:
                image_bytes = file.read()

                image_data = base64.b64encode(
                    image_bytes
                ).decode()

                img = Image.open(
                    io.BytesIO(image_bytes)
                )

                result = analyze_image(img)

            except Exception as e:
                error = f"Lỗi: {e}"

    return render_template_string(
        HTML,
        result=result,
        image_data=image_data,
        error=error
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
