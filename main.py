from flask import Flask, send_file, jsonify
import requests
import os
import threading
import time

app = Flask(__name__)

API_URL = "https://apisunhpt.onrender.com/sunlon"

DATA_FOLDER = "data"
MAX_LINES = 300
MAX_NUMBERS_PER_LINE = 30
MAX_FILES = 5

running = False
full_flag = False

# ================= INIT =================
def init_files():
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)

    for i in range(1, MAX_FILES + 1):
        file_path = f"{DATA_FOLDER}/data_{i}.txt"
        if not os.path.exists(file_path):
            open(file_path, "w").close()

init_files()

# ================= FILE =================
def get_current_file():
    init_files()  # đảm bảo luôn đủ file

    for i in range(1, MAX_FILES + 1):
        path = f"{DATA_FOLDER}/data_{i}.txt"
        with open(path, "r") as f:
            if len(f.readlines()) < MAX_LINES:
                return path
    return None

def save_number(number):
    global full_flag

    file_path = get_current_file()
    if file_path is None:
        full_flag = True
        return "FULL"

    with open(file_path, "r") as f:
        lines = f.readlines()

    if not lines:
        lines = [""]

    last_line = lines[-1].strip()

    if last_line.endswith("],"):
        last_line = last_line[:-2]
    elif last_line.endswith("]"):
        last_line = last_line[:-1]

    numbers = last_line.replace("[", "").split(",")
    numbers = [n for n in numbers if n != ""]

    if len(numbers) < MAX_NUMBERS_PER_LINE:
        numbers.append(str(number))
        lines[-1] = "[" + ",".join(numbers) + "],\n"
    else:
        lines.append(f"[{number}],\n")

    with open(file_path, "w") as f:
        f.writelines(lines)

    return "OK"

# ================= AUTO =================
def auto_fetch():
    global running

    while running:
        if full_flag:
            break

        init_files()

        try:
            res = requests.get(API_URL).json()
            tong = res.get("tong")

            status = save_number(tong)

            if status == "FULL":
                running = False

        except Exception as e:
            print("Lỗi:", e)

        time.sleep(3)

# ================= WEB =================
@app.route("/")
def index():
    files = os.listdir(DATA_FOLDER)

    html = f"""
    <html>
    <head>
    <title>API LOGGER PRO MAX</title>
    <style>
    body {{
        background:#0f172a;
        color:white;
        font-family:Arial;
        text-align:center;
    }}
    .box {{
        background:#1e293b;
        margin:10px auto;
        padding:15px;
        border-radius:12px;
        width:90%;
        max-width:600px;
    }}
    button {{
        padding:10px;
        margin:5px;
        border:none;
        border-radius:8px;
        cursor:pointer;
        font-weight:bold;
    }}
    .start{{background:#22c55e}}
    .stop{{background:#ef4444}}
    .reset{{background:#f59e0b}}
    .view{{background:#3b82f6}}
    .download{{background:#8b5cf6}}
    textarea {{
        width:100%;
        height:150px;
        background:#020617;
        color:#22c55e;
        border-radius:8px;
        padding:10px;
    }}
    </style>
    </head>

    <body>

    <h2>🚀 API LOGGER PRO MAX</h2>

    <div class="box">
        <button class="start" onclick="start()">▶ Start</button>
        <button class="stop" onclick="stop()">⛔ Stop</button>
        <button class="reset" onclick="reset()">🔄 Reset</button>
        <h3 id="status">Đang chờ...</h3>
    </div>
    """

    for f in files:
        html += f"""
        <div class="box">
            <h3>{f}</h3>
            <button class="view" onclick="viewFile('{f}')">👁 Xem</button>
            <a href="/download/{f}">
                <button class="download">📥 Download</button>
            </a>
            <textarea id="content_{f}"></textarea>
        </div>
        """

    html += """
    <script>

    function start(){ fetch('/start') }
    function stop(){ fetch('/stop') }

    function reset(){
        if(confirm("Reset toàn bộ dữ liệu?")){
            fetch('/reset').then(()=>location.reload())
        }
    }

    function viewFile(name){
        fetch('/view/'+name)
        .then(res=>res.text())
        .then(data=>{
            document.getElementById("content_"+name).value=data
        })
    }

    setInterval(()=>{
        fetch('/status').then(r=>r.json()).then(d=>{
            document.getElementById("status").innerHTML=d.msg
        })
    },2000)

    </script>

    </body>
    </html>
    """

    return html

@app.route("/view/<filename>")
def view_file(filename):
    with open(f"{DATA_FOLDER}/{filename}", "r") as f:
        return f.read()

@app.route("/start")
def start():
    global running, full_flag

    if full_flag:
        return jsonify({"msg":"FULL - cần reset"})

    if not running:
        running = True
        threading.Thread(target=auto_fetch).start()

    return jsonify({"msg":"RUNNING"})

@app.route("/stop")
def stop():
    global running
    running = False
    return jsonify({"msg":"STOPPED"})

@app.route("/reset")
def reset():
    global running, full_flag

    running = False
    full_flag = False

    for i in range(1, MAX_FILES + 1):
        open(f"{DATA_FOLDER}/data_{i}.txt", "w").close()

    return jsonify({"msg":"RESET DONE"})

@app.route("/status")
def status():
    if full_flag:
        return jsonify({"msg":"⚠️ FULL 5 FILE - BẤM RESET"})
    elif running:
        return jsonify({"msg":"▶ ĐANG CHẠY"})
    else:
        return jsonify({"msg":"⛔ ĐÃ DỪNG"})

@app.route("/download/<filename>")
def download(filename):
    return send_file(f"{DATA_FOLDER}/{filename}", as_attachment=True)

# ================= RUN =================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
