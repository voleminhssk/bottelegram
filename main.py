
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, jsonify, render_template_string, request

app = Flask(__name__)

HTML = r'''
<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Stock Trade Lab</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    :root{
      --bg:#0b1020; --card:#121a33; --muted:#91a4c7; --text:#eef3ff;
      --line:#223055; --good:#2fe07d; --bad:#ff6b6b; --warn:#ffd166; --accent:#7c9cff;
    }
    *{box-sizing:border-box}
    body{margin:0;font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;background:linear-gradient(180deg,#09101f 0%, #0b1020 100%);color:var(--text)}
    .wrap{max-width:1280px;margin:0 auto;padding:20px}
    .hero{display:grid;grid-template-columns:1.4fr .9fr;gap:16px;align-items:stretch}
    .card{background:rgba(18,26,51,.92);border:1px solid rgba(124,156,255,.15);border-radius:20px;box-shadow:0 18px 40px rgba(0,0,0,.25);padding:18px}
    h1{margin:0 0 8px;font-size:30px;line-height:1.1}
    h3{margin:0 0 10px}
    p{margin:.25rem 0;color:var(--muted)}
    .grid{display:grid;grid-template-columns:repeat(12,1fr);gap:14px;margin-top:16px}
    .span-8{grid-column:span 8}.span-4{grid-column:span 4}.span-12{grid-column:span 12}
    .controls{display:grid;grid-template-columns:repeat(6,1fr);gap:10px}
    .controls input,.controls select,.controls button{width:100%;padding:12px 14px;border-radius:14px;border:1px solid #2b3b69;background:#0e1630;color:var(--text);outline:none}
    .controls button{background:linear-gradient(135deg,#7c9cff,#5de0ff);color:#05101f;font-weight:700;cursor:pointer;border:none}
    .controls button:hover{filter:brightness(1.05)}
    .stats{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}
    .stat{padding:14px;border-radius:16px;background:#0e1630;border:1px solid #23335c}
    .stat .k{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.08em}
    .stat .v{font-size:22px;font-weight:800;margin-top:6px}
    .stat .s{font-size:12px;margin-top:6px;color:var(--muted)}
    .v.good{color:var(--good)}.v.bad{color:var(--bad)}.v.warn{color:var(--warn)}
    .row{display:grid;grid-template-columns:1.5fr .7fr;gap:14px}
    canvas{width:100%;height:440px}
    .tablewrap{overflow:auto;max-height:500px}
    table{width:100%;border-collapse:collapse}
    th,td{padding:10px 12px;border-bottom:1px solid #223055;text-align:left;font-size:14px;white-space:nowrap}
    th{color:#cfe0ff;position:sticky;top:0;background:#101830}
    .chip{display:inline-block;padding:7px 10px;border-radius:999px;background:#0f1a35;border:1px solid #243457;font-size:12px;margin-right:6px;margin-bottom:6px;color:#d9e6ff}
    .note{font-size:13px;color:var(--muted);line-height:1.55}
    .badge{display:inline-block;padding:8px 12px;border-radius:999px;font-weight:800}
    .buy{background:rgba(47,224,125,.15);color:#92f0b8;border:1px solid rgba(47,224,125,.3)}
    .sell{background:rgba(255,107,107,.14);color:#ffb2b2;border:1px solid rgba(255,107,107,.25)}
    .hold{background:rgba(255,209,102,.14);color:#ffe59f;border:1px solid rgba(255,209,102,.25)}
    .footer{margin-top:16px;color:var(--muted);font-size:12px;text-align:center}
    .small{font-size:12px;color:var(--muted)}
    .split{display:flex;gap:8px;flex-wrap:wrap;align-items:center}
    .mini{font-size:12px;color:#c9d7f3;margin-top:8px;line-height:1.5}
    @media (max-width: 980px){
      .hero,.row,.controls,.stats{grid-template-columns:1fr}
      .span-8,.span-4,.span-12{grid-column:span 1}
      canvas{height:360px}
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="hero">
      <div class="card">
        <h1>Stock Trade Lab</h1>
        <p>Dashboard phân tích cổ phiếu theo thị trường, chỉ báo kỹ thuật và quản trị rủi ro. Không có công cụ nào đảm bảo lợi nhuận.</p>
        <div class="controls" style="margin-top:14px;">
          <select id="market">
            <option value="US" selected>US: NYSE / Nasdaq</option>
            <option value="VN">Vietnam: HOSE / HNX / UPCoM</option>
            <option value="HK">Hong Kong: HKEX</option>
            <option value="JP">Japan: TSE / JPX</option>
            <option value="UK">UK: LSE</option>
          </select>
          <input id="symbol" value="AAPL" placeholder="Mã cổ phiếu, ví dụ: AAPL, 0700.HK, 7203.T">
          <select id="period">
            <option value="1mo">1 tháng</option>
            <option value="3mo" selected>3 tháng</option>
            <option value="6mo">6 tháng</option>
            <option value="1y">1 năm</option>
            <option value="2y">2 năm</option>
            <option value="5y">5 năm</option>
          </select>
          <select id="interval">
            <option value="1d" selected>1 ngày</option>
            <option value="1h">1 giờ</option>
            <option value="30m">30 phút</option>
          </select>
          <select id="sma_fast">
            <option value="10">SMA nhanh 10</option>
            <option value="20" selected>SMA nhanh 20</option>
            <option value="30">SMA nhanh 30</option>
          </select>
          <select id="sma_slow">
            <option value="50">SMA chậm 50</option>
            <option value="100" selected>SMA chậm 100</option>
            <option value="200">SMA chậm 200</option>
          </select>
          <button onclick="analyze()">Phân tích</button>
        </div>
        <div class="mini">Ví dụ mã: US = <b>AAPL</b>, HKEX = <b>0700.HK</b>, JPX = <b>7203.T</b>, LSE = <b>VOD.L</b>. Với thị trường Việt Nam, dữ liệu phụ thuộc nguồn cung cấp của Yahoo Finance nên có thể cần thử đúng mã.</div>
        <div style="margin-top:14px" class="split">
          <span class="chip">RSI</span>
          <span class="chip">MACD</span>
          <span class="chip">SMA Cross</span>
          <span class="chip">Bollinger Bands</span>
          <span class="chip">Backtest</span>
          <span class="chip">Risk/Reward</span>
        </div>
      </div>
      <div class="card">
        <div class="stats">
          <div class="stat"><div class="k">Khuyến nghị</div><div class="v" id="rec">--</div><div class="s" id="rec_sub">Sẵn sàng phân tích</div></div>
          <div class="stat"><div class="k">Giá hiện tại</div><div class="v" id="price">--</div><div class="s" id="chg">--</div></div>
          <div class="stat"><div class="k">RSI(14)</div><div class="v" id="rsi">--</div><div class="s" id="rsi_sub">--</div></div>
          <div class="stat"><div class="k">Xu hướng</div><div class="v" id="trend">--</div><div class="s" id="trend_sub">--</div></div>
        </div>
        <div style="margin-top:12px" class="note" id="summary">Nhập mã cổ phiếu rồi bấm phân tích để xem tín hiệu, biểu đồ và bảng dữ liệu gần nhất.</div>
      </div>
    </div>

    <div class="grid">
      <div class="card span-8">
        <div class="row">
          <div>
            <h3>Biểu đồ giá</h3>
            <canvas id="chart"></canvas>
          </div>
          <div>
            <h3>Kết quả giao dịch</h3>
            <div class="card" style="padding:14px;margin-bottom:12px;background:#0e1630">
              <div class="small">Tín hiệu hiện tại</div>
              <div id="signalBadge" class="badge hold" style="margin-top:8px">HOLD</div>
              <div style="height:10px"></div>
              <div class="small">Stop loss gợi ý</div>
              <div id="sl" style="font-size:20px;font-weight:800;margin-top:4px">--</div>
              <div class="small" style="margin-top:10px">Take profit gợi ý</div>
              <div id="tp" style="font-size:20px;font-weight:800;margin-top:4px">--</div>
              <div class="small" style="margin-top:10px">Độ tin cậy</div>
              <div id="conf" style="font-size:20px;font-weight:800;margin-top:4px">--</div>
            </div>
            <div class="card" style="padding:14px;background:#0e1630">
              <div class="small">Backtest SMA đơn giản</div>
              <div id="bt" style="font-size:18px;font-weight:800;margin-top:6px">--</div>
              <div class="small" id="bt_sub" style="margin-top:6px">--</div>
            </div>
          </div>
        </div>
      </div>

      <div class="card span-4">
        <h3>Chỉ báo & dữ liệu gần nhất</h3>
        <div class="tablewrap">
          <table>
            <thead><tr><th>Thời gian</th><th>Open</th><th>High</th><th>Low</th><th>Close</th><th>Volume</th></tr></thead>
            <tbody id="rows"></tbody>
          </table>
        </div>
      </div>

      <div class="card span-12">
        <h3>Ghi chú quản trị rủi ro</h3>
        <div class="note">
          Công cụ này chỉ là dashboard phân tích kỹ thuật. Hãy dùng kèm kiểm soát vốn, dừng lỗ và tự đánh giá rủi ro trước khi giao dịch. Không có tín hiệu nào đảm bảo thắng.
        </div>
      </div>
    </div>

    <div class="footer">Render-ready Flask app • Single file app.py • HTML/CSS/JS embedded</div>
  </div>

<script>
let chart;

function fmt(n, d=2){
  if (n === null || n === undefined || isNaN(n)) return '--';
  return Number(n).toFixed(d);
}

function setBadge(kind, text){
  const el = document.getElementById('signalBadge');
  el.className = 'badge ' + kind;
  el.textContent = text;
}

function renderChart(labels, close, smaFast, smaSlow, upper, lower){
  const ctx = document.getElementById('chart').getContext('2d');
  if(chart) chart.destroy();
  chart = new Chart(ctx, {
    data: {
      labels,
      datasets: [
        {type:'line', label:'Close', data: close, borderWidth: 2, tension:.25},
        {type:'line', label:'SMA Fast', data: smaFast, borderWidth: 2, tension:.25},
        {type:'line', label:'SMA Slow', data: smaSlow, borderWidth: 2, tension:.25},
        {type:'line', label:'BB Upper', data: upper, borderWidth: 1, borderDash:[6,4], tension:.2},
        {type:'line', label:'BB Lower', data: lower, borderWidth: 1, borderDash:[6,4], tension:.2}
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { labels: { color: '#dce7ff' } } },
      scales: {
        x: { ticks: { color: '#a9b9de', maxRotation: 0, autoSkip: true }, grid: { color: 'rgba(34,48,85,.4)' } },
        y: { ticks: { color: '#a9b9de' }, grid: { color: 'rgba(34,48,85,.4)' } }
      }
    }
  });
}

function buildRows(rows){
  const tbody = document.getElementById('rows');
  tbody.innerHTML = rows.map(r => `
    <tr>
      <td>${r.time}</td>
      <td>${fmt(r.open)}</td>
      <td>${fmt(r.high)}</td>
      <td>${fmt(r.low)}</td>
      <td>${fmt(r.close)}</td>
      <td>${Math.round(r.volume || 0).toLocaleString('en-US')}</td>
    </tr>
  `).join('');
}

async function analyze(){
  const market = document.getElementById('market').value;
  const symbol = document.getElementById('symbol').value.trim();
  const period = document.getElementById('period').value;
  const interval = document.getElementById('interval').value;
  const sma_fast = document.getElementById('sma_fast').value;
  const sma_slow = document.getElementById('sma_slow').value;
  document.getElementById('summary').textContent = 'Đang phân tích...';
  const res = await fetch(`/api/analyze?market=${encodeURIComponent(market)}&symbol=${encodeURIComponent(symbol)}&period=${period}&interval=${interval}&sma_fast=${sma_fast}&sma_slow=${sma_slow}`);
  const data = await res.json();
  if(!data.ok){
    document.getElementById('summary').textContent = data.error || 'Không lấy được dữ liệu';
    setBadge('hold','HOLD');
    return;
  }

  document.getElementById('price').textContent = fmt(data.latest.close, 2);
  document.getElementById('chg').textContent = `${fmt(data.latest.change_pct, 2)}% so với phiên trước`;
  document.getElementById('rsi').textContent = fmt(data.latest.rsi, 2);
  document.getElementById('rsi_sub').textContent = data.latest.rsi_state;
  document.getElementById('trend').textContent = data.trend;
  document.getElementById('trend_sub').textContent = data.trend_detail;
  document.getElementById('rec').textContent = data.signal;
  document.getElementById('rec_sub').textContent = data.signal_reason;
  document.getElementById('summary').textContent = data.summary + (data.market_note ? ' ' + data.market_note : '');
  document.getElementById('sl').textContent = data.risk.stop_loss;
  document.getElementById('tp').textContent = data.risk.take_profit;
  document.getElementById('conf').textContent = data.confidence;
  document.getElementById('bt').textContent = data.backtest.cagr;
  document.getElementById('bt_sub').textContent = data.backtest.detail;

  const kind = data.signal === 'BUY' ? 'buy' : (data.signal === 'SELL' ? 'sell' : 'hold');
  setBadge(kind, data.signal);

  renderChart(data.chart.labels, data.chart.close, data.chart.sma_fast, data.chart.sma_slow, data.chart.bb_upper, data.chart.bb_lower);
  buildRows(data.rows);
}

analyze();
</script>
</body>
</html>
'''


def _to_float(v):
    try:
        if pd.isna(v):
            return np.nan
        return float(v)
    except Exception:
        return np.nan


def normalize_symbol(symbol: str, market: str) -> str:
    symbol = symbol.strip().upper()
    if not symbol:
        return symbol

    market = market.upper().strip()
    if market == 'HK' and '.' not in symbol:
        return f'{symbol}.HK'
    if market == 'JP' and '.' not in symbol:
        return f'{symbol}.T'
    if market == 'UK' and '.' not in symbol:
        return f'{symbol}.L'
    return symbol


def download_data(symbol: str, period: str, interval: str):
    df = yf.download(
        symbol,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False
    )

    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()

    df.columns = [str(c).title() for c in df.columns]

    if 'Datetime' in df.columns and 'Date' not in df.columns:
        df.rename(columns={'Datetime': 'Date'}, inplace=True)

    return df


def add_indicators(df: pd.DataFrame, fast: int = 20, slow: int = 100) -> pd.DataFrame:
    df = df.copy()
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df['Open'] = pd.to_numeric(df['Open'], errors='coerce')
    df['High'] = pd.to_numeric(df['High'], errors='coerce')
    df['Low'] = pd.to_numeric(df['Low'], errors='coerce')
    df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')

    df[f'SMA_{fast}'] = df['Close'].rolling(fast).mean()
    df[f'SMA_{slow}'] = df['Close'].rolling(slow).mean()

    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))

    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_SIGNAL'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_HIST'] = df['MACD'] - df['MACD_SIGNAL']

    mid = df['Close'].rolling(20).mean()
    std = df['Close'].rolling(20).std()
    df['BB_MID'] = mid
    df['BB_UPPER'] = mid + 2 * std
    df['BB_LOWER'] = mid - 2 * std

    return df


def backtest_sma(df: pd.DataFrame, fast: int, slow: int) -> dict:
    if len(df) < max(fast, slow) + 5:
        return {"cagr": "N/A", "detail": "Không đủ dữ liệu để backtest."}

    d = df[['Date', 'Close']].copy()
    d['fast'] = d['Close'].rolling(fast).mean()
    d['slow'] = d['Close'].rolling(slow).mean()
    d = d.dropna().reset_index(drop=True)
    if len(d) < 10:
        return {"cagr": "N/A", "detail": "Không đủ dữ liệu sau khi tính chỉ báo."}

    pos = 0
    entry = 0.0
    equity = 1.0
    trades = 0
    wins = 0
    equity_curve = [equity]

    for i in range(1, len(d)):
        prev = d.iloc[i - 1]
        curr = d.iloc[i]
        buy = prev['fast'] <= prev['slow'] and curr['fast'] > curr['slow']
        sell = prev['fast'] >= prev['slow'] and curr['fast'] < curr['slow']

        price = float(curr['Close'])
        if pos == 0 and buy:
            pos = 1
            entry = price
            trades += 1
        elif pos == 1 and sell:
            ret = (price - entry) / entry
            equity *= (1 + ret)
            wins += 1 if ret > 0 else 0
            pos = 0
            entry = 0.0
        equity_curve.append(equity)

    years = max(len(d) / 252.0, 1e-9)
    cagr = (equity ** (1 / years) - 1) * 100
    win_rate = (wins / trades * 100) if trades else 0.0
    max_dd = 0.0
    peak = equity_curve[0]
    for x in equity_curve:
        peak = max(peak, x)
        dd = (peak - x) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)

    return {
        'cagr': f"{cagr:.2f}%",
        'detail': f"Trades: {trades} | Win rate: {win_rate:.1f}% | Max drawdown: {max_dd*100:.1f}%"
    }


def signal_engine(latest: pd.Series, prev: pd.Series, fast_col: str, slow_col: str) -> tuple[str, str, str, str, str, float, float, float]:
    close = _to_float(latest['Close'])
    sma_fast = _to_float(latest[fast_col])
    sma_slow = _to_float(latest[slow_col])
    rsi = _to_float(latest['RSI'])
    macd = _to_float(latest['MACD'])
    macd_signal = _to_float(latest['MACD_SIGNAL'])
    prev_macd = _to_float(prev['MACD']) if prev is not None else np.nan
    prev_macd_signal = _to_float(prev['MACD_SIGNAL']) if prev is not None else np.nan

    trend = 'Sideways'
    trend_detail = 'Giá chưa tạo xu hướng rõ.'
    if np.isfinite(sma_fast) and np.isfinite(sma_slow):
        if sma_fast > sma_slow:
            trend = 'Uptrend'
            trend_detail = 'SMA nhanh nằm trên SMA chậm.'
        elif sma_fast < sma_slow:
            trend = 'Downtrend'
            trend_detail = 'SMA nhanh nằm dưới SMA chậm.'

    rsi_state = 'Neutral'
    if np.isfinite(rsi):
        if rsi >= 70:
            rsi_state = 'Overbought'
        elif rsi <= 30:
            rsi_state = 'Oversold'
        else:
            rsi_state = 'Neutral'

    cross_up = np.isfinite(prev_macd) and np.isfinite(prev_macd_signal) and np.isfinite(macd) and np.isfinite(macd_signal) and prev_macd <= prev_macd_signal and macd > macd_signal
    cross_down = np.isfinite(prev_macd) and np.isfinite(prev_macd_signal) and np.isfinite(macd) and np.isfinite(macd_signal) and prev_macd >= prev_macd_signal and macd < macd_signal

    signal = 'HOLD'
    reason = 'Chưa có đủ tín hiệu đồng thuận.'
    confidence = 0.0

    bullish = np.isfinite(sma_fast) and np.isfinite(sma_slow) and np.isfinite(rsi) and sma_fast > sma_slow and rsi < 68 and macd > macd_signal
    bearish = np.isfinite(sma_fast) and np.isfinite(sma_slow) and np.isfinite(rsi) and sma_fast < sma_slow and rsi > 32 and macd < macd_signal

    if bullish:
        confidence += 0.35
    if bearish:
        confidence += 0.35
    if cross_up:
        confidence += 0.2
    if cross_down:
        confidence += 0.2
    if np.isfinite(rsi):
        if 40 <= rsi <= 65:
            confidence += 0.15
        elif rsi <= 25 or rsi >= 75:
            confidence -= 0.1

    if bullish and cross_up:
        signal = 'BUY'
        reason = 'SMA xu hướng tăng, MACD cắt lên và RSI còn an toàn.'
        confidence = min(0.95, confidence)
    elif bearish and cross_down:
        signal = 'SELL'
        reason = 'SMA xu hướng giảm, MACD cắt xuống và RSI chưa quá thấp.'
        confidence = min(0.95, confidence)
    elif np.isfinite(rsi) and rsi >= 75:
        signal = 'SELL'
        reason = 'RSI quá cao, rủi ro điều chỉnh tăng.'
        confidence = 0.55
    elif np.isfinite(rsi) and rsi <= 25:
        signal = 'BUY'
        reason = 'RSI rất thấp, có thể xuất hiện nhịp hồi.'
        confidence = 0.55
    elif np.isfinite(sma_fast) and np.isfinite(sma_slow):
        if sma_fast > sma_slow:
            signal = 'BUY'
            reason = 'Xu hướng tăng nhưng cần thêm xác nhận.'
            confidence = 0.45
        elif sma_fast < sma_slow:
            signal = 'SELL'
            reason = 'Xu hướng giảm nhưng cần thêm xác nhận.'
            confidence = 0.45

    if np.isfinite(close):
        atr_like = max(close * 0.03, 0.01)
        stop_loss = close - atr_like * 1.5 if signal == 'BUY' else close + atr_like * 1.5 if signal == 'SELL' else close - atr_like
        take_profit = close + atr_like * 2.5 if signal == 'BUY' else close - atr_like * 2.5 if signal == 'SELL' else close + atr_like * 1.5
    else:
        stop_loss = np.nan
        take_profit = np.nan

    return signal, reason, trend, trend_detail, rsi_state, stop_loss, take_profit, max(0.0, min(0.95, confidence))


@app.route('/')
def index():
    return render_template_string(HTML)


@app.route('/api/analyze')
def analyze():
    symbol = request.args.get('symbol', 'AAPL').strip().upper()
    market = request.args.get('market', 'US').strip().upper()
    period = request.args.get('period', '3mo')
    interval = request.args.get('interval', '1d')
    fast = int(request.args.get('sma_fast', '20'))
    slow = int(request.args.get('sma_slow', '100'))

    if fast >= slow:
        fast = max(5, slow // 2)

    symbol = normalize_symbol(symbol, market)

    try:
        df = download_data(symbol, period, interval)
        if df.empty:
            return jsonify(ok=False, error=f'Không lấy được dữ liệu cho mã {symbol}. Hãy thử mã khác hoặc đổi period/interval.')

        df = add_indicators(df, fast, slow)
        df = df.dropna(subset=['Close']).reset_index(drop=True)
        if len(df) < max(fast, slow, 25):
            return jsonify(ok=False, error='Dữ liệu quá ít để tính chỉ báo. Hãy chọn period lớn hơn.')

        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) >= 2 else latest
        signal, reason, trend, trend_detail, rsi_state, stop_loss, take_profit, confidence = signal_engine(
            latest, prev, f'SMA_{fast}', f'SMA_{slow}'
        )

        prev_close = _to_float(prev['Close'])
        close = _to_float(latest['Close'])
        change_pct = ((close - prev_close) / prev_close * 100) if np.isfinite(close) and np.isfinite(prev_close) and prev_close != 0 else np.nan

        time_col = 'Date' if 'Date' in df.columns else ('Datetime' if 'Datetime' in df.columns else df.columns[0])

        recent = df.tail(18).copy()
        rows = []
        for _, r in recent.iterrows():
            t = r.get(time_col)
            if pd.isna(t):
                ts = ''
            else:
                if isinstance(t, pd.Timestamp):
                    ts = t.strftime('%Y-%m-%d %H:%M') if interval != '1d' else t.strftime('%Y-%m-%d')
                else:
                    ts = str(t)
            rows.append({
                'time': ts,
                'open': _to_float(r.get('Open')),
                'high': _to_float(r.get('High')),
                'low': _to_float(r.get('Low')),
                'close': _to_float(r.get('Close')),
                'volume': _to_float(r.get('Volume')),
            })

        chart_df = df.tail(120).copy()
        chart_labels = []
        for _, r in chart_df.iterrows():
            t = r.get(time_col)
            if pd.isna(t):
                chart_labels.append('')
            else:
                if isinstance(t, pd.Timestamp):
                    chart_labels.append(t.strftime('%m-%d %H:%M') if interval != '1d' else t.strftime('%m-%d'))
                else:
                    chart_labels.append(str(t))

        def arr(col):
            return [None if pd.isna(x) else float(x) for x in chart_df[col].tolist()]

        backtest = backtest_sma(df, fast, slow)

        market_names = {
            'US': 'NYSE / Nasdaq',
            'VN': 'HOSE / HNX / UPCoM',
            'HK': 'HKEX',
            'JP': 'TSE / JPX',
            'UK': 'LSE',
        }
        market_label = market_names.get(market, market)

        summary = (
            f'{symbol} trên {market_label}: giá hiện tại {close:.2f}, xu hướng {trend.lower()}, RSI {rsi_state.lower()}, '
            f'tín hiệu {signal}, độ tin cậy {confidence * 100:.0f}%.'
        )
        if signal == 'BUY':
            market_note = 'Ưu tiên chờ nến xác nhận hoặc retest trước khi vào lệnh.'
        elif signal == 'SELL':
            market_note = 'Nên chờ nhịp hồi kỹ thuật nếu cần thoát lệnh.'
        else:
            market_note = 'Hợp lý nhất là đứng ngoài chờ thêm xác nhận.'
        summary += ' Dùng stop loss và quản trị vốn chặt chẽ.'

        return jsonify(
            ok=True,
            symbol=symbol,
            market=market,
            latest={
                'close': close,
                'change_pct': change_pct,
                'rsi': _to_float(latest['RSI']),
                'rsi_state': rsi_state,
            },
            trend=trend,
            trend_detail=trend_detail,
            signal=signal,
            signal_reason=reason,
            confidence=f'{confidence * 100:.0f}%',
            risk={
                'stop_loss': f'{_to_float(stop_loss):.2f}' if np.isfinite(stop_loss) else '--',
                'take_profit': f'{_to_float(take_profit):.2f}' if np.isfinite(take_profit) else '--',
            },
            backtest=backtest,
            chart={
                'labels': chart_labels,
                'close': arr('Close'),
                'sma_fast': arr(f'SMA_{fast}'),
                'sma_slow': arr(f'SMA_{slow}'),
                'bb_upper': arr('BB_UPPER'),
                'bb_lower': arr('BB_LOWER'),
            },
            rows=rows,
            summary=summary,
            market_note=market_note,
            disclaimer='Đây là công cụ hỗ trợ học tập và phân tích, không phải lời khuyên đầu tư.'
        )
    except Exception as e:
        return jsonify(ok=False, error=f'Lỗi phân tích: {e}')


@app.route('/health')
def health():
    return {'ok': True, 'time': datetime.utcnow().isoformat() + 'Z'}


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
