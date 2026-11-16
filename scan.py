"""每天收盤後掃描台股全市場，產生手機可看的靜態網頁（site/index.html）

雲端：由 GitHub Actions 每天自動執行（見 daily.yml）
本機試跑：
    python scan.py                              全市場（要幾分鐘）
    SCAN_ONLY="2330 2317 0050" python scan.py   只掃幾檔，快速測試（Windows 用 set SCAN_ONLY=... 再執行）
產出的 site/index.html 是單一檔案，直接用瀏覽器打開就能看。
"""
import io
import json
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path(os.environ.get("SCAN_OUT", "site"))
ONLY = os.environ.get("SCAN_ONLY", "").strip()

KD_N, K_SMOOTH, D_SMOOTH = 9, 3, 3   # KD 參數（9, 3, 3）
MAX_THR = 50          # 網頁滑桿的上限；只收 K、D 都小於此值的股票，檔案才不會太大
CHUNK = 40            # 每批向 Yahoo 要幾檔
MIN_COVERAGE = 0.6    # 取得資料的比例低於這個值就視為失敗，不更新網頁
PLOTLY_URL = "https://cdn.plot.ly/plotly-finance-4.1.2.min.js"

UA = {"User-Agent": "Mozilla/5.0"}
FIELDS = ["Open", "High", "Low", "Close", "Volume"]
# (市場, 證交所/櫃買 OpenAPI, Yahoo 後綴, ISIN 備援網頁的 strMode)
MARKETS = (
    ("上市", "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL", ".TW", "2"),
    ("上櫃", "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_daily_close_quotes", ".TWO", "4"),
)
CODE_KEYS = ("Code", "SecuritiesCompanyCode", "證券代號", "股票代號", "公司代號")
NAME_KEYS = ("Name", "CompanyName", "證券名稱", "股票名稱", "公司簡稱")
MARKET_BY_SUFFIX = {".TW": "上市", ".TWO": "上櫃"}


# ---------------------------------------------------------------- 股票清單
def _pick(rec, keys):
    for k in keys:
        v = rec.get(k)
        if v not in (None, ""):
            return str(v).strip()
    return ""


def classify(code: str):
    """4 碼一般股票、00 開頭 ETF；權證、特別股、債券等其他商品略過。"""
    if re.fullmatch(r"[1-9]\d{3}", code):
        return "股票"
    if re.fullmatch(r"00\d{2,4}[A-Z]?", code):
        return "ETF"
    return None


def fetch_openapi(url):
    import requests
    r = requests.get(url, timeout=30, headers=UA)
    r.raise_for_status()
    rows = [(_pick(rec, CODE_KEYS), _pick(rec, NAME_KEYS)) for rec in r.json()]
    rows = [(c, n) for c, n in rows if c]
    if not rows:
        raise ValueError("回傳內容裡找不到股票代號欄位")
    return rows


def fetch_isin(mode):
    """備援：證交所 ISIN 網頁（Big5 編碼的 HTML 表格，第一欄是「代號　名稱」）。"""
    import requests
    r = requests.get(f"https://isin.twse.com.tw/isin/C_public.jsp?strMode={mode}",
                     timeout=60, headers=UA)
    r.raise_for_status()
    html = r.content.decode("cp950", errors="ignore")
    rows = []
    for table in pd.read_html(io.StringIO(html)):
        for cell in table.iloc[:, 0].astype(str):
            parts = re.split(r"[　\s]+", cell.strip(), maxsplit=1)
            if len(parts) == 2 and classify(parts[0]):
                rows.append((parts[0], parts[1]))
    if not rows:
        raise ValueError("ISIN 網頁解析不到任何股票")
    return rows


def load_universe():
    """回傳 {yahoo 代號: {code, name, market, kind}}。"""
    lists = {}
    for market, url, suffix, mode in MARKETS:
        for label, fn in (("OpenAPI", lambda u=url: fetch_openapi(u)),
                          ("ISIN 備援", lambda m=mode: fetch_isin(m))):
            try:
                lists[market] = (fn(), suffix)
                print(f"{market}清單：{label} 取得 {len(lists[market][0])} 筆", flush=True)
                break
            except Exception as e:  # noqa: BLE001
                print(f"{market}清單：{label} 失敗（{type(e).__name__}: {e}）", flush=True)
    if not lists:
        sys.exit("兩個市場的股票清單都取不到，結束。")

    rows = []
    if ONLY:
        known = {c: (n, s) for items, s in lists.values() for c, n in items}
        for code in dict.fromkeys(re.split(r"[,\s，、;；]+", ONLY.upper())):
            if not code:
                continue
            kind = classify(code) or "股票"
            if code in known:
                rows.append((code, known[code][0], known[code][1], kind))
            else:  # 不知道在哪個市場就兩邊都試，沒資料的那邊會被丟掉
                rows += [(code, "", ".TW", kind), (code, "", ".TWO", kind)]
    else:
        for items, suffix in lists.values():
            for code, name in items:
                kind = classify(code)
                if kind:
                    rows.append((code, name, suffix, kind))

    return {
        f"{c}{s}": dict(code=c, name=n, market=MARKET_BY_SUFFIX[s], kind=k)
        for c, n, s, k in dict.fromkeys(rows)
    }


# ---------------------------------------------------------------- 下載
def split_frames(raw, tickers):
    """把 yfinance 一次下載多檔的結果拆成 {ticker: DataFrame}。"""
    out = {}
    if raw is None or raw.empty:
        return out
    if not isinstance(raw.columns, pd.MultiIndex):
        frames = {tickers[0]: raw}
    else:
        if "Close" in set(raw.columns.get_level_values(0)):  # (欄位, 代號) → (代號, 欄位)
            raw = raw.swaplevel(axis=1).sort_index(axis=1)
        names = set(raw.columns.get_level_values(0))
        frames = {t: raw[t] for t in tickers if t in names}
    for t, df in frames.items():
        if not set(FIELDS) <= set(df.columns):
            continue
        df = df[FIELDS].dropna(subset=["Open", "High", "Low", "Close"])
        df = df.assign(Volume=df["Volume"].fillna(0))
        if len(df) >= 20:
            out[t] = df
    return out


def download_chunk(tickers):
    import yfinance as yf
    for attempt in range(3):
        try:
            raw = yf.download(list(tickers), period="1y", interval="1d", auto_adjust=False,
                              group_by="ticker", threads=4, progress=False)
            return split_frames(raw, tickers), True
        except Exception as e:  # noqa: BLE001
            wait = 5 * 3 ** attempt
            print(f"  下載失敗（{type(e).__name__}），{wait} 秒後重試", flush=True)
            time.sleep(wait)
    return {}, False


def scan_all(tickers):
    chunks = [tuple(tickers[i:i + CHUNK]) for i in range(0, len(tickers), CHUNK)]
    frames, weak = {}, []
    for n, ch in enumerate(chunks, 1):
        res, ok = download_chunk(ch)
        frames.update(res)
        if not ok or len(res) < 0.3 * len(ch):
            weak.append(ch)
        print(f"[{n}/{len(chunks)}] {len(res)}/{len(ch)} 檔有資料", flush=True)
        time.sleep(1.5)
    if weak and not ONLY:
        print(f"有 {len(weak)} 批資料偏少，等 30 秒後重試一次…", flush=True)
        time.sleep(30)
        for ch in weak:
            res, _ = download_chunk(ch)
            frames.update(res)
            time.sleep(2)
    return frames


# ---------------------------------------------------------------- 指標與資料打包
def kd_arrays(df, n=KD_N, k_smooth=K_SMOOTH, d_smooth=D_SMOOTH):
    """台灣看盤軟體常見的 KD：RSV 後 K、D 初始值 50，遞迴平滑。"""
    low_n = df["Low"].rolling(n).min()
    high_n = df["High"].rolling(n).max()
    rng = (high_n - low_n).replace(0, np.nan)
    rsv = ((df["Close"] - low_n) / rng * 100).fillna(50).to_numpy()
    k_arr, d_arr = np.empty(len(rsv)), np.empty(len(rsv))
    k = d = 50.0
    for i, r in enumerate(rsv):
        k = (k * (k_smooth - 1) + r) / k_smooth
        d = (d * (d_smooth - 1) + k) / d_smooth
        k_arr[i], d_arr[i] = k, d
    return k_arr, d_arr


def build_data(frames, meta, generated, keep_all=False):
    """只收 K、D 都小於 MAX_THR 的股票（網頁最大門檻），並把歷史資料打包成精簡 JSON。"""
    newest = max(df.index[-1] for df in frames.values())
    recs = []
    for t, df in frames.items():
        if (newest - df.index[-1]).days > 7:  # 久未交易（停牌、下市）的不納入
            continue
        k, d = kd_arrays(df)
        recs.append((t, df, k, d))
    fresh = len(recs)
    if not keep_all:
        recs = [r for r in recs if r[2][-1] < MAX_THR and r[3][-1] < MAX_THR]

    dates = sorted({ix.strftime("%Y-%m-%d") for _, df, _, _ in recs for ix in df.index})
    pos = {d: i for i, d in enumerate(dates)}
    stocks = []
    for t, df, k, d in sorted(recs, key=lambda r: meta[r[0]]["code"]):
        m = meta[t]
        stocks.append({
            "c": m["code"], "n": m["name"], "m": m["market"], "t": m["kind"],
            "di": [pos[ix.strftime("%Y-%m-%d")] for ix in df.index],
            "o": df["Open"].round(2).tolist(), "h": df["High"].round(2).tolist(),
            "l": df["Low"].round(2).tolist(), "cl": df["Close"].round(2).tolist(),
            "v": (df["Volume"] / 1000).round().astype(int).tolist(),
            "K": np.round(k, 2).tolist(), "D": np.round(d, 2).tolist(),
        })
    return {
        "asof": newest.strftime("%Y-%m-%d"), "generated": generated,
        "scanned": fresh, "maxThr": MAX_THR, "dates": dates, "stocks": stocks,
    }


def render_html(data):
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    payload = payload.replace("</", "<\\/")
    return TEMPLATE.replace("__PLOTLY__", PLOTLY_URL).replace("__DATA__", payload)


# ---------------------------------------------------------------- 主程式
def main():
    meta = load_universe()
    print(f"待掃描 {len(meta)} 檔", flush=True)
    frames = scan_all(list(meta))
    print(f"取得資料 {len(frames)} 檔", flush=True)
    if not frames:
        sys.exit("完全沒有取得資料，不更新網頁。")
    if not ONLY and len(frames) < MIN_COVERAGE * len(meta):
        sys.exit(f"只取得 {len(frames)}/{len(meta)} 檔（低於 {MIN_COVERAGE:.0%}），"
                 "可能被 Yahoo 限流，不更新網頁，沿用上一次的版本。")

    generated = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M")
    data = build_data(frames, meta, generated, keep_all=bool(ONLY))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "index.html"
    out.write_text(render_html(data), encoding="utf-8")
    print(f"資料日 {data['asof']}，有資料 {data['scanned']} 檔，"
          f"收錄 K、D 都小於 {MAX_THR} 的 {len(data['stocks'])} 檔 → {out}"
          f"（{out.stat().st_size / 1e6:.1f} MB）", flush=True)


# ---------------------------------------------------------------- 網頁模板
TEMPLATE = r"""<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>台股 KD 掃描</title>
<style>
:root{--bg:#fff;--fg:#1d1d1f;--mut:#6e6e73;--card:#f4f4f6;--line:#e2e2e7;--acc:#1e88e5;--up:#e53935;--down:#2e9d4a}
@media (prefers-color-scheme:dark){:root{--bg:#121214;--fg:#f2f2f4;--mut:#9a9aa2;--card:#1d1d21;--line:#2c2c32;--acc:#4a9ff0;--up:#ef5350;--down:#4cc16a}}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang TC","Noto Sans TC","Microsoft JhengHei",sans-serif}
main{max-width:860px;margin:0 auto;padding:14px 14px 40px}
h1{font-size:20px;margin:4px 0 2px}
h2{font-size:18px;margin:0}
.mut{color:var(--mut);font-size:13px}
.card{background:var(--card);border-radius:12px;padding:12px 14px;margin:12px 0}
.row{display:flex;align-items:center;justify-content:space-between;gap:12px;margin:8px 0}
.row input[type=range]{flex:1;max-width:55%}
.row input[type=number]{width:100px;padding:6px 8px;font-size:16px;border-radius:8px;border:1px solid var(--line);background:var(--bg);color:var(--fg)}
.chk{display:inline-flex;align-items:center;gap:6px;margin:6px 16px 2px 0}
.tw{overflow-x:auto;border:1px solid var(--line);border-radius:12px}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{padding:8px 10px;text-align:right;white-space:nowrap;border-bottom:1px solid var(--line)}
th:first-child,td:first-child{text-align:left}
th{font-size:12px;color:var(--mut);font-weight:600}
tbody tr{cursor:pointer}
tbody tr:last-child td{border-bottom:0}
tbody tr.sel{background:var(--card)}
.up{color:var(--up)}.down{color:var(--down)}
.tag{font-size:11px;background:var(--up);color:#fff;border-radius:4px;padding:0 4px;vertical-align:1px}
td.empty{text-align:center;color:var(--mut);padding:24px 10px;white-space:normal}
[hidden]{display:none!important}
.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin:10px 0}
.mc{background:var(--bg);border-radius:10px;padding:8px 10px}
.mc .lb{display:block;font-size:12px;color:var(--mut)}
.mc b{font-size:17px}
.mc .sub{display:block;font-size:12px}
.btns{display:flex;flex-wrap:wrap;gap:6px;margin:8px 0}
.btns button{border:1px solid var(--line);background:var(--bg);color:var(--fg);border-radius:16px;padding:4px 10px;font-size:13px}
.btns button.on{background:var(--acc);border-color:var(--acc);color:#fff}
#chart{width:100%;min-height:200px}
footer{margin-top:24px}
</style>
</head>
<body>
<main>
<header>
  <h1>台股 KD 掃描</h1>
  <div class="mut" id="meta"></div>
</header>

<section class="card">
  <label class="row"><span>K 和 D 都小於 <b id="thrv"></b></span><input type="range" id="thr" min="5" max="50" step="1"></label>
  <label class="row"><span>5 日均量下限（張）</span><input type="number" id="vol" min="0" step="100" inputmode="numeric"></label>
  <label class="chk"><input type="checkbox" id="golden"> 只看 K ≥ D</label>
  <label class="chk"><input type="checkbox" id="etf"> 含 ETF</label>
</section>

<p class="mut">符合 <b id="count">0</b> 檔，依 K 值由小到大。點一列查看均線與價格。</p>
<div class="tw">
  <table>
    <thead><tr><th>股票</th><th>收盤</th><th>K</th><th>D</th><th>距月線</th><th>距季線</th></tr></thead>
    <tbody id="rows"></tbody>
  </table>
</div>

<section class="card" id="detail" hidden>
  <h2 id="dtitle"></h2>
  <div class="mut" id="dsub"></div>
  <div class="grid" id="metrics"></div>
  <div class="btns" id="chips"></div>
  <div class="btns" id="ranges"></div>
  <div id="chart"></div>
</section>

<footer class="mut">
  資料來源：Yahoo Finance，每個交易日收盤後自動更新。KD 參數 9、3、3，K 線紅漲綠跌。僅供觀察參考，非投資建議。
</footer>
</main>

<script src="__PLOTLY__" charset="utf-8"></script>
<script>
(function () {
  "use strict";
  var DATA = __DATA__;
  var MA_NAMES = {5: "週線", 10: "雙週線", 20: "月線", 60: "季線", 120: "半年線", 240: "年線"};
  var MA_COLORS = {5: "#fb8c00", 10: "#8e24aa", 20: "#1e88e5", 60: "#00897b", 120: "#6d4c41", 240: "#546e7a"};
  var MA_LIST = [5, 10, 20, 60, 120, 240];
  var RANGES = [["1月", 22], ["3月", 66], ["6月", 132], ["1年", 250]];
  var UP = "#e53935", DOWN = "#43a047";
  var state = {thr: 25, minVol: 500, golden: false, etf: false, sel: -1, mas: [5, 20, 60], range: 2};

  function $(id) { return document.getElementById(id); }
  function esc(s) {
    return String(s).replace(/[&<>"]/g, function (c) { return {"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]; });
  }
  function fmt(x, n) { return (x === null || x === undefined || isNaN(x)) ? "—" : x.toFixed(n); }
  function sgn(x, n) { return (x === null || x === undefined || isNaN(x)) ? "—" : (x > 0 ? "+" : "") + x.toFixed(n); }
  function cls(x) { return x > 0 ? "up" : (x < 0 ? "down" : ""); }

  // ---- 設定記憶（存在這支手機的瀏覽器裡）
  function saveSettings() {
    try {
      localStorage.setItem("kdscan", JSON.stringify({thr: state.thr, minVol: state.minVol, golden: state.golden,
        etf: state.etf, mas: state.mas, range: state.range}));
    } catch (e) {}
  }
  function loadSettings() {
    try {
      var s = JSON.parse(localStorage.getItem("kdscan") || "null");
      if (!s) return;
      if (typeof s.thr === "number" && s.thr >= 5 && s.thr <= DATA.maxThr) state.thr = s.thr;
      if (typeof s.minVol === "number" && s.minVol >= 0) state.minVol = s.minVol;
      state.golden = !!s.golden; state.etf = !!s.etf;
      if (Array.isArray(s.mas)) state.mas = s.mas.filter(function (w) { return MA_LIST.indexOf(w) >= 0; });
      if (typeof s.range === "number" && s.range >= 0 && s.range < RANGES.length) state.range = s.range;
    } catch (e) {}
  }

  // ---- 計算
  function maSeries(arr, w) {
    var out = new Array(arr.length), t = 0;
    for (var i = 0; i < arr.length; i++) {
      t += arr[i];
      if (i >= w) t -= arr[i - w];
      out[i] = i >= w - 1 ? t / w : null;
    }
    return out;
  }
  function maLast(arr, w) {
    var n = arr.length;
    if (n < w) return null;
    var t = 0;
    for (var i = n - w; i < n; i++) t += arr[i];
    return t / w;
  }
  var S = DATA.stocks.map(function (s) {
    var n = s.cl.length, c = s.cl[n - 1], pc = s.cl[n - 2];
    var m20 = maLast(s.cl, 20), m60 = maLast(s.cl, 60);
    var k = s.K[n - 1], d = s.D[n - 1];
    return {
      c: c, chg: (c / pc - 1) * 100, k: k, d: d,
      golden: k > d && s.K[n - 2] <= s.D[n - 2],
      b20: m20 ? (c / m20 - 1) * 100 : null, b60: m60 ? (c / m60 - 1) * 100 : null,
      vol: maLast(s.v, 5)
    };
  });

  function visible() {
    var out = [];
    for (var i = 0; i < S.length; i++) {
      var x = S[i];
      if (!(x.k < state.thr && x.d < state.thr)) continue;
      if (x.vol < state.minVol) continue;
      if (state.golden && x.k < x.d) continue;
      if (!state.etf && DATA.stocks[i].t === "ETF") continue;
      out.push(i);
    }
    out.sort(function (a, b) { return S[a].k - S[b].k; });
    return out;
  }

  // ---- 畫面
  function render() {
    var idx = visible();
    $("count").textContent = idx.length;
    var h = idx.map(function (i) {
      var s = DATA.stocks[i], x = S[i];
      return '<tr data-i="' + i + '"' + (i === state.sel ? ' class="sel"' : "") + ">" +
        '<td><b>' + esc(s.c) + "</b>" + (x.golden ? ' <span class="tag">金叉</span>' : "") +
        '<br><span class="mut">' + esc(s.n) + "</span></td>" +
        "<td>" + fmt(x.c, 2) + '<br><span class="' + cls(x.chg) + '">' + sgn(x.chg, 2) + "%</span></td>" +
        "<td><b>" + fmt(x.k, 1) + "</b></td><td>" + fmt(x.d, 1) + "</td>" +
        '<td class="' + cls(x.b20) + '">' + sgn(x.b20, 1) + "%</td>" +
        '<td class="' + cls(x.b60) + '">' + sgn(x.b60, 1) + "%</td></tr>";
    }).join("");
    $("rows").innerHTML = h ||
      '<tr><td colspan="6" class="empty">目前沒有符合條件的股票，可以調高門檻或降低均量下限。</td></tr>';
  }

  function drawChips() {
    var h = "";
    MA_LIST.forEach(function (w) {
      h += '<button data-ma="' + w + '"' + (state.mas.indexOf(w) >= 0 ? ' class="on"' : "") + ">" + w + "MA " + MA_NAMES[w] + "</button>";
    });
    $("chips").innerHTML = h;
    var r = "";
    RANGES.forEach(function (x, i) {
      r += '<button data-r="' + i + '"' + (i === state.range ? ' class="on"' : "") + ">" + x[0] + "</button>";
    });
    $("ranges").innerHTML = r;
  }

  function drawMetrics() {
    var s = DATA.stocks[state.sel], x = S[state.sel], n = s.cl.length;
    var diff = s.cl[n - 1] - s.cl[n - 2];
    var h = '<div class="mc"><span class="lb">收盤</span><b class="' + cls(diff) + '">' + fmt(x.c, 2) + "</b>" +
      '<span class="sub ' + cls(diff) + '">' + sgn(diff, 2) + " (" + sgn(x.chg, 2) + "%)</span></div>" +
      '<div class="mc"><span class="lb">K</span><b>' + fmt(x.k, 1) + "</b></div>" +
      '<div class="mc"><span class="lb">D</span><b>' + fmt(x.d, 1) + "</b></div>";
    state.mas.forEach(function (w) {
      var v = maLast(s.cl, w);
      h += '<div class="mc"><span class="lb">' + w + "MA " + MA_NAMES[w] + "</span><b>" + fmt(v, 2) + "</b>" +
        '<span class="sub ' + (v ? cls(x.c - v) : "") + '">' + (v ? "價格 " + sgn((x.c / v - 1) * 100, 1) + "%" : "資料不足") + "</span></div>";
    });
    $("metrics").innerHTML = h;
  }

  function drawChart() {
    if (state.sel < 0) return;
    var s = DATA.stocks[state.sel], n = s.cl.length;
    if (typeof Plotly === "undefined") {
      $("chart").textContent = "圖表元件載入失敗，請確認網路後重新整理。";
      return;
    }
    var N = Math.min(RANGES[state.range][1], n), a = n - N;
    var x = s.di.slice(a).map(function (j) { return DATA.dates[j]; });
    var cs = getComputedStyle(document.documentElement);
    var fg = cs.getPropertyValue("--fg").trim() || "#222";
    var line = cs.getPropertyValue("--line").trim() || "#ddd";

    var traces = [{
      type: "candlestick", x: x, open: s.o.slice(a), high: s.h.slice(a), low: s.l.slice(a), close: s.cl.slice(a),
      increasing: {line: {color: UP}, fillcolor: UP}, decreasing: {line: {color: DOWN}, fillcolor: DOWN},
      name: "K線", showlegend: false, xaxis: "x", yaxis: "y"
    }];
    state.mas.forEach(function (w) {
      traces.push({type: "scatter", mode: "lines", x: x, y: maSeries(s.cl, w).slice(a), name: w + "MA",
        line: {width: 1.5, color: MA_COLORS[w]}, xaxis: "x", yaxis: "y"});
    });
    var vcol = [];
    for (var i = a; i < n; i++) vcol.push(s.cl[i] >= s.o[i] ? UP : DOWN);
    traces.push({type: "bar", x: x, y: s.v.slice(a), marker: {color: vcol}, name: "成交量(張)",
      showlegend: false, xaxis: "x", yaxis: "y2"});
    traces.push({type: "scatter", mode: "lines", x: x, y: s.K.slice(a), name: "K",
      line: {width: 1.5, color: "#fb8c00"}, xaxis: "x", yaxis: "y3"});
    traces.push({type: "scatter", mode: "lines", x: x, y: s.D.slice(a), name: "D",
      line: {width: 1.5, color: "#1e88e5"}, xaxis: "x", yaxis: "y3"});

    var layout = {
      height: window.innerWidth < 600 ? 520 : 680,
      margin: {l: 44, r: 8, t: 8, b: 28},
      legend: {orientation: "h", x: 0, y: 1.07, font: {size: 11}},
      paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
      font: {color: fg, size: 11}, hovermode: "x unified", dragmode: false,
      xaxis: {type: "category", anchor: "y3", rangeslider: {visible: false}, nticks: 6, showgrid: false, fixedrange: true},
      yaxis: {domain: [0.40, 1], gridcolor: line, fixedrange: true},
      yaxis2: {domain: [0.22, 0.36], showgrid: false, fixedrange: true},
      yaxis3: {domain: [0, 0.18], range: [0, 100], gridcolor: line, fixedrange: true},
      shapes: [state.thr, 100 - state.thr].map(function (v) {
        return {type: "line", xref: "paper", x0: 0, x1: 1, yref: "y3", y0: v, y1: v,
          line: {color: "gray", width: 1, dash: "dot"}};
      })
    };
    Plotly.newPlot("chart", traces, layout, {displayModeBar: false, responsive: true});
  }

  function showDetail(i) {
    state.sel = i;
    var s = DATA.stocks[i], n = s.cl.length;
    $("detail").hidden = false;
    $("dtitle").textContent = s.c + " " + s.n;
    $("dsub").textContent = s.m + (s.t === "ETF" ? " ETF" : "") + "　" + DATA.dates[s.di[n - 1]] +
      "　開 " + fmt(s.o[n - 1], 2) + "　高 " + fmt(s.h[n - 1], 2) + "　低 " + fmt(s.l[n - 1], 2) +
      "　收 " + fmt(s.cl[n - 1], 2) + "　量 " + s.v[n - 1].toLocaleString() + " 張";
    drawChips();
    drawMetrics();
    drawChart();
    render();
    $("detail").scrollIntoView({behavior: "smooth", block: "start"});
  }

  // ---- 事件
  function onFilter() {
    state.thr = +$("thr").value;
    state.minVol = Math.max(0, +$("vol").value || 0);
    state.golden = $("golden").checked;
    state.etf = $("etf").checked;
    $("thrv").textContent = state.thr;
    saveSettings();
    render();
    if (state.sel >= 0) drawChart();  // 門檻線跟著動
  }
  $("thr").addEventListener("input", onFilter);
  $("vol").addEventListener("change", onFilter);
  $("golden").addEventListener("change", onFilter);
  $("etf").addEventListener("change", onFilter);

  $("rows").addEventListener("click", function (e) {
    var tr = e.target && e.target.closest ? e.target.closest("tr") : null;
    if (tr && tr.dataset && tr.dataset.i !== undefined) showDetail(+tr.dataset.i);
  });
  $("chips").addEventListener("click", function (e) {
    var b = e.target && e.target.closest ? e.target.closest("button") : null;
    if (!b || !b.dataset || b.dataset.ma === undefined) return;
    var w = +b.dataset.ma, at = state.mas.indexOf(w);
    if (at >= 0) state.mas.splice(at, 1); else state.mas.push(w);
    state.mas.sort(function (p, q) { return p - q; });
    saveSettings(); drawChips(); drawMetrics(); drawChart();
  });
  $("ranges").addEventListener("click", function (e) {
    var b = e.target && e.target.closest ? e.target.closest("button") : null;
    if (!b || !b.dataset || b.dataset.r === undefined) return;
    state.range = +b.dataset.r;
    saveSettings(); drawChips(); drawChart();
  });

  // ---- 初始化
  loadSettings();
  $("thr").max = DATA.maxThr;
  $("thr").value = state.thr;
  $("vol").value = state.minVol;
  $("golden").checked = state.golden;
  $("etf").checked = state.etf;
  $("thrv").textContent = state.thr;
  $("meta").textContent = "資料日 " + DATA.asof + "　更新於 " + DATA.generated + "　共 " + DATA.scanned + " 檔有資料";
  render();
  window.__app = {state: state, S: S, visible: visible, render: render, showDetail: showDetail, maSeries: maSeries};
})();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
