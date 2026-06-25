"""Build a single self-contained HTML file that runs GeoCell in the browser.

Bundles the real `geocell/` package (zipped + base64) into one HTML file
that loads Pyodide (Python + numpy + networkx in WebAssembly), unpacks the
package into the in-browser filesystem, and serves the interactive demo --
no server, no install. Open the file in any modern browser and use it.

    python tools/build_browser.py            # -> dist/geocell-browser.html
"""
from __future__ import annotations

import base64
import io
import os
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(ROOT, "geocell")
OUT = os.path.join(ROOT, "dist", "geocell-browser.html")


def zip_package() -> str:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for dirpath, _dirs, files in os.walk(PKG):
            for fn in files:
                if not fn.endswith(".py"):
                    continue
                full = os.path.join(dirpath, fn)
                arc = os.path.relpath(full, ROOT)   # keep geocell/ prefix
                z.write(full, arc)
    return base64.b64encode(buf.getvalue()).decode("ascii")


HTML = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>GeoCell Field — in-browser</title>
<style>
  :root { --navy:#0b3d5c; --ink:#1a2730; --line:#d6dee4; --bg:#f4f7f9; }
  *{box-sizing:border-box;} body{margin:0;font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:var(--ink);background:var(--bg);}
  header{background:var(--navy);color:#fff;padding:16px 22px;} header h1{margin:0;font-size:19px;} header p{margin:4px 0 0;opacity:.85;font-size:13px;}
  #boot{padding:14px 22px;background:#fff8e1;border-bottom:1px solid #ffe082;font-size:13px;color:#7a5c00;}
  .wrap{max-width:1040px;margin:0 auto;padding:18px;display:grid;grid-template-columns:1fr 1fr;gap:18px;}
  .card{background:#fff;border:1px solid var(--line);border-radius:10px;padding:16px;}
  .card h2{margin:0 0 10px;font-size:14px;color:var(--navy);}
  label{display:block;font-size:12px;font-weight:600;margin:8px 0 3px;}
  input[type=text],textarea{width:100%;padding:8px;border:1px solid var(--line);border-radius:6px;font-size:13px;font-family:inherit;}
  textarea{resize:vertical;min-height:46px;}
  .row{display:flex;gap:10px;} .row>div{flex:1;}
  .sliders{display:flex;gap:14px;font-size:12px;align-items:center;margin-top:8px;}
  button{background:var(--navy);color:#fff;border:0;border-radius:6px;padding:9px 14px;font-size:13px;font-weight:600;cursor:pointer;}
  button:disabled{opacity:.5;cursor:wait;} button.ghost{background:#fff;color:var(--navy);border:1px solid var(--navy);}
  .btns{display:flex;gap:8px;margin-top:12px;flex-wrap:wrap;}
  .badge{display:inline-block;font-size:11px;font-weight:700;padding:2px 8px;border-radius:20px;color:#fff;text-transform:uppercase;letter-spacing:.3px;}
  .active{background:#2e7d32;}.contested{background:#ef6c00;}.superseded{background:#757575;}.retracted{background:#c62828;}
  .ans-text{font-size:15px;margin:8px 0;} .meta{font-size:12px;color:#5a6b76;} .cite{font-size:12px;color:var(--navy);font-weight:600;}
  ul.ev{list-style:none;padding:0;margin:8px 0 0;font-size:12px;} ul.ev li{padding:6px 8px;border-left:3px solid var(--line);margin-bottom:5px;background:#fafbfc;}
  .hist li{border-left-color:#757575;} .conflict li{border-left-color:#ef6c00;}
  .small{font-size:11px;color:#7a8a94;} #stats{font-size:12px;color:#5a6b76;padding:6px 0 0;} .full{grid-column:1/-1;}
</style></head>
<body>
<header>
  <h1>GeoCell Field <span style="opacity:.7;font-weight:400">· runs entirely in your browser</span></h1>
  <p>Deterministic geometry-native memory — supersession, trust, citations, contradiction handling. No server, no install, no GPU.</p>
</header>
<div id="boot">Loading the GeoCell engine (Python + numpy + networkx via WebAssembly)… first load takes ~15–30s.</div>
<div class="wrap" id="ui" style="display:none">
  <div class="card">
    <h2>1 · Add a memory</h2>
    <label>Fact</label>
    <textarea id="fact" placeholder="e.g. Project Orion budget was revised to $2.7 million."></textarea>
    <div class="row">
      <div><label>Source</label><input id="source" type="text" value="user"></div>
      <div><label>Date (YYYY-MM-DD)</label><input id="date" type="text" placeholder="2026-02-01"></div>
    </div>
    <div class="sliders">
      <span>authority <b id="aL">0.50</b></span><input id="authority" type="range" min="0" max="1" step="0.05" value="0.5" oninput="aL.textContent=(+this.value).toFixed(2)">
      <span>confidence <b id="cL">0.75</b></span><input id="confidence" type="range" min="0" max="1" step="0.05" value="0.75" oninput="cL.textContent=(+this.value).toFixed(2)">
    </div>
    <div class="btns">
      <button onclick="ingest()">Ingest</button>
      <button class="ghost" onclick="demo()">Load demo corpus</button>
      <button class="ghost" onclick="reset()">Reset</button>
    </div>
    <div id="stats">field empty</div>
  </div>
  <div class="card">
    <h2>2 · Ask</h2>
    <label>Question</label>
    <textarea id="q" placeholder="e.g. What is the current Project Orion budget?"></textarea>
    <div class="btns">
      <button onclick="ask()">Ask</button>
      <button class="ghost" onclick="document.getElementById('q').value='What is the current Project Orion budget?';ask()">Try: current budget</button>
      <button class="ghost" onclick="document.getElementById('q').value='What did the old memo say?';ask()">Try: old memo</button>
    </div>
    <div id="answer"></div>
  </div>
  <div class="card full">
    <h2>Contradictions in the field</h2>
    <div id="contras" class="small">none yet</div>
  </div>
</div>

<script src="https://cdn.jsdelivr.net/pyodide/v0.26.2/full/pyodide.js"></script>
<script>
const ZIP_B64 = "__ZIP_B64__";
let pyodide, callPy;

function esc(s){return (s||'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
function call(method, args){ return JSON.parse(callPy(method, JSON.stringify(args||{}))); }

async function boot(){
  pyodide = await loadPyodide();
  await pyodide.loadPackage(["numpy","networkx"]);
  pyodide.globals.set("ZIP_B64", ZIP_B64);
  await pyodide.runPythonAsync(`
import base64, io, zipfile, sys, json
data = base64.b64decode(ZIP_B64)
zipfile.ZipFile(io.BytesIO(data)).extractall("/geocell_pkg")
sys.path.insert(0, "/geocell_pkg")
from geocell.field import GeoCellField
from geocell.demo import load_demo
state = {"f": GeoCellField()}
def _rebuild(f):
    f.hypothesize(); f.propagate_trust()
def call(method, args_json):
    args = json.loads(args_json or "{}"); f = state["f"]
    if method == "demo":
        state["f"] = GeoCellField(); load_demo(state["f"]); _rebuild(state["f"]); return json.dumps(state["f"].stats())
    if method == "reset":
        state["f"] = GeoCellField(); return json.dumps(state["f"].stats())
    if method == "stats":
        return json.dumps(f.stats())
    if method == "contradictions":
        return json.dumps(f.contradictions())
    if method == "ingest":
        cid = f.ingest(args.get("fact","").strip(), source=args.get("source") or "user",
                       date=args.get("date") or "", authority=float(args.get("authority",0.5)),
                       confidence=float(args.get("confidence",0.75)))
        _rebuild(f); return json.dumps({"id": cid, "stats": f.stats()})
    if method == "ask":
        a = f.ask(args.get("query","").strip()); ev = a.get("evidence", [])
        if ev: a["citation"] = f.cells[ev[0]["id"]].source
        return json.dumps(a)
    return json.dumps({"error": "unknown method"})
`);
  callPy = pyodide.globals.get("call");
  document.getElementById("boot").style.display = "none";
  document.getElementById("ui").style.display = "grid";
  refresh();
}

function refresh(){
  const s = call("stats");
  document.getElementById("stats").innerHTML =
    `<b>${s.cells}</b> claims · <b>${s.edges}</b> relations · status: ` +
    Object.entries(s.by_status||{}).map(([k,v])=>`${v} ${k}`).join(", ") +
    ` · <b>${s.open_contradictions}</b> open contradictions`;
  const c = call("contradictions");
  document.getElementById("contras").innerHTML = c.length ?
    '<ul class="ev conflict">'+c.map(x=>`<li>[${x.a_status}] ${esc(x.a)} <span class="small">(${esc(x.a_source)})</span><br>↔ [${x.b_status}] ${esc(x.b)} <span class="small">(${esc(x.b_source)})</span></li>`).join('')+'</ul>'
    : '<span class="small">none yet</span>';
}
function demo(){ call("demo"); refresh(); }
function reset(){ call("reset"); document.getElementById("answer").innerHTML=""; refresh(); }
function ingest(){
  const fact=document.getElementById("fact").value.trim(); if(!fact)return;
  call("ingest",{fact, source:document.getElementById("source").value, date:document.getElementById("date").value,
    authority:+document.getElementById("authority").value, confidence:+document.getElementById("confidence").value});
  document.getElementById("fact").value=""; refresh();
}
function ask(){
  const q=document.getElementById("q").value.trim(); if(!q)return;
  const a=call("ask",{query:q}); const st=a.belief_status||"active";
  let h=`<span class="badge ${st}">${st}</span> <span class="meta">trust ${a.trust??'—'} · confidence ${a.confidence??'—'}</span><div class="ans-text">${esc(a.answer)}</div>`;
  if(a.citation) h+=`<div class="cite">source: ${esc(a.citation)}</div>`;
  if(a.reasoning_chain) h+=`<div class="small">reasoning: ${a.reasoning_chain.map(s=>esc(s.source)).join(' → ')}</div>`;
  if(a.evidence&&a.evidence.length) h+='<ul class="ev">'+a.evidence.slice(0,3).map(e=>`<li>${esc(e.content)} <span class="small">[${esc(e.source)}] res ${e.resonance}</span></li>`).join('')+'</ul>';
  if(a.superseded_history&&a.superseded_history.length) h+='<div class="small" style="margin-top:6px">displaced history:</div><ul class="ev hist">'+a.superseded_history.map(e=>`<li>${esc(e.content)} <span class="small">[${esc(e.source)}]</span></li>`).join('')+'</ul>';
  if(a.open_contradictions&&a.open_contradictions.length) h+='<div class="small" style="margin-top:6px">⚠ disputed in memory</div>';
  document.getElementById("answer").innerHTML=h;
}
boot().catch(e=>{document.getElementById("boot").textContent="Failed to load engine: "+e;});
</script>
</body></html>"""


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    html = HTML.replace("__ZIP_B64__", zip_package())
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"wrote {OUT} ({os.path.getsize(OUT)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
