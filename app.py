"""GeoCell web demo — zero extra dependencies (Python stdlib only).

A single-file HTTP server that exposes the GeoCell engine as a small,
interactive web app: ingest facts with provenance, ask questions and see
the answer's belief status, trust, citation, superseded history, and any
contradictions. This is the "Run" target on Replit — it shows the engine's
differentiators (supersession, citations, contradiction handling) live in
the browser.

Run:
    python app.py            # then open the printed URL (or Replit webview)

The REST API (also usable directly):
    POST /api/ingest   {fact, source, date, authority, confidence}
    POST /api/ask      {query}
    GET  /api/demo     load the demo corpus
    GET  /api/stats
    GET  /api/contradictions
    POST /api/reset
"""
from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geocell import __version__
from geocell.field import GeoCellField
from geocell.demo import load_demo

FIELD = GeoCellField()


def rebuild(field: GeoCellField) -> None:
    """Run the maintenance passes so answers reflect inference + trust."""
    field.hypothesize()
    field.propagate_trust()


PAGE = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>GeoCell Field v__VERSION__</title>
<style>
  :root { --navy:#0b3d5c; --ink:#1a2730; --line:#d6dee4; --bg:#f4f7f9; }
  * { box-sizing:border-box; }
  body { margin:0; font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
         color:var(--ink); background:var(--bg); }
  header { background:var(--navy); color:#fff; padding:16px 22px; }
  header h1 { margin:0; font-size:19px; } header p { margin:4px 0 0; opacity:.85; font-size:13px; }
  .wrap { max-width:1040px; margin:0 auto; padding:18px; display:grid; grid-template-columns:1fr 1fr; gap:18px; }
  .card { background:#fff; border:1px solid var(--line); border-radius:10px; padding:16px; }
  .card h2 { margin:0 0 10px; font-size:14px; color:var(--navy); }
  label { display:block; font-size:12px; font-weight:600; margin:8px 0 3px; }
  input[type=text], textarea { width:100%; padding:8px; border:1px solid var(--line); border-radius:6px; font-size:13px; font-family:inherit; }
  textarea { resize:vertical; min-height:46px; }
  .row { display:flex; gap:10px; } .row > div { flex:1; }
  .sliders { display:flex; gap:14px; font-size:12px; align-items:center; margin-top:8px; }
  button { background:var(--navy); color:#fff; border:0; border-radius:6px; padding:9px 14px; font-size:13px; font-weight:600; cursor:pointer; }
  button.ghost { background:#fff; color:var(--navy); border:1px solid var(--navy); }
  .btns { display:flex; gap:8px; margin-top:12px; flex-wrap:wrap; }
  .answer { margin-top:6px; }
  .badge { display:inline-block; font-size:11px; font-weight:700; padding:2px 8px; border-radius:20px; color:#fff; text-transform:uppercase; letter-spacing:.3px; }
  .active{background:#2e7d32;} .contested{background:#ef6c00;} .superseded{background:#757575;} .retracted{background:#c62828;}
  .ans-text { font-size:15px; margin:8px 0; }
  .meta { font-size:12px; color:#5a6b76; }
  .cite { font-size:12px; color:var(--navy); font-weight:600; }
  ul.ev { list-style:none; padding:0; margin:8px 0 0; font-size:12px; }
  ul.ev li { padding:6px 8px; border-left:3px solid var(--line); margin-bottom:5px; background:#fafbfc; }
  .hist li { border-left-color:#757575; } .conflict li { border-left-color:#ef6c00; }
  .small { font-size:11px; color:#7a8a94; }
  #stats { font-size:12px; color:#5a6b76; padding:6px 0 0; }
  .full { grid-column:1 / -1; }
</style></head>
<body>
<header>
  <h1>GeoCell Field <span style="opacity:.7;font-weight:400">v__VERSION__</span></h1>
  <p>A deterministic, geometry-native memory layer — supersession, trust, citations, contradiction handling. CPU-only, no GPU.</p>
</header>
<div class="wrap">
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
    </div>
    <div id="answer" class="answer"></div>
  </div>

  <div class="card full">
    <h2>Contradictions in the field</h2>
    <div id="contras" class="small">none yet</div>
  </div>
</div>

<script>
async function api(path, body){
  const o = body ? {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)} : {};
  const r = await fetch(path, o); return r.json();
}
function esc(s){return (s||'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
async function refresh(){
  const s = await api('/api/stats');
  document.getElementById('stats').innerHTML =
    `<b>${s.cells}</b> claims · <b>${s.edges}</b> relations · status: ` +
    Object.entries(s.by_status||{}).map(([k,v])=>`${v} ${k}`).join(', ') +
    ` · <b>${s.open_contradictions}</b> open contradictions`;
  const c = await api('/api/contradictions');
  document.getElementById('contras').innerHTML = c.length ?
    '<ul class="ev conflict">'+c.map(x=>`<li>[${x.a_status}] ${esc(x.a)} <span class="small">(${esc(x.a_source)})</span><br>↔ [${x.b_status}] ${esc(x.b)} <span class="small">(${esc(x.b_source)})</span></li>`).join('')+'</ul>'
    : '<span class="small">none yet</span>';
}
async function ingest(){
  const fact=document.getElementById('fact').value.trim(); if(!fact)return;
  await api('/api/ingest',{fact, source:document.getElementById('source').value,
    date:document.getElementById('date').value,
    authority:+document.getElementById('authority').value,
    confidence:+document.getElementById('confidence').value});
  document.getElementById('fact').value=''; refresh();
}
async function demo(){ await api('/api/demo'); refresh(); }
async function reset(){ await api('/api/reset',{}); document.getElementById('answer').innerHTML=''; refresh(); }
async function ask(){
  const q=document.getElementById('q').value.trim(); if(!q)return;
  const a=await api('/api/ask',{query:q});
  const st=a.belief_status||'active';
  let h=`<div class="answer"><span class="badge ${st}">${st}</span>
     <span class="meta">trust ${a.trust??'—'} · confidence ${a.confidence??'—'}</span>
     <div class="ans-text">${esc(a.answer)}</div>`;
  if(a.citation) h+=`<div class="cite">source: ${esc(a.citation)}</div>`;
  if(a.reasoning_chain) h+=`<div class="small">reasoning: ${a.reasoning_chain.map(s=>esc(s.source)).join(' → ')}</div>`;
  if(a.evidence&&a.evidence.length) h+='<ul class="ev">'+a.evidence.slice(0,3).map(e=>`<li>${esc(e.content)} <span class="small">[${esc(e.source)}] res ${e.resonance}</span></li>`).join('')+'</ul>';
  if(a.superseded_history&&a.superseded_history.length) h+='<div class="small" style="margin-top:6px">displaced history:</div><ul class="ev hist">'+a.superseded_history.map(e=>`<li>${esc(e.content)} <span class="small">[${esc(e.source)}]</span></li>`).join('')+'</ul>';
  if(a.open_contradictions&&a.open_contradictions.length) h+='<div class="small" style="margin-top:6px">⚠ disputed in memory</div>';
  h+='</div>'; document.getElementById('answer').innerHTML=h;
}
refresh();
</script>
</body></html>"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _json(self, code, obj):
        self._send(code, json.dumps(obj), "application/json")

    def _body(self):
        n = int(self.headers.get("Content-Length", 0) or 0)
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            return {}

    def log_message(self, *a):  # quiet
        pass

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            return self._send(200, PAGE.replace("__VERSION__", __version__), "text/html; charset=utf-8")
        if self.path == "/health":
            return self._json(200, {"ok": True, "version": __version__})
        if self.path == "/api/demo":
            global FIELD
            FIELD = GeoCellField()
            load_demo(FIELD)
            rebuild(FIELD)
            return self._json(200, FIELD.stats())
        if self.path == "/api/stats":
            return self._json(200, FIELD.stats())
        if self.path == "/api/contradictions":
            return self._json(200, FIELD.contradictions())
        return self._json(404, {"error": "not found"})

    def do_POST(self):
        global FIELD
        body = self._body()
        if self.path == "/api/reset":
            FIELD = GeoCellField()
            return self._json(200, FIELD.stats())
        if self.path == "/api/ingest":
            fact = (body.get("fact") or "").strip()
            if not fact:
                return self._json(400, {"error": "fact required"})
            cid = FIELD.ingest(
                fact,
                source=body.get("source") or "user",
                date=body.get("date") or "",
                authority=float(body.get("authority", 0.5)),
                confidence=float(body.get("confidence", 0.75)),
            )
            rebuild(FIELD)
            return self._json(200, {"id": cid, "stats": FIELD.stats()})
        if self.path == "/api/ask":
            q = (body.get("query") or "").strip()
            if not q:
                return self._json(400, {"error": "query required"})
            ans = FIELD.ask(q)
            ev = ans.get("evidence", [])
            if ev:
                ans["citation"] = FIELD.cells[ev[0]["id"]].source
            return self._json(200, ans)
        return self._json(404, {"error": "not found"})


def main():
    host = "0.0.0.0"
    port = int(os.environ.get("PORT", "8080"))
    srv = ThreadingHTTPServer((host, port), Handler)
    print(f"GeoCell Field v{__version__} running at http://{host}:{port}")
    print("Open the web preview (Replit) or http://localhost:%d locally." % port)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
