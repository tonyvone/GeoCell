"""Interactive REPL for the GeoCell field."""
from __future__ import annotations

import json
from typing import Any, Dict, Tuple

from geocell import __version__
from geocell.demo import load_demo, run_benchmark
from geocell.field import GeoCellField
from geocell.synthesis import brief

HELP = """Commands:
  demo                          load the demo corpus
  ingest <fact> | source=<s> | date=<YYYY-MM-DD> | confidence=<0-1> | authority=<0-1>
  recall <query>                spreading-activation retrieval
  ask <query>                   answer with belief status, trust, evidence
  contradictions [query]        all conflicts (resolved + open)
  timeline <subject>            evolution of belief about a subject
  path <start> => <end>         trust-weighted route between two memories
  inspect <id>                  cell + neighborhood
  why <id>                      epistemic ledger + trust breakdown
  brief <query>                 synthesized multi-fact answer with citations (no LLM)
  quantize                      switch to the 96-bytes-per-memory serving index
  infer                         run defeasible inference (writes hypotheses)
  settle                        relax the field geometry
  trust                         propagate trust to a fixed point
  sleep                         consolidate clusters into concept cells
  stats                         field overview
  benchmark                     fresh-field QA + structural self-test
  save <path> / load <path>     persistence (full float geometry)
  savec <path> / loadc <path>   compact binary snapshot (sign signatures)
  help / quit
"""


def parse_meta(line: str) -> Tuple[str, Dict[str, Any]]:
    parts = [p.strip() for p in line.split("|")]
    content = parts[0].replace("ingest", "", 1).strip()
    meta: Dict[str, Any] = {"source": "user", "date": "", "confidence": 0.75, "authority": 0.5}
    for part in parts[1:]:
        if "=" not in part:
            continue
        k, v = [x.strip() for x in part.split("=", 1)]
        if k in {"confidence", "authority"}:
            meta[k] = float(v)
        elif k in {"source", "date"}:
            meta[k] = v
    return content, meta


def print_json(obj: Any) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False))


def cli() -> None:
    mem = GeoCellField()
    print(f"GeoCell Field v{__version__} ready. Type 'help' for commands.")
    while True:
        try:
            line = input("geocell> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            return
        if not line:
            continue
        cmd = line.split()[0].lower()
        try:
            if cmd in {"quit", "exit"}:
                return
            elif cmd == "help":
                print(HELP)
            elif line == "demo":
                load_demo(mem)
                print_json(mem.stats())
            elif line.startswith("ingest "):
                content, meta = parse_meta(line)
                idx = mem.ingest(content, **meta)
                print(f"Ingested GeoCell #{idx} ({mem.cells[idx].status})")
            elif line.startswith("recall "):
                print_json(mem.recall(line.replace("recall", "", 1).strip()))
            elif line.startswith("ask "):
                print_json(mem.ask(line.replace("ask", "", 1).strip()))
            elif line.startswith("brief "):
                out = brief(mem, line.replace("brief", "", 1).strip())
                print(out.pop("brief"))
                print_json(out)
            elif cmd == "quantize":
                print_json(mem.quantize())
            elif line.startswith("savec "):
                print_json(mem.save_compact(line.split(maxsplit=1)[1]))
            elif line.startswith("loadc "):
                path = line.split(maxsplit=1)[1]
                mem = GeoCellField.load_compact(path)
                print(f"Loaded {len(mem.cells)} cells (quantized) from {path}")
            elif line.startswith("contradictions"):
                print_json(mem.contradictions(line.replace("contradictions", "", 1).strip()))
            elif line.startswith("timeline "):
                print_json(mem.timeline(line.replace("timeline", "", 1).strip()))
            elif line.startswith("path ") and "=>" in line:
                start, end = [x.strip() for x in line.replace("path", "", 1).split("=>", 1)]
                print_json(mem.path(start, end) or {"error": "No stable path found"})
            elif line.startswith("inspect "):
                print_json(mem.inspect(int(line.split()[1])))
            elif line.startswith("why "):
                print_json(mem.why(int(line.split()[1])))
            elif cmd in {"infer", "hypothesize"}:
                created = mem.hypothesize()
                print_json({"hypotheses": created} if created else {"hypotheses": [], "note": "nothing new to infer"})
            elif cmd == "settle":
                print_json(mem.settle())
            elif cmd == "trust":
                print_json(mem.propagate_trust())
            elif cmd in {"sleep", "consolidate"}:
                created = mem.consolidate()
                print_json({"concepts": created} if created else {"concepts": [], "note": "no cluster large enough"})
            elif line in {"stats", "show"}:
                print_json(mem.stats())
            elif line == "benchmark":
                print_json(run_benchmark())
            elif line.startswith("save "):
                path = line.split(maxsplit=1)[1]
                mem.save(path)
                print(f"Saved to {path}")
            elif line.startswith("load "):
                path = line.split(maxsplit=1)[1]
                mem = GeoCellField.load(path)
                print(f"Loaded {len(mem.cells)} cells from {path}")
            elif line.startswith("export"):
                parts = line.split(maxsplit=1)
                path = parts[1] if len(parts) > 1 else "geocell_export.json"
                mem.save(path)
                print(f"Exported to {path}")
            else:
                print("Unknown command. Type 'help'.")
        except Exception as e:
            print_json({"error": str(e)})


if __name__ == "__main__":
    cli()
