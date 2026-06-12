"""Disk-cached wrapper around the real `claude` CLI.

Every prompt is hashed; identical prompts return the cached completion,
so reruns cost nothing and the eval is reproducible. A token/cost
estimate is tracked per uncached call.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from typing import Optional

CACHE_DIR = os.path.join(os.path.dirname(__file__), ".llm_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Rough public pricing for the small/fast model used here ($/1M tokens).
PRICE_IN = 0.80
PRICE_OUT = 4.00


class LLMClient:
    def __init__(self, model: str = "haiku"):
        self.model = model
        self.calls = 0
        self.cache_hits = 0
        self.in_tokens = 0
        self.out_tokens = 0
        self.wall_s = 0.0

    def _key(self, prompt: str) -> str:
        h = hashlib.sha256((self.model + "\0" + prompt).encode()).hexdigest()
        return os.path.join(CACHE_DIR, h + ".json")

    def complete(self, prompt: str, timeout: int = 90) -> str:
        path = self._key(prompt)
        if os.path.exists(path):
            self.cache_hits += 1
            return json.load(open(path))["text"]
        t0 = time.perf_counter()
        proc = subprocess.run(
            ["claude", "-p", prompt, "--model", self.model],
            capture_output=True, text=True, timeout=timeout,
        )
        dt = time.perf_counter() - t0
        text = proc.stdout.strip()
        self.calls += 1
        self.wall_s += dt
        # Approximate token accounting (4 chars/token heuristic).
        self.in_tokens += max(1, len(prompt) // 4)
        self.out_tokens += max(1, len(text) // 4)
        json.dump({"text": text, "wall_s": dt}, open(path, "w"))
        return text

    def cost(self) -> float:
        return self.in_tokens / 1e6 * PRICE_IN + self.out_tokens / 1e6 * PRICE_OUT

    def report(self) -> dict:
        return {
            "live_calls": self.calls,
            "cache_hits": self.cache_hits,
            "est_cost_usd": round(self.cost(), 5),
            "wall_s": round(self.wall_s, 1),
        }
