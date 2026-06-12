"""Sandbox helpers: clean repo copies, bug injection, single-test runs."""
from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple

CLEAN_REPO = Path(__file__).resolve().parent / "repo"


def fresh_repo() -> Path:
    dst = Path(tempfile.mkdtemp(prefix="geocell_eval_"))
    shutil.copytree(CLEAN_REPO, dst / "repo")
    return dst / "repo"


def inject(repo: Path, rel_file: str, broken: str, clean: str, func: str) -> bool:
    """Corrupt one function. The clean->broken substring is identical
    across sibling functions, so the edit must be scoped to `func`."""
    old_func = read_func(repo, rel_file, func)
    if old_func is None or clean not in old_func:
        return False
    new_func = old_func.replace(clean, broken, 1)
    return replace_func(repo, rel_file, func, new_func)


def read_func(repo: Path, rel_file: str, func: str) -> Optional[str]:
    """Return the source block of a top-level or method `def func`."""
    text = (repo / rel_file).read_text().splitlines()
    out, capturing, indent = [], False, 0
    for line in text:
        m = re.match(r"^(\s*)def\s+" + re.escape(func) + r"\b", line)
        if m and not capturing:
            capturing, indent = True, len(m.group(1))
            out.append(line)
            continue
        if capturing:
            if line.strip() and (len(line) - len(line.lstrip())) <= indent and not line.lstrip().startswith(")"):
                break
            out.append(line)
    return "\n".join(out).rstrip() if out else None


def replace_func(repo: Path, rel_file: str, func: str, new_src: str) -> bool:
    old = read_func(repo, rel_file, func)
    if old is None:
        return False
    p = repo / rel_file
    p.write_text(p.read_text().replace(old, new_src.rstrip(), 1))
    return True


def run_test(repo: Path, test_node: str, timeout: int = 60) -> Tuple[bool, str]:
    proc = subprocess.run(
        ["python", "-m", "pytest", test_node, "-q", "--no-header", "-p", "no:cacheprovider"],
        cwd=repo, capture_output=True, text=True, timeout=timeout,
    )
    return proc.returncode == 0, proc.stdout + proc.stderr


def run_full_suite(repo: Path, timeout: int = 120) -> Tuple[int, int]:
    proc = subprocess.run(
        ["python", "-m", "pytest", "tests/", "-q", "--no-header", "-p", "no:cacheprovider"],
        cwd=repo, capture_output=True, text=True, timeout=timeout,
    )
    out = proc.stdout + proc.stderr
    passed = sum(int(x) for x in re.findall(r"(\d+) passed", out))
    failed = sum(int(x) for x in re.findall(r"(\d+) failed", out))
    return passed, failed


def classify_failure(log: str) -> str:
    """Distinguish hallucinated references from plain logic errors."""
    if re.search(r"\b(NameError|AttributeError|ImportError|ModuleNotFoundError)\b", log):
        return "hallucinated_reference"
    if "SyntaxError" in log:
        return "syntax_error"
    if "AssertionError" in log or "assert" in log:
        return "logic_error"
    return "other"
