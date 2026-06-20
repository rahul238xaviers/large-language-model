"""AST syntax checker — invoked by `make check`.

Scans all .py files in the repo (excluding venvs and caches) and reports
any files that fail to parse.  Exits 0 if all files are clean, 1 otherwise.
"""
import ast
import pathlib
import sys

_SKIP = {".apple_env", ".venv", "__pycache__", ".git", "node_modules"}

root = pathlib.Path(__file__).resolve().parents[2]
files = [
    f for f in root.rglob("*.py")
    if not any(s in f.parts for s in _SKIP)
]

ok = True
for f in sorted(files):
    try:
        ast.parse(f.read_text(encoding="utf-8"))
        print(f"  OK   {f.relative_to(root)}")
    except SyntaxError as exc:
        print(f"  ERR  {f.relative_to(root)}: {exc}")
        ok = False

sys.exit(0 if ok else 1)
