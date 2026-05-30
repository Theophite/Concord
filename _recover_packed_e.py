"""Recover prototype_packed_e.py from transcript .jsonl.

Robust method: find every Read tool_use whose input.file_path points at
prototype_packed_e.py, capture its tool_use_id, then pull the matching
tool_result content (the cat -n text). Merge all reads by line number.
This catches reads of ANY line range, not just the early markered part.
"""
import json
import glob
import os
import re

JSONL_DIR = r"C:\Users\ophit\.claude\projects\C--foliated-onetrainer"
LINE_RE = re.compile(r"^\s*(\d+)\t(.*)$")


def walk(obj):
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from walk(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from walk(v)


def result_text(content):
    """tool_result.content may be str or list of {type:text,text:..}."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for c in content:
            if isinstance(c, dict) and isinstance(c.get("text"), str):
                parts.append(c["text"])
            elif isinstance(c, str):
                parts.append(c)
        return "\n".join(parts)
    return ""


want_ids = {}     # tool_use_id -> (offset, limit) for diagnostics
results = {}       # tool_use_id -> text
n_use = 0
for path in sorted(glob.glob(os.path.join(JSONL_DIR, "*.jsonl"))):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            if "prototype_packed_e" not in raw and "tool_result" not in raw:
                continue
            try:
                obj = json.loads(raw)
            except Exception:
                continue
            for d in walk(obj):
                t = d.get("type")
                if t == "tool_use" and d.get("name") in ("Read", "read"):
                    inp = d.get("input") or {}
                    fp = str(inp.get("file_path", ""))
                    if "prototype_packed_e" in fp:
                        want_ids[d.get("id")] = (inp.get("offset"),
                                                  inp.get("limit"))
                        n_use += 1
                elif t == "tool_result":
                    tid = d.get("tool_use_id")
                    if tid is not None:
                        txt = result_text(d.get("content"))
                        if txt:
                            results.setdefault(tid, txt)

print(f"Read tool_uses on packed_e: {n_use}")
for tid, (off, lim) in want_ids.items():
    have = tid in results
    nlines = results[tid].count("\n") + 1 if have else 0
    print(f"  use {str(tid)[-8:]}: offset={off} limit={lim}  "
          f"result={'YES' if have else 'MISSING'} ({nlines} text lines)")

recovered = {}
for tid in want_ids:
    txt = results.get(tid, "")
    for ln in txt.split("\n"):
        m = LINE_RE.match(ln)
        if m:
            recovered[int(m.group(1))] = m.group(2)

if not recovered:
    print("NO numbered content recovered.")
    raise SystemExit(1)

lo, hi = min(recovered), max(recovered)
missing = [i for i in range(lo, hi + 1) if i not in recovered]
print(f"recovered line range {lo}..{hi}  ({len(recovered)} lines)  "
      f"missing {len(missing)}")
if missing:
    rngs = []
    s = p = missing[0]
    for x in missing[1:]:
        if x == p + 1:
            p = x
        else:
            rngs.append((s, p)); s = p = x
    rngs.append((s, p))
    print("  missing:", ", ".join(f"{a}-{b}" if a != b else f"{a}"
                                   for a, b in rngs[:60]))

out = os.path.join(os.path.dirname(__file__), "prototype_packed_e.RECOVERED.py")
with open(out, "w", encoding="utf-8") as f:
    for i in range(lo, hi + 1):
        f.write(recovered.get(i, f"# <<<MISSING {i}>>>") + "\n")
print(f"wrote {out}  ({hi} lines)")
