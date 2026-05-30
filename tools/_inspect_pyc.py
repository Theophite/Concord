"""Inspect prototype_packed_e.cpython-311.pyc: recover structure,
docstrings, signatures, and constants from the compiled module without
needing a decompiler. Also dump full disassembly to a file.
"""
import dis
import marshal
import importlib.util
import sys
import types

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

PYC = r"C:\concord\__pycache__\prototype_packed_e.cpython-311.pyc"

with open(PYC, "rb") as f:
    f.read(16)  # skip 16-byte header (magic, flags, mtime, size) for 3.7+
    code = marshal.load(f)

print(f"=== module: {code.co_filename} ===")
if code.co_consts and isinstance(code.co_consts[0], str):
    print("MODULE DOCSTRING:")
    print(code.co_consts[0][:1500])
    print("...<truncated>" if len(code.co_consts[0]) > 1500 else "")
print()

# module-level float/int constants (catch eps defaults etc. at module scope)
mod_nums = [c for c in code.co_consts if isinstance(c, (int, float))
            and not isinstance(c, bool)]
print("module-level numeric consts:", sorted(set(mod_nums))[:40])
print()


def describe(co, depth=0):
    pad = "  " * depth
    args = co.co_varnames[:co.co_argcount + co.co_kwonlyargcount]
    print(f"{pad}- {co.co_name}({', '.join(args)})  "
          f"argc={co.co_argcount} kwonly={co.co_kwonlyargcount} "
          f"line={co.co_firstlineno}")
    # docstring = first const if str
    if co.co_consts and isinstance(co.co_consts[0], str) \
            and co.co_name not in ("<module>",):
        ds = co.co_consts[0].strip().splitlines()
        if ds:
            print(f"{pad}  doc: {ds[0][:90]}")
    # numeric consts in this code object (defaults, magic numbers)
    nums = sorted(set(c for c in co.co_consts
                      if isinstance(c, (int, float)) and not isinstance(c, bool)))
    if nums:
        print(f"{pad}  nums: {nums[:30]}")
    for c in co.co_consts:
        if isinstance(c, types.CodeType):
            describe(c, depth + 1)


print("=== nested code objects (functions / kernels / classes) ===")
describe(code)

# Full disassembly to file for manual reconstruction reference.
out = r"C:\concord\packed_e_disasm.txt"
with open(out, "w", encoding="utf-8") as f:
    def rec_dis(co, depth=0):
        f.write("\n" + "=" * 70 + "\n")
        f.write(f"{'  '*depth}CODE: {co.co_name}  (line {co.co_firstlineno})\n")
        f.write("=" * 70 + "\n")
        try:
            dis.dis(co, file=f)
        except Exception as e:
            f.write(f"<dis failed: {e}>\n")
        for c in co.co_consts:
            if isinstance(c, types.CodeType):
                rec_dis(c, depth + 1)
    rec_dis(code)
print(f"\nfull disassembly -> {out}")

# decompiler availability
print("\n=== decompiler availability ===")
for mod in ("decompyle3", "uncompyle6", "xdis", "decompile3"):
    print(f"  {mod}:", "yes" if importlib.util.find_spec(mod) else "no")
