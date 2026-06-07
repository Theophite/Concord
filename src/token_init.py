"""Initialization strategy for adding a vocabulary of new tokens to SDXL.

Each new token is seeded from the MEAN of its surface form's CLIP subword
embeddings (resolve_token_init's word path) -- it starts where its spelling already
points, then training moves it. Per-reason handling derives the init STRING:
  - [REDEFINE] (existing words): tokenize to themselves -> init = current meaning;
  - camelCase: split the form on case boundaries (tok -> "tok") for cleaner subwords;
  - name / not_in_dictionary: the normalized word, subword-tokenized as-is.

`init_specs_from_list` -> (names, init_specs) feeds straight into
concord_embedding_packed.insert_new_tokens(te, tokenizer, names, init_specs).
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.resolve()))

# The token list (name, count, reason, forms). Paste-tolerant: parsed by regex.
LIST = "REDACTED -- token vocabulary kept in gitignored token_list.local.txt"


def parse_list(text=LIST):
    rows = []
    for line in text.splitlines():
        m = re.search(r"(\S+)\s+count=\s*(\d+)\s+reason=(\S+)\s+forms=\(([^)]*)\)", line)
        if not m:
            continue
        rows.append(dict(name=m.group(1), count=int(m.group(2)), reason=m.group(3),
                         forms=[f.strip() for f in m.group(4).split(",")],
                         redefine="[REDEFINE]" in line))
    return rows


def init_string(e):
    """The string whose CLIP subwords seed this token's embedding."""
    if e["redefine"] or e["reason"] == "name":
        return e["name"]                                  # existing/name -> its own subwords
    if e["reason"] == "camelCase":
        s = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", e["forms"][0]).lower()
        if any(len(w) == 1 for w in s.split()):     # messy multi-caps -> plain name
            return e["name"]
        return s
    return e["name"]


def init_specs_from_list(entries=None):
    """-> (names, init_specs) for insert_new_tokens."""
    entries = entries or parse_list()
    names = ["<" + e["name"] + ">" for e in entries]      # placeholder tokens
    return names, [init_string(e) for e in entries]


if __name__ == "__main__":
    from transformers import CLIPTokenizer
    tok = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    rows = parse_list()
    print(f"{len(rows)} tokens; showing the init string + CLIP subwords (whose embeddings "
          f"are averaged for the init):\n")
    n_single = 0
    for e in rows:
        s = init_string(e)
        ids = tok(s, add_special_tokens=False).input_ids
        pieces = [p.replace("</w>", "") for p in tok.convert_ids_to_tokens(ids)]
        if len(ids) == 1:
            n_single += 1
        flag = "  [REDEFINE: existing token]" if e["redefine"] else ""
        print(f"  {e['name']:12} {e['reason']:16} init='{s}' -> {pieces}{flag}")
    print(f"\n{n_single}/{len(rows)} already exist as single CLIP tokens "
          f"(init = that token's own embedding); the rest seed from subword means.")
