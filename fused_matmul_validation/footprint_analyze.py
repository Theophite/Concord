"""Measure the Concord packed-state footprint breakdown from the epoch-45 backup,
grouped by UNet layer category (transformer attn+MLP vs conv resnets) and by tensor
suffix. This says exactly how much a layer-subset filter would shrink the footprint.
No GPU, no run -- just parse the safetensors headers."""
import glob, os, json, struct
from collections import defaultdict

BK = r"C:/Concord/overfit/workspace/backup/2026-06-06_09-45-02-backup-540-45-0"
DT = {'F64':8,'F32':4,'F16':2,'BF16':2,'I64':8,'I32':4,'I16':2,'I8':1,'U8':1,'BOOL':1}


def category(name):
    n = name.lower()
    if 'attentions' in n or 'transformer_blocks' in n or 'attn' in n or '.ff.' in n:
        return 'attn+mlp (transformer)'
    if 'resnet' in n or 'conv' in n or 'sampler' in n or 'time_emb' in n:
        return 'conv resnets'
    return 'other (norm/embed/proj)'


shards = glob.glob(os.path.join(BK, "**", "*.safetensors"), recursive=True)
by_cat = defaultdict(float)
by_suffix = defaultdict(float)
by_dtype = defaultdict(float)
elems_by_cat = defaultdict(int)
total = 0.0
sample_names = []

for sh in shards:
    with open(sh, 'rb') as f:
        hlen = struct.unpack('<Q', f.read(8))[0]
        hdr = json.loads(f.read(hlen).decode('utf-8'))
    for name, meta in hdr.items():
        if name == '__metadata__':
            continue
        elems = 1
        for s in meta['shape']:
            elems *= s
        nbytes = elems * DT.get(meta['dtype'], 4)
        total += nbytes
        cat = category(name)
        by_cat[cat] += nbytes
        by_suffix[name.split('.')[-1]] += nbytes
        by_dtype[meta['dtype']] += nbytes
        elems_by_cat[cat] += elems
        if len(sample_names) < 8:
            sample_names.append(f"{name}  {meta['dtype']}{meta['shape']}")

G = 1e9
print(f"=== backup: {os.path.basename(BK)} | {len(shards)} shard(s) ===")
print(f"total packed-state on disk: {total/G:.2f} G\n")
print("sample tensor names:")
for s in sample_names:
    print("   ", s)
print("\nby tensor suffix (top 8):")
for k, v in sorted(by_suffix.items(), key=lambda x: -x[1])[:8]:
    print(f"   {k:24s} {v/G:6.2f} G")
print("\nby dtype:")
for k, v in sorted(by_dtype.items(), key=lambda x: -x[1]):
    print(f"   {k:6s} {v/G:6.2f} G")
print("\nby UNet layer category (= the layer-subset lever):")
for k, v in sorted(by_cat.items(), key=lambda x: -x[1]):
    print(f"   {k:26s} {v/G:6.2f} G  ({100*v/total:4.1f}%)  ~{elems_by_cat[k]/G:.3f}B params")

# Footprint projection: GPU runtime ~ packed_state + bf16 weight cache (2B x packed params).
attn = elems_by_cat['attn+mlp (transformer)']
conv = elems_by_cat['conv resnets']
oth = elems_by_cat['other (norm/embed/proj)']
allp = attn + conv + oth
print("\n=== GPU footprint projection (packed 4B + bf16 cache 2B = 6B/param) ===")
print(f"   full UNet     : ~{6*allp/G:5.2f} G   ({allp/G:.3f}B packed params)")
print(f"   attn+mlp only : ~{6*attn/G:5.2f} G   (freeze conv -> saves ~{6*conv/G:.2f} G)")
