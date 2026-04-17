import numpy as np
a = np.fromfile("llama_cpp_embed.bin", dtype=np.float32)
b = np.fromfile("ssd_llm_embed.bin", dtype=np.float32)
diff = np.abs(a-b)
# Find first index where diff > 1e-4
first_bad = np.where(diff > 1e-4)[0]
if len(first_bad):
    i = first_bad[0]
    print(f"First differ at index {i}: llama={a[i]:+.6f} ssd={b[i]:+.6f}")
    # Show neighborhood
    lo, hi = max(0, i-4), min(a.size, i+12)
    for j in range(lo, hi):
        ok = "" if diff[j] < 1e-4 else " <<< DIFF"
        print(f"  [{j:4d}] llama={a[j]:+.6f} ssd={b[j]:+.6f} diff={diff[j]:+.6f}{ok}")
print()
# Check per 32-element sub-blocks: count diffs per block
N = a.size
for s in [0, 1]:
    pass
print("Diff-rate per 256-block (Q4K super-block):")
for block in range(8):
    s = block * 256
    e = min(s + 256, N)
    bd = (diff[s:e] > 1e-4).sum()
    print(f"  block {block} [{s:4d}..{e-1:4d}]: {bd}/{e-s} differ")

print("\nDiff-rate per 32-element sub-block (Q4K sub-block):")
for block in range(min(16, N // 32)):
    s = block * 32
    e = s + 32
    bd = (diff[s:e] > 1e-4).sum()
    mark = " <<< all-match" if bd == 0 else ""
    print(f"  sub {block:2d} [{s:4d}..{e-1:4d}]: {bd:2d}/32 differ{mark}")
