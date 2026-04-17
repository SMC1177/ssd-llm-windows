import numpy as np
a = np.fromfile("llama_cpp_embed.bin", dtype=np.float32)
b = np.fromfile("ssd_llm_embed.bin", dtype=np.float32)

N = a.size
print(f"N={N}")

# For each ssd index i, find closest llama index j
# This is O(N^2) but fine for N=2048
print("\nIndex mapping (first 20 ssd indices):")
for i in range(20):
    j = np.argmin(np.abs(a - b[i]))
    best_diff = abs(a[j] - b[i])
    print(f"  ssd[{i:4d}]={b[i]:+.4f} best match llama[{j:4d}]={a[j]:+.4f} (diff={best_diff:.6f})")

# Look for a periodic permutation pattern
# Test: does ssd[i] match llama[f(i)] for some function f?
# For a quantized tensor, block dequantization may reorder.
# Qwen3 embedding is Q4K or similar - dequant may interleave.
# Let's check stride patterns.

# Compute: for each ssd index i, what llama index has exact match?
mapping = []
for i in range(N):
    matches = np.where(np.abs(a - b[i]) < 1e-5)[0]
    mapping.append(matches.tolist() if len(matches) else [])

exact_match_count = sum(1 for m in mapping if len(m) > 0)
print(f"\n{exact_match_count}/{N} ssd values have exact llama match")

# If exact matches exist, look at index pattern
first_matches = [m[0] if m else -1 for m in mapping]
print(f"First 32 unique-match indices: {[m[0] if len(m)==1 else '?' for m in mapping[:32]]}")

# Check if ssd block-wise reads
# Q4K super-block = 256 elements? Check block boundaries.
# Try interpreting: ssd reads Q4K block differently
# A Q4K super-block has 256 weights. If ssd mis-reads the nibble ordering,
# we'd see 256-element blocks that are permuted.

# Test if first 256 of ssd are a permutation of first 256 of llama
a_block0 = set(np.round(a[:256], 5).tolist())
b_block0 = set(np.round(b[:256], 5).tolist())
common = a_block0 & b_block0
print(f"Block 0 (elements 0..256): {len(common)}/{len(a_block0)} llama values present in ssd block 0")

# Same for block 1
a_block1 = set(np.round(a[256:512], 5).tolist())
b_block1 = set(np.round(b[256:512], 5).tolist())
common = a_block1 & b_block1
print(f"Block 1 (elements 256..512): {len(common)}/{len(a_block1)} llama values present in ssd block 1")

# And whole array as multiset
from collections import Counter
ac = Counter(np.round(a, 5).tolist())
bc = Counter(np.round(b, 5).tolist())
shared = sum(min(ac[v], bc[v]) for v in ac)
print(f"\nAs multisets: {shared}/{N} shared values (exact permutation test)")
