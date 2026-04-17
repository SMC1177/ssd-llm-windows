import numpy as np
a = np.fromfile("llama_cpp_embed.bin", dtype=np.float32)  # reference
b = np.fromfile("ssd_llm_embed.bin", dtype=np.float32)    # suspected permuted

N = a.size  # 2048
half = N // 2  # 1024

# Hypothesis 1: ssd[2k] = a[k], ssd[2k+1] = a[half+k] (NEOX-style half-dim interleave)
hyp1 = np.empty_like(a)
hyp1[0::2] = a[:half]
hyp1[1::2] = a[half:]
diff1 = np.abs(hyp1 - b).max()
print(f"Hyp1 (even=first_half, odd=second_half interleave): max|diff|={diff1:.6e}")

# Hypothesis 2: ssd reading every other byte (stride error)
# Already covered by hyp1 if it matches

# Hypothesis 3: just a reshape/transpose
# a has shape (2048,) - if someone reshaped to (n_head, head_dim) and transposed to (head_dim, n_head)
for n_head in [16, 8, 32]:
    if N % n_head == 0:
        head_dim = N // n_head
        a_r = a.reshape(n_head, head_dim).T.flatten()
        diff = np.abs(a_r - b).max()
        if diff < 0.01:
            print(f"Hyp: reshape ({n_head},{head_dim}).T matches (max diff {diff:.4f})")

# Hypothesis 4: bit-pair or other stride
# Test: any simple cyclic shift?
for s in [1, 2, 1024]:
    sh = np.roll(a, s)
    d = np.abs(sh - b).max()
    if d < 0.01:
        print(f"Roll by {s} matches (max diff {d:.4f})")

# Verify hyp1 in detail if close
if diff1 < 0.01:
    print("\n>>> CONFIRMED: ssd embedding is NEOX half-dim interleaved view of the real embedding")
    print(f"    ssd[2k]   = llama[k]         for k in 0..{half}")
    print(f"    ssd[2k+1] = llama[{half}+k]  for k in 0..{half}")
