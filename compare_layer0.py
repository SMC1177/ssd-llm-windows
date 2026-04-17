import numpy as np
a = np.fromfile('llama_cpp_layer0.bin', dtype=np.float32)
b = np.fromfile('ssd_llm_layer0.bin', dtype=np.float32)
print(f"llama.cpp: {a.shape} bytes={a.nbytes} min={a.min():.4f} max={a.max():.4f} mean={a.mean():.4f}")
print(f"ssd-llm:   {b.shape} bytes={b.nbytes} min={b.min():.4f} max={b.max():.4f} mean={b.mean():.4f}")
print(f"\nmax abs diff: {np.max(np.abs(a-b)):.6f}")
print(f"mean abs diff: {np.mean(np.abs(a-b)):.6f}")
print(f"cos sim: {np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)):.6f}")
print(f"\nllama.cpp[0..8]: {a[:8]}")
print(f"ssd-llm[0..8]:   {b[:8]}")
print(f"\nllama.cpp[1024..1032]: {a[1024:1032]}")
print(f"ssd-llm[1024..1032]:   {b[1024:1032]}")

# First big divergence location
diff = np.abs(a-b)
biggest = np.argmax(diff)
print(f"\nBiggest divergence at idx {biggest}: llama={a[biggest]:.4f} ssd={b[biggest]:.4f} diff={diff[biggest]:.4f}")

# Where do values first significantly differ? (>0.1)
mask = diff > 0.1
first_big = np.where(mask)[0]
if len(first_big):
    print(f"First idx where diff > 0.1: {first_big[0]} (total {len(first_big)}/{len(a)} indices exceed 0.1)")
else:
    print("No indices where diff > 0.1 — close match")
