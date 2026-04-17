import numpy as np, os
os.chdir(r"C:\ssd-llm")

print("=== attn_norm output (= attn_input in ssd-llm, gated to prefill only) ===")
a = np.fromfile("llama_cpp_attn_norm.bin", dtype=np.float32)
b = np.fromfile("ssd_llm_attn_input.bin", dtype=np.float32)
diff = np.abs(a - b)
cs = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
print(f"llama.cpp first8: {a[:8].tolist()}")
print(f"ssd-llm   first8: {b[:8].tolist()}")
print(f"max|diff|={diff.max():.4f}  mean|diff|={diff.mean():.4f}  cos_sim={cs:.4f}  >0.01: {np.sum(diff>0.01)}/{a.size}")

# Also test if there's a scaling factor - maybe ssd-llm is missing the RMS divide or weight multiply
if np.sum(diff > 0.01) > 1000:
    # Try: is ssd-llm = llama * const?
    mask = np.abs(a) > 0.01
    if mask.sum() > 10:
        ratios = b[mask] / a[mask]
        print(f"ratio b/a: median={np.median(ratios):.4f} mean={ratios.mean():.4f} std={ratios.std():.4f}")
    # Try: is ssd-llm proportional to embedding (no norm applied)?
    # (can't test without embedding dump)

print("\n=== Qraw (gated prefill) ===")
a = np.fromfile("llama_cpp_Qraw.bin", dtype=np.float32)
b = np.fromfile("ssd_llm_Qraw.bin", dtype=np.float32)
diff = np.abs(a - b)
cs = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
print(f"llama.cpp Q head0[0..8]: {a[:8].tolist()}")
print(f"ssd-llm   Q head0[0..8]: {b[:8].tolist()}")
print(f"max|diff|={diff.max():.4f}  mean|diff|={diff.mean():.4f}  cos_sim={cs:.4f}")

print("\n=== Vraw (gated prefill) ===")
a = np.fromfile("llama_cpp_Vraw.bin", dtype=np.float32)
b = np.fromfile("ssd_llm_Vraw.bin", dtype=np.float32)
diff = np.abs(a - b)
cs = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
print(f"llama.cpp V head0[0..8]: {a[:8].tolist()}")
print(f"ssd-llm   V head0[0..8]: {b[:8].tolist()}")
print(f"max|diff|={diff.max():.4f}  mean|diff|={diff.mean():.4f}  cos_sim={cs:.4f}")
