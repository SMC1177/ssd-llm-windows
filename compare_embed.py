import numpy as np, os, shutil
os.chdir(r"C:\ssd-llm")
shutil.copy(r"C:\llama.cpp\build\bin\Release\llama_cpp_embed.bin", "llama_cpp_embed.bin")

a = np.fromfile("llama_cpp_embed.bin", dtype=np.float32)
b = np.fromfile("ssd_llm_embed.bin", dtype=np.float32)
diff = np.abs(a-b)
cs = np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12)
print("=== EMBED for token 13048 ===")
print(f"llama.cpp first8: {a[:8].tolist()}")
print(f"ssd-llm   first8: {b[:8].tolist()}")
print(f"llama.cpp stats: min={a.min():+.4f} max={a.max():+.4f} mean={a.mean():+.4f} std={a.std():+.4f}")
print(f"ssd-llm   stats: min={b.min():+.4f} max={b.max():+.4f} mean={b.mean():+.4f} std={b.std():+.4f}")
print(f"max|diff|={diff.max():.6f}  mean|diff|={diff.mean():.6f}  cos_sim={cs:.6f}")
print(f"elements differ >1e-3: {np.sum(diff > 1e-3)}/{a.size}")
print(f"elements differ >0.01:  {np.sum(diff > 0.01)}/{a.size}")

# If embeddings match, problem is elsewhere
if diff.max() < 1e-3:
    print("\n>>> EMBEDDINGS MATCH. Bug is downstream (RMSNorm weight or eps).")
else:
    print("\n>>> EMBEDDINGS DO NOT MATCH. Bug is in embedding lookup / dequant.")
    # Look for scaling or sign patterns
    nonzero_a = np.abs(a) > 1e-4
    if nonzero_a.sum() > 100:
        ratios = b[nonzero_a]/a[nonzero_a]
        print(f"b/a ratio: median={np.median(ratios):+.4f} mean={ratios.mean():+.4f} std={ratios.std():.4f}")
    # Any hint of transpose? Check if permuted
    # Quick: sort both and compare - if same sorted distributions, it's a permutation bug
    a_sort = np.sort(a)
    b_sort = np.sort(b)
    sort_diff = np.abs(a_sort - b_sort).max()
    print(f"max|sorted_a - sorted_b| = {sort_diff:.6f}  (small = same values, different order = permutation)")
