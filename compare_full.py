import numpy as np
import os

os.chdir(r"C:\ssd-llm")

targets = [
    ("attn_norm",    "llama_cpp_attn_norm.bin",    "ssd_llm_attn_norm.bin"),
    ("Qraw",         "llama_cpp_Qraw.bin",         "ssd_llm_Qraw.bin"),
    ("Kraw",         "llama_cpp_Kraw.bin",         "ssd_llm_Kraw.bin"),
    ("Vraw",         "llama_cpp_Vraw.bin",         "ssd_llm_Vraw.bin"),
    ("attn_post_wo", "llama_cpp_attn_post_wo.bin", None),
    ("layer0_out",   "llama_cpp_layer0.bin",       "ssd_llm_layer0.bin"),
]

# Copy llama.cpp bins to current dir
import shutil
for _, lf, _ in targets:
    src = r"C:\llama.cpp\build\bin\Release\\" + lf
    if os.path.exists(src):
        shutil.copy(src, lf)

for name, lf, sf in targets:
    print(f"\n=== {name} ===")
    if not os.path.exists(lf):
        print(f"MISSING {lf}")
        continue
    a = np.fromfile(lf, dtype=np.float32)
    print(f"llama.cpp {lf}: n={a.size} min={a.min():+.4f} max={a.max():+.4f} mean={a.mean():+.4f} first8={a[:8].tolist()}")
    if sf and os.path.exists(sf):
        b = np.fromfile(sf, dtype=np.float32)
        print(f"ssd-llm   {sf}: n={b.size} min={b.min():+.4f} max={b.max():+.4f} mean={b.mean():+.4f} first8={b[:8].tolist()}")
        if a.size == b.size:
            diff = np.abs(a - b)
            cs = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
            print(f"  max|diff|={diff.max():.4f}  mean|diff|={diff.mean():.4f}  cos_sim={cs:.4f}  elements_diff>0.01={np.sum(diff > 0.01)}/{a.size}")
        else:
            print(f"  SHAPE MISMATCH {a.size} vs {b.size}")
    elif sf:
        print(f"  (missing {sf})")
