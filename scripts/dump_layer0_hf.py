"""
Ground-truth layer-0 hidden state dump for ssd-llm comparison.

Loads Qwen/Qwen3-1.7B from HuggingFace (fp16 safetensors), runs forward pass
on token 13048 ("Hi"), and dumps the layer-0 output (2048 floats) to disk.

Compare against ssd-llm's layer-0 output from `SSD_MAX_LAYERS=1` with input "Hi".
"""

import json
import sys
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-1.7B"
EXPECTED_TOKEN_ID = 13048   # "Hi" per ssd-llm's tokenizer
OUT_BIN = "C:/ssd-llm/scripts/layer0_hf_ref.bin"
OUT_JSON = "C:/ssd-llm/scripts/layer0_hf_ref.json"


def main():
    print(f"[1/5] Loading tokenizer from {MODEL_ID}...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)

    # Verify "Hi" tokenizes to 13048 (matches ssd-llm's GPT-2 byte tokenizer)
    ids_hi = tok("Hi", add_special_tokens=False).input_ids
    print(f"      tokenize('Hi') = {ids_hi}")
    if ids_hi != [EXPECTED_TOKEN_ID]:
        print(f"      WARNING: expected [{EXPECTED_TOKEN_ID}], got {ids_hi}")
        print(f"      This may indicate a tokenizer mismatch with ssd-llm.")

    print(f"[2/5] Loading model from {MODEL_ID} (this may download ~3.4GB)...")
    # Use fp32 for max precision in reference; CPU is fine for single token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        device_map="cpu",
    )
    model.eval()

    cfg = model.config
    print(f"      arch: {cfg.architectures}")
    print(f"      n_layer: {cfg.num_hidden_layers}")
    print(f"      hidden_size: {cfg.hidden_size}")
    print(f"      n_head: {cfg.num_attention_heads}")
    print(f"      n_kv_head: {getattr(cfg, 'num_key_value_heads', 'n/a')}")
    print(f"      rope_theta: {getattr(cfg, 'rope_theta', 'n/a')}")
    print(f"      rms_eps: {getattr(cfg, 'rms_norm_eps', 'n/a')}")
    print(f"      vocab_size: {cfg.vocab_size}")

    print(f"[3/5] Forward pass with single token id={EXPECTED_TOKEN_ID}...")
    input_ids = torch.tensor([[EXPECTED_TOKEN_ID]], dtype=torch.long)
    with torch.no_grad():
        out = model(input_ids, output_hidden_states=True, return_dict=True)

    # hidden_states is a tuple: [embeddings, layer0_out, layer1_out, ..., layerN_out]
    hs = out.hidden_states
    print(f"      hidden_states tuple length: {len(hs)}")
    print(f"      hs[0] (embeddings) shape: {hs[0].shape}")
    print(f"      hs[1] (after layer 0) shape: {hs[1].shape}")

    # Embedding (for separate sanity check vs ssd-llm's embedding)
    emb = hs[0][0, 0].cpu().numpy().astype(np.float32)  # [hidden_size]
    # Layer 0 OUTPUT (after the first transformer block)
    layer0 = hs[1][0, 0].cpu().numpy().astype(np.float32)  # [hidden_size]

    print(f"[4/5] Dumping to {OUT_BIN} and {OUT_JSON}...")
    # Binary: just the 2048 floats for layer 0 output
    layer0.tofile(OUT_BIN)

    # JSON: includes diagnostics + first/last 16 values for quick eyeball
    summary = {
        "model_id": MODEL_ID,
        "token_id": EXPECTED_TOKEN_ID,
        "tokenize_check": ids_hi,
        "config": {
            "n_layer": cfg.num_hidden_layers,
            "hidden_size": cfg.hidden_size,
            "n_head": cfg.num_attention_heads,
            "n_kv_head": getattr(cfg, "num_key_value_heads", None),
            "rope_theta": getattr(cfg, "rope_theta", None),
            "rms_eps": getattr(cfg, "rms_norm_eps", None),
            "vocab_size": cfg.vocab_size,
        },
        "embedding": {
            "first16": emb[:16].tolist(),
            "max": float(emb.max()),
            "min": float(emb.min()),
            "mean": float(emb.mean()),
            "abs_max": float(np.abs(emb).max()),
        },
        "layer0_output": {
            "first16": layer0[:16].tolist(),
            "last16": layer0[-16:].tolist(),
            "max": float(layer0.max()),
            "min": float(layer0.min()),
            "mean": float(layer0.mean()),
            "abs_max": float(np.abs(layer0).max()),
            "norm_l2": float(np.linalg.norm(layer0)),
        },
        "binary_file": OUT_BIN,
        "binary_format": "2048 little-endian float32",
    }
    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[5/5] Done.")
    print(f"      Embedding[token {EXPECTED_TOKEN_ID}][0..8]: {emb[:8].tolist()}")
    print(f"      Layer 0 output[0..8]: {layer0[:8].tolist()}")
    print(f"      Layer 0 output[-8:]:  {layer0[-8:].tolist()}")
    print(f"      Layer 0 ||x||_2 = {np.linalg.norm(layer0):.4f}")


if __name__ == "__main__":
    sys.exit(main())
