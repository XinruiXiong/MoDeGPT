import torch
from transformers import AutoModelForCausalLM

def compare_vo_weights(original_model_path, compressed_model_path, head_dim=128, n_heads=32, n_layers=32):
    print("Loading compressed model to CUDA...")
    model_compressed = AutoModelForCausalLM.from_pretrained(
        compressed_model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    print("Loading original model to CPU...")
    model_original = AutoModelForCausalLM.from_pretrained(
        original_model_path,
        torch_dtype=torch.float32,
        device_map={"": "cpu"}
    )

    print("Starting VO weight comparison (v_proj & o_proj)...")

    for layer_idx in range(n_layers):
        try:
            v_orig = model_original.model.layers[layer_idx].self_attn.v_proj.weight.cpu()
            v_comp = model_compressed.model.layers[layer_idx].self_attn.v_proj.weight.cpu()

            o_orig = model_original.model.layers[layer_idx].self_attn.o_proj.weight.cpu()
            o_comp = model_compressed.model.layers[layer_idx].self_attn.o_proj.weight.cpu()

            v_diff = torch.mean(torch.abs(v_orig - v_comp)).item()
            o_diff = torch.mean(torch.abs(o_orig - o_comp)).item()

            print(f"[Layer {layer_idx:02d}] v_proj diff: {v_diff:.6f} | o_proj diff: {o_diff:.6f}")

        except Exception as e:
            print(f"[Layer {layer_idx:02d}] Comparison failed: {e}")

if __name__ == "__main__":
    original_model_path = "meta-llama/Llama-2-7b-hf"  # or your local path
    compressed_model_path = "/u/scratch/x/xxiong/compressed_output/llama2-7b"

    compare_vo_weights(original_model_path, compressed_model_path)
