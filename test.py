# test_compressed_model.py

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    # ===== 配置路径 =====
    model_dir = "compressed_output/opt-6.7b"    # 压缩后的模型权重目录
    original_model_name = "facebook/opt-6.7b"   # 原始模型名字（用于加载tokenizer）
    output_file = "inference_output.txt"        # 保存生成结果的位置

    # ===== 检查设备 =====
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ===== 加载Tokenizer和Model =====
    try:
        tokenizer = AutoTokenizer.from_pretrained(original_model_name)
    except Exception as e:
        raise RuntimeError(f"Failed to load tokenizer from {original_model_name}: {e}")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            device_map="auto",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_dir}: {e}")

    # ===== 推理 =====
    prompt = "The quick brown fox"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    try:
        outputs = model.generate(**inputs, max_new_tokens=50)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        raise RuntimeError(f"Inference failed: {e}")

    # ===== 保存输出到文件 =====
    with open(output_file, "w") as f:
        f.write(generated_text)

    print(f"Inference completed. Output saved to {output_file}")

if __name__ == "__main__":
    main()
