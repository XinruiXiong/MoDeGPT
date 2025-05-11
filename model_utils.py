# import os
# import torch
# import logging
# from transformers import AutoModelForCausalLM, AutoTokenizer

# logger = logging.getLogger("MoDeGPT")


# def _get_max_memory_map(reserved_gb=4):
#     """
#     动态获取每张 GPU 的最大可用内存限制，保留 reserved_gb（默认 4GB）显存作为 buffer。
#     返回用于 device_map='auto' 的 max_memory 映射。
#     """
#     max_memory = {}
#     if torch.cuda.is_available():
#         for i in range(torch.cuda.device_count()):
#             total = torch.cuda.get_device_properties(i).total_memory // (1024 ** 3)
#             usable = max(total - reserved_gb, 1)  # 至少保留 1GB
#             max_memory[f"cuda:{i}"] = f"{usable}GB"
#     max_memory["cpu"] = "20GB"
#     return max_memory


# def load_model(model_name: str, device: str = "cuda"):
#     """
#     Load HuggingFace model and tokenizer with float16, auto device map, and low memory usage.
#     Structure is unchanged; safe for MoDeGPT compression.
#     """
#     try:
#         logger.info(f"Loading model from: {model_name}")
#         tokenizer = AutoTokenizer.from_pretrained(model_name)

#         try:
#             model = AutoModelForCausalLM.from_pretrained(
#                 model_name,
#                 device_map="auto",
#                 max_memory=_get_max_memory_map(),
#                 torch_dtype=torch.float16,
#                 low_cpu_mem_usage=True,
#             )
#             logger.info("✔ Loaded with device_map='auto', float16, low_cpu_mem_usage=True.")
#         except Exception as e:
#             logger.warning(f"[Fallback] device_map='auto' failed: {e}")
#             model = AutoModelForCausalLM.from_pretrained(
#                 model_name,
#                 torch_dtype=torch.float16,
#                 low_cpu_mem_usage=True,
#             )
#             model.to(device)
#             logger.info("✔ Loaded with model.to(device) fallback.")

#         if tokenizer.pad_token is None:
#             tokenizer.pad_token = tokenizer.eos_token
#             logger.info("No pad_token found. Set pad_token = eos_token.")

#         model.eval()
#         return model, tokenizer, model.config

#     except Exception as e:
#         logger.error(f"[Error] Failed to load model {model_name}: {e}")
#         raise


# def save_model(model: torch.nn.Module, tokenizer, save_dir: str, source_model_name: str):
#     """
#     Save model and tokenizer. Only weights are compressed; structure unchanged.
#     """
#     try:
#         os.makedirs(save_dir, exist_ok=True)

#         # Save weights + config (assumes structure unchanged)
#         model.save_pretrained(save_dir, torch_dtype=torch.float16)
#         tokenizer.save_pretrained(save_dir)

#         # Save tokenizer origin
#         with open(os.path.join(save_dir, "tokenizer_source.txt"), "w") as f:
#             f.write(source_model_name.strip())

#         logger.info(f"✔ Model, tokenizer, and tokenizer_source.txt saved to {save_dir}")

#     except Exception as e:
#         logger.error(f"[Error] Failed to save model to {save_dir}: {e}")
#         raise


# def reload_compressed_model(model_dir: str, device: str = "cuda"):
#     """
#     Reload a compressed model assuming weights are patched but structure unchanged.
#     """
#     try:
#         logger.info(f"Reloading compressed model from: {model_dir}")
#         tokenizer_source_path = os.path.join(model_dir, "tokenizer_source.txt")

#         if not os.path.exists(tokenizer_source_path):
#             raise FileNotFoundError("Missing tokenizer_source.txt. Cannot reload tokenizer.")

#         with open(tokenizer_source_path, "r") as f:
#             tokenizer_source = f.read().strip()

#         tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)

#         try:
#             model = AutoModelForCausalLM.from_pretrained(
#                 model_dir,
#                 device_map="auto",
#                 max_memory=_get_max_memory_map(),
#                 torch_dtype=torch.float16,
#                 low_cpu_mem_usage=True,
#             )
#         except Exception as e:
#             logger.warning(f"[Reload fallback] device_map failed: {e}")
#             model = AutoModelForCausalLM.from_pretrained(
#                 model_dir,
#                 torch_dtype=torch.float16,
#                 low_cpu_mem_usage=True,
#             )
#             model.to(device)

#         model.eval()
#         logger.info("✔ Reloaded compressed model and tokenizer successfully.")
#         return model, tokenizer

#     except Exception as e:
#         logger.error(f"[Error] Failed to reload compressed model from {model_dir}: {e}")
#         raise



import os
import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger("MoDeGPT")


def load_model(model_name: str, device: int = 0):
    """
    加载原始 HuggingFace 模型和 tokenizer，使用 float16 精度并指定显式 CUDA 设备。
    不使用 device_map='auto'，确保稳定加载。
    """
    try:
        logger.info(f"Loading model from: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        model.to(f"cuda:{device}")
        logger.info(f"✔ Loaded model on cuda:{device} with float16.")

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("No pad_token found. Set pad_token = eos_token.")

        model.eval()
        return model, tokenizer, model.config

    except Exception as e:
        logger.error(f"[Error] Failed to load model {model_name}: {e}")
        raise


def save_model(model: torch.nn.Module, tokenizer, save_dir: str, source_model_name: str):
    """
    保存压缩后的模型和 tokenizer，不改变模型结构，仅保存权重。
    """
    try:
        os.makedirs(save_dir, exist_ok=True)

        # Save model weights + config
        model.save_pretrained(save_dir, torch_dtype=torch.float16)
        tokenizer.save_pretrained(save_dir)

        # Save tokenizer source for reload
        with open(os.path.join(save_dir, "tokenizer_source.txt"), "w") as f:
            f.write(source_model_name.strip())

        logger.info(f"✔ Model, tokenizer, and tokenizer_source.txt saved to {save_dir}")

    except Exception as e:
        logger.error(f"[Error] Failed to save model to {save_dir}: {e}")
        raise


def reload_compressed_model(model_dir: str, device: int = 0):
    """
    重新加载压缩后的模型和 tokenizer，假设模型结构未变，仅参数已压缩。
    """
    try:
        logger.info(f"Reloading compressed model from: {model_dir}")
        tokenizer_source_path = os.path.join(model_dir, "tokenizer_source.txt")

        if not os.path.exists(tokenizer_source_path):
            raise FileNotFoundError("Missing tokenizer_source.txt. Cannot reload tokenizer.")

        with open(tokenizer_source_path, "r") as f:
            tokenizer_source = f.read().strip()

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)

        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        model.to(f"cuda:{device}")
        logger.info(f"✔ Reloaded compressed model to cuda:{device} successfully.")

        model.eval()
        return model, tokenizer

    except Exception as e:
        logger.error(f"[Error] Failed to reload compressed model from {model_dir}: {e}")
        raise
