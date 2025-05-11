# import torch
# import logging

# logger = logging.getLogger("MoDeGPT")

# @torch.no_grad()
# def compress_vo(model, cov=None, keep_ratios=None, rank=None,
#                 n_layers=None, n_heads=None, head_dim=None,
#                 ridge_lambda=1e-2, min_rank=16, max_condition_number=1e4,
#                 logger=None):
#     """
#     MoDeGPT Type-III Compression (VO): Nyström projection on v_proj and o_proj heads,
#     with full-size reconstruction to prevent shape mismatch.
#     """
#     for i in range(n_layers):
#         keep_ratio = keep_ratios[i]
#         rank_i = int(head_dim * keep_ratio) if rank is None else rank
#         rank_i = max(min_rank, min(rank_i, head_dim))

#         try:
#             block = model.model.layers[i]
#             W_v = block.self_attn.v_proj.weight  # DO NOT .to() here
#             W_o = block.self_attn.o_proj.weight
#         except Exception as e:
#             if logger: logger.warning(f"[VO] Layer {i}: cannot access v_proj/o_proj: {e}")
#             continue

#         for h in range(n_heads):
#             s, e = h * head_dim, (h + 1) * head_dim
#             try:
#                 # move only slices to float32 + cuda
#                 V_h = W_v[s:e, :].clone().float().to("cuda")  # [Hd, D]
#                 O_h = W_o[:, s:e].clone().float().to("cuda")  # [D, Hd]

#                 C = V_h @ V_h.T
#                 ridge = ridge_lambda * torch.eye(head_dim, device=C.device)
#                 C_reg = C + ridge

#                 C_JJ = C_reg[:rank_i, :rank_i]
#                 cond = torch.linalg.cond(C_JJ).item()
#                 if cond > max_condition_number:
#                     if logger: logger.warning(f"[VO] Layer {i} Head {h}: cond={cond:.1e}, skipping")
#                     continue

#                 S = C_reg[:, :rank_i] @ torch.linalg.pinv(C_JJ)
#                 V_proj = S.T @ V_h        # [r, D]
#                 O_proj = O_h @ S          # [D, r]

#                 # full-size reconstruction
#                 V_new = torch.zeros((head_dim, V_h.shape[1]), device="cuda", dtype=torch.float32)
#                 O_new = torch.zeros((O_h.shape[0], head_dim), device="cuda", dtype=torch.float32)
#                 V_new[:rank_i, :] = V_proj
#                 O_new[:, :rank_i] = O_proj

#                 # write back to original model weight (still float16)
#                 W_v[s:e, :].data.copy_(V_new.to(dtype=W_v.dtype, device=W_v.device))
#                 W_o[:, s:e].data.copy_(O_new.to(dtype=W_o.dtype, device=W_o.device))

#             except Exception as e:
#                 if logger: logger.warning(f"[VO] Layer {i} Head {h}: compression failed: {e}")

#         if logger: logger.info(f"[VO] ✅ Compressed layer {i} to rank {rank_i} per head (λ={ridge_lambda})")
#         torch.cuda.empty_cache()




import torch
import logging

logger = logging.getLogger("MoDeGPT")

@torch.no_grad()
def compress_vo(model, cov=None, keep_ratios=None, rank=None,
                n_layers=None, n_heads=None, head_dim=None,
                ridge_lambda=1e-2, min_rank=16, max_condition_number=1e4,
                logger=None):
    """
    MoDeGPT Type-III VO Compression: Nyström projection (with top-k selection) on v_proj/o_proj.
    Supports LLaMA and OPT architectures.
    """

    # Detect architecture
    if hasattr(model.model, "layers"):  # LLaMA
        get_block = lambda i: model.model.layers[i]
    elif hasattr(model.model, "decoder") and hasattr(model.model.decoder, "layers"):  # OPT
        get_block = lambda i: model.model.decoder.layers[i]
    else:
        raise ValueError("Unsupported model architecture")

    for i in range(n_layers):
        keep_ratio = keep_ratios[i]
        rank_i = int(head_dim * keep_ratio) if rank is None else rank
        rank_i = max(min_rank, min(rank_i, head_dim))

        try:
            block = get_block(i)
            W_v = block.self_attn.v_proj.weight  # [D, D]
            W_o = (block.self_attn.o_proj.weight
                   if hasattr(block.self_attn, "o_proj")
                   else block.self_attn.out_proj.weight)  # [D, D]
        except Exception as e:
            if logger:
                logger.warning(f"[VO] Layer {i}: cannot access v_proj/o_proj: {e}")
            continue

        for h in range(n_heads):
            s, e = h * head_dim, (h + 1) * head_dim

            try:
                V_h = W_v[s:e, :].clone().to(dtype=torch.float32, device="cuda")  # [Hd, D]
                O_h = W_o[:, s:e].clone().to(dtype=torch.float32, device="cuda")  # [D, Hd]

                C = V_h @ V_h.T  # [Hd, Hd]
                diag = torch.diag(C)
                topk = torch.topk(diag, k=rank_i, largest=True).indices

                C_JJ = C[topk][:, topk] + ridge_lambda * torch.eye(rank_i, device=C.device)
                cond = torch.linalg.cond(C_JJ).item()
                if cond > max_condition_number:
                    if logger:
                        logger.warning(f"[VO] Layer {i} Head {h}: cond={cond:.1e}, skipped.")
                    continue

                C_IJ = C[:, topk]
                pinv_CJJ = torch.linalg.pinv(C_JJ)
                S = C_IJ @ pinv_CJJ  # [Hd, r]

                V_proj = S.T @ V_h         # [r, D]
                O_proj = O_h @ S           # [D, r]

                clamp_val = 2.5
                V_proj = V_proj.clamp(-clamp_val, clamp_val)
                O_proj = O_proj.clamp(-clamp_val, clamp_val)
                

                V_new = torch.zeros_like(V_h)  # [Hd, D]
                O_new = torch.zeros_like(O_h)  # [D, Hd]
                V_new[topk, :] = V_proj
                O_new[:, topk] = O_proj

                # 原始未压缩的 VO 切片
                V_orig = W_v[s:e, :].to(dtype=torch.float32, device="cuda")
                O_orig = W_o[:, s:e].to(dtype=torch.float32, device="cuda")

                alpha = 0.9  # 可以调节，越接近1表示越相信新权重
                W_v[s:e, :].data.copy_((alpha * V_new + (1 - alpha) * V_orig).to(dtype=W_v.dtype, device=W_v.device))
                W_o[:, s:e].data.copy_((alpha * O_new + (1 - alpha) * O_orig).to(dtype=W_o.dtype, device=W_o.device))

            except Exception as e:
                if logger:
                    logger.warning(f"[VO] Layer {i} Head {h}: compression failed: {e}")

        if logger:
            logger.info(f"[VO] ✅ Compressed layer {i} to rank {rank_i} per head (λ={ridge_lambda})")

        torch.cuda.empty_cache()
