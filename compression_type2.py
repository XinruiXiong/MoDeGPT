# # compression_type2.py

# import torch
# import logging

# logger = logging.getLogger("MoDeGPT")

# @torch.no_grad()
# def compress_qk(model, cov, keep_ratios, rank=None, n_layers=None, n_heads=None, head_dim=None, logger=None):
#     cov_q_list, cov_k_list = cov

#     for i in range(n_layers):
#         try:
#             keep_ratio = keep_ratios[i]
#             rank_i = int(head_dim * keep_ratio) if rank is None else rank

#             # Get attention weights
#             try:
#                 block = model.model.decoder.layers[i]  # OPT
#                 W_q = block.self_attn.q_proj.weight.data
#                 W_k = block.self_attn.k_proj.weight.data
#             except AttributeError:
#                 try:
#                     block = model.transformer.h[i]  # GPT
#                     W_q = block.attn.c_attn.weight.data
#                     W_k = W_q  # packed
#                 except AttributeError:
#                     block = model.model.layers[i]  # LLaMA
#                     W_q = block.self_attn.q_proj.weight.data
#                     W_k = block.self_attn.k_proj.weight.data

#             for h in range(n_heads):
#                 h_start = h * head_dim
#                 h_end = (h + 1) * head_dim
#                 cov_q = cov_q_list[i][h].float()
#                 cov_k = cov_k_list[i][h].float()

#                 diag = torch.diag(cov_q + cov_k)
#                 topk = torch.topk(diag, k=rank_i, largest=True).indices
#                 rest = torch.tensor([j for j in range(head_dim) if j not in topk], dtype=torch.long)

#                 # Q projection
#                 cov_q_JJ = cov_q[topk][:, topk]
#                 alpha_q = cov_q[rest][:, topk] @ torch.linalg.pinv(cov_q_JJ)
#                 W_q_h = W_q[h_start:h_end, :].float()
#                 W_q_new = torch.zeros_like(W_q_h)
#                 W_q_new[topk, :] = W_q_h[topk, :]
#                 if len(rest) > 0:
#                     W_q_new[rest, :] = alpha_q @ W_q_h[topk, :]
#                 W_q[h_start:h_end, :] = W_q_new.to(W_q.dtype)

#                 # K projection
#                 cov_k_JJ = cov_k[topk][:, topk]
#                 alpha_k = cov_k[rest][:, topk] @ torch.linalg.pinv(cov_k_JJ)
#                 W_k_h = W_k[h_start:h_end, :].float()
#                 W_k_new = torch.zeros_like(W_k_h)
#                 W_k_new[topk, :] = W_k_h[topk, :]
#                 if len(rest) > 0:
#                     W_k_new[rest, :] = alpha_k @ W_k_h[topk, :]
#                 W_k[h_start:h_end, :] = W_k_new.to(W_k.dtype)

#             if logger:
#                 logger.info(f"[QK] Compressed layer {i} to rank {rank_i}")
#         except Exception as e:
#             if logger:
#                 logger.warning(f"[QK] Compression failed at layer {i}: {e}")



import torch
import logging

logger = logging.getLogger("MoDeGPT")

@torch.no_grad()
def compress_qk(model, cov, keep_ratios, rank=None, n_layers=None, n_heads=None, head_dim=None, ridge_lambda=1e-2, logger=None):
    """
    MoDeGPT Type-II Compression (Q/K): Ridge-leverage-based head-wise Nyström projection.

    Args:
        model: HuggingFace model
        cov: Tuple of (cov_q_list, cov_k_list), each is List[List[Tensor (H x H)]]
        keep_ratios: List[float], layer-wise keep ratio
        rank: Optional fixed per-head rank
        n_layers: Total number of transformer layers
        n_heads: Total attention heads per layer
        head_dim: Dimension per head
        ridge_lambda: Regularization strength for (C_JJ + λI)^-1
        logger: Logger instance
    """
    cov_q_list, cov_k_list = cov

    for i in range(n_layers):
        try:
            keep_ratio = keep_ratios[i]
            rank_i = int(head_dim * keep_ratio) if rank is None else rank
            rank_i = max(1, min(rank_i, head_dim))

            # === Get Q, K weight reference ===
            try:
                # OPT
                block = model.model.decoder.layers[i]
                W_q = block.self_attn.q_proj.weight
                W_k = block.self_attn.k_proj.weight
            except AttributeError:
                try:
                    # GPT
                    block = model.transformer.h[i]
                    W_q = block.attn.c_attn.weight
                    W_k = W_q  # packed QKV
                    raise NotImplementedError("Packed QKV (GPT) is not yet supported in MoDeGPT QK compression.")
                except AttributeError:
                    # LLaMA
                    block = model.model.layers[i]
                    W_q = block.self_attn.q_proj.weight
                    W_k = block.self_attn.k_proj.weight

            for h in range(n_heads):
                h_start = h * head_dim
                h_end = (h + 1) * head_dim

                cov_q = cov_q_list[i][h].float().to("cuda")  # [H, H]
                cov_k = cov_k_list[i][h].float().to("cuda")  # [H, H]
                cov_sum = cov_q + cov_k  # [H, H]

                # === Select top-k indices by leverage scores ===
                diag = torch.diag(cov_sum)
                topk = torch.topk(diag, k=rank_i, largest=True).indices
                rest = torch.tensor([j for j in range(head_dim) if j not in topk], dtype=torch.long, device=topk.device)

                # === Compute projection matrix for Q ===
                C_q_JJ = cov_q[topk][:, topk]
                ridge_q = ridge_lambda * torch.eye(rank_i, device=C_q_JJ.device)
                pinv_q = torch.linalg.pinv(C_q_JJ + ridge_q)
                alpha_q = cov_q[rest][:, topk] @ pinv_q  # [|rest|, r]

                W_q_h = W_q[h_start:h_end, :].float().to("cuda")  # [H, D]
                W_q_new = torch.zeros_like(W_q_h)
                W_q_new[topk, :] = W_q_h[topk, :]
                if len(rest) > 0:
                    W_q_new[rest, :] = alpha_q @ W_q_h[topk, :]
                W_q[h_start:h_end, :].data.copy_(W_q_new.to(W_q.dtype))

                # === Compute projection matrix for K ===
                C_k_JJ = cov_k[topk][:, topk]
                ridge_k = ridge_lambda * torch.eye(rank_i, device=C_k_JJ.device)
                pinv_k = torch.linalg.pinv(C_k_JJ + ridge_k)
                alpha_k = cov_k[rest][:, topk] @ pinv_k

                W_k_h = W_k[h_start:h_end, :].float().to("cuda")
                W_k_new = torch.zeros_like(W_k_h)
                W_k_new[topk, :] = W_k_h[topk, :]
                if len(rest) > 0:
                    W_k_new[rest, :] = alpha_k @ W_k_h[topk, :]
                W_k[h_start:h_end, :].data.copy_(W_k_new.to(W_k.dtype))

            if logger:
                logger.info(f"[QK] Compressed layer {i} to rank {rank_i} per head (ridge λ={ridge_lambda})")

        except Exception as e:
            if logger:
                logger.warning(f"[QK] Compression failed at layer {i}: {e}")
