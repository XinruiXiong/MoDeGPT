# compression_type1.py

# import torch
# import logging

# logger = logging.getLogger("MoDeGPT")

# @torch.no_grad()
# def compress_mlp(model, cov, keep_ratios, rank=None, logger=None):
#     n_layers = len(keep_ratios)

#     for i in range(n_layers):
#         try:
#             # 把cov搬到GPU加速
#             cov_i = cov[i].to(device="cuda", dtype=torch.float32)
#             rank_i = int(cov_i.size(0) * keep_ratios[i]) if rank is None else rank

#             # GPU上跑SVD
#             U, _, _ = torch.linalg.svd(cov_i, full_matrices=False)
#             U_r = U[:, :rank_i]
#             P = U_r @ U_r.T  # [D, D]

#             # Locate MLP weights
#             try:
#                 block = model.model.decoder.layers[i]  # OPT
#                 W1 = block.fc1.weight.data
#                 W2 = block.fc2.weight.data
#             except AttributeError:
#                 try:
#                     block = model.transformer.h[i]  # GPT
#                     W1 = block.mlp.c_fc.weight.data
#                     W2 = block.mlp.c_proj.weight.data
#                 except AttributeError:
#                     block = model.model.layers[i]  # LLaMA
#                     W1 = block.mlp.gate_proj.weight.data
#                     W2 = block.mlp.down_proj.weight.data

#             # 注意：因为P在cuda，W1和W2在cpu，所以要搬动一下
#             W1_proj = (P @ W1.to(device="cuda", dtype=torch.float32)).to(W1.dtype).cpu()
#             W2_proj = (W2.to(device="cuda", dtype=torch.float32) @ P).to(W2.dtype).cpu()

#             W1.copy_(W1_proj)
#             W2.copy_(W2_proj)

#             if logger:
#                 logger.info(f"[MLP] Compressed layer {i} to rank {rank_i}")

#             # 每压一层手动释放内存
#             del cov_i, U, U_r, P
#             torch.cuda.empty_cache()

#         except Exception as e:
#             if logger:
#                 logger.warning(f"[MLP] Compression failed at layer {i}: {e}")



import torch
import logging

logger = logging.getLogger("MoDeGPT")

@torch.no_grad()
def compress_mlp(model, cov, keep_ratios, rank=None, ridge_lambda=1e-2, logger=None):
    """
    MoDeGPT Type-I Compression (MLP): Nyström projection with ridge leverage scores.

    Args:
        model: HuggingFace model object
        cov: List of covariance matrices C_σ (per layer), each of shape [H, H]
        keep_ratios: List[float], layer-wise keep ratio
        rank: Optional fixed rank override
        ridge_lambda: float, regularization strength
        logger: Logger instance
    """
    config = model.config
    n_layers = getattr(config, "n_layer", None) or getattr(config, "num_hidden_layers", None) or getattr(config, "num_layers", None)

    for i in range(n_layers):
        try:
            keep_ratio = keep_ratios[i]
            C = cov[i].float().to("cuda")  # [H, H]
            H = C.shape[0]
            r = int(H * keep_ratio) if rank is None else rank
            r = max(1, min(r, H))

            # Step 1: Ridge leverage score selection (top diag)
            diag = torch.diag(C)
            topk = torch.topk(diag, k=r, largest=True).indices
            rest = torch.tensor([j for j in range(H) if j not in topk], dtype=torch.long, device=topk.device)

            # Step 2: Compute Nyström projection matrix
            C_JJ = C[topk][:, topk]  # [r, r]
            ridge = ridge_lambda * torch.eye(r, device=C_JJ.device)
            S_inv = torch.linalg.pinv(C_JJ + ridge)
            S = C[:, topk] @ S_inv  # [H, r]

            # Step 3: Get MLP weights
            try:
                # OPT
                block = model.model.decoder.layers[i]
                W_u = block.fc1.weight  # [H, D]
                W_d = block.fc2.weight  # [D, H]
            except AttributeError:
                try:
                    # GPT
                    block = model.transformer.h[i]
                    W_u = block.mlp.c_fc.weight
                    W_d = block.mlp.c_proj.weight
                except AttributeError:
                    # LLaMA
                    block = model.model.layers[i]
                    W_u = block.mlp.gate_proj.weight
                    W_d = block.mlp.down_proj.weight

            # Step 4: Apply Nyström projection
            W_u = W_u.to(dtype=torch.float32, device="cuda")  # [H, D]
            W_d = W_d.to(dtype=torch.float32, device="cuda")  # [D, H]

            # 修复关键点：S 是 [H, r]，因此应转置后用于左乘
            W_u_proj = S.T @ W_u  # [r, H] x [H, D] -> [r, D]
            W_d_proj = W_d @ S    # [D, H] x [H, r] -> [D, r]

            # Step 5: Patch weights
            W_u.data.zero_()
            W_u.data[:r, :] = W_u_proj.to(W_u.dtype)

            W_d.data.zero_()
            W_d.data[:, :r] = W_d_proj.to(W_d.dtype)

            if logger:
                logger.info(f"[MLP] Compressed layer {i} to rank {r} (Nyström, λ={ridge_lambda})")

            torch.cuda.empty_cache()

        except Exception as e:
            if logger:
                logger.warning(f"[MLP] Compression failed at layer {i}: {e}")
