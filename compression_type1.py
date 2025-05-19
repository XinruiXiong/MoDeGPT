import torch
import logging

logger = logging.getLogger("MoDeGPT")

@torch.no_grad()
def compress_mlp(model, cov, keep_ratios, rank=None, ridge_lambda=1e-2, logger=None):
    """
    MoDeGPT Type-I Compression (MLP): Full Nyström approximation with ridge leverage scoring.
    """
    config = model.config
    n_layers = getattr(config, "n_layer", None) or getattr(config, "num_hidden_layers", None) or getattr(config, "num_layers", None)

    for i in range(n_layers):
        try:
            keep_ratio = keep_ratios[i]
            C = cov[i].float().to("cuda")  # [D_int, D_int]
            D_int = C.shape[0]
            r = int(D_int * keep_ratio) if rank is None else rank
            r = max(1, min(r, D_int))

            # Step 1: ridge leverage scores
            ridge = ridge_lambda * torch.eye(D_int, device=C.device)
            inv_term = torch.linalg.inv(C + ridge)
            ridge_proj = C @ inv_term
            scores = torch.diag(ridge_proj)

            # Step 2: top-k selection
            topk = torch.topk(scores, k=r, largest=True).indices
            Sk = torch.eye(D_int, device=C.device)[:, topk]  # [D_int, r]

            # Step 3: get weights
            try:
                block = model.model.decoder.layers[i]  # OPT
                W_u = block.fc1.weight  # [D_int, D_h]
                W_d = block.fc2.weight  # [D_h, D_int]
            except AttributeError:
                try:
                    block = model.transformer.h[i]  # GPT
                    W_u = block.mlp.c_fc.weight
                    W_d = block.mlp.c_proj.weight
                except AttributeError:
                    block = model.model.layers[i]  # LLaMA
                    W_u = block.mlp.gate_proj.weight
                    W_d = block.mlp.down_proj.weight

            W_u = W_u.to(torch.float32).to("cuda")  # [D_int, D_h]
            W_d = W_d.to(torch.float32).to("cuda")  # [D_h, D_int]

            # Step 4: Nyström reconstruction
            C_JJ = Sk.T @ C @ Sk  # [r, r]
            C_JJ_inv = torch.linalg.pinv(C_JJ)

            W_u_proj = Sk.T @ W_u  # [r, D_h]
            W_d_proj = W_d @ C @ Sk @ C_JJ_inv  # [D_h, r]

            # Step 5: write back
            W_u.data.zero_()
            W_u.data[:r, :] = W_u_proj.to(W_u.dtype).to(W_u.device)

            W_d.data.zero_()
            W_d.data[:, :r] = W_d_proj.to(W_d.dtype).to(W_d.device)

            if logger:
                logger.info(f"[MLP] ✅ Layer {i}: compressed to rank {r} (Nyström λ={ridge_lambda})")

            torch.cuda.empty_cache()

        except Exception as e:
            if logger:
                logger.warning(f"[MLP] Compression failed at layer {i}: {e}")
