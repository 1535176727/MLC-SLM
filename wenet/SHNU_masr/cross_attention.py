import torch

import torch.nn as nn

class CrossAttentionFusion(nn.Module):
    def __init__(self, query_dim=1280, kv_dim=768, num_heads=8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=query_dim,
            kdim=kv_dim,
            vdim=kv_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm = nn.LayerNorm(query_dim)

    def forward(self, whisper_emb, mhubert_emb, whisper_mask=None, mhubert_mask=None):
        key_padding_mask = (mhubert_mask == 0) if mhubert_mask is not None else None
        if key_padding_mask is None:
            raise ValueError("mhubert_mask must be provided for M->W attention")
        attn_output, _ = self.cross_attn(
            query=whisper_emb,       # (B, T1, 1280)
            key=mhubert_emb,         # (B, T2, 768)
            value=mhubert_emb,       # (B, T2, 768)
            key_padding_mask=key_padding_mask
        )
        fused = self.norm(whisper_emb + attn_output)
        return fused

class BiCrossAttentionFusion(nn.Module):
    def __init__(self, whisper_dim=1280, mhubert_dim=768, num_heads=8):
        super().__init__()
        # whisper attends to mhubert
        self.cross_attn_w2m = nn.MultiheadAttention(
            embed_dim=whisper_dim,
            kdim=mhubert_dim,
            vdim=mhubert_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm_w2m = nn.LayerNorm(whisper_dim)

        # mhubert attends to whisper
        self.cross_attn_m2w = nn.MultiheadAttention(
            embed_dim=mhubert_dim,
            kdim=whisper_dim,
            vdim=whisper_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm_m2w = nn.LayerNorm(mhubert_dim)

    def forward(self, whisper_emb, mhubert_emb, whisper_mask=None, mhubert_mask=None):
        # whisper attends to mhubert
        key_padding_mask_m = (mhubert_mask==0) if mhubert_mask is not None else None
        if key_padding_mask_m is None:
            raise ValueError("mhubert_mask must be provided for M->W attention")
        attn_output_w2m, _ = self.cross_attn_w2m(
            query=whisper_emb,
            key=mhubert_emb,
            value=mhubert_emb,
            key_padding_mask=key_padding_mask_m
        )
        fused_whisper = self.norm_w2m(whisper_emb + attn_output_w2m)

        # mhubert attends to whisper
        key_padding_mask_w = (mhubert_mask == 0) if mhubert_mask is not None else None
        # Ensure that key_padding_mask_w is provided
        if key_padding_mask_w is None:
            raise ValueError("whisper_mask must be provided for W->M attention")
        attn_output_m2w, _ = self.cross_attn_m2w(
            query=mhubert_emb,
            key=whisper_emb,
            value=whisper_emb,
            key_padding_mask=key_padding_mask_w
        )
        fused_mhubert = self.norm_m2w(mhubert_emb + attn_output_m2w)

        return fused_whisper, fused_mhubert
    
class GatedCrossAttentionFusion(nn.Module):
    """
    Unidirectional Gated Cross-Attention: Whisper queries mHuBERT, based on the paper's formula.
    """
    def __init__(self, query_dim=1280, kv_dim=768, num_heads=8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=query_dim,
            kdim=kv_dim,
            vdim=kv_dim,
            num_heads=num_heads,
            batch_first=True
        )
        # --- ADDED: The linear layer (Wg) for the gate, as described in the paper ---
        self.gating_linear = nn.Linear(query_dim, query_dim)
        self.norm = nn.LayerNorm(query_dim)

    def forward(self, whisper_emb, mhubert_emb, whisper_mask=None, mhubert_mask=None):
        # The key_padding_mask in PyTorch should be True for positions to be masked.
        key_padding_mask = (mhubert_mask == 0) if mhubert_mask is not None else None
        if key_padding_mask is None:
            raise ValueError("mhubert_mask must be provided for M->W attention")
        # This is Hca = Cross-Attention(Q=Hs, K=Hp, V=Hp) [cite: 96]
        attn_output, _ = self.cross_attn(
            query=whisper_emb,
            key=mhubert_emb,
            value=mhubert_emb,
            key_padding_mask=key_padding_mask
        )

        # --- MODIFIED: Implement the gating mechanism from Equation (2) ---
        # 1. Calculate the gate: gate = σ(Wg * Hca)
        gate = torch.sigmoid(self.gating_linear(attn_output))
        
        # 2. Apply the gate and add the residual connection: Ho = gate * Hca + Hs
        # This corresponds to σ(WgHca) ⊙ Hca + Hs 
        gated_attn_output = gate * attn_output
        
        # 3. Apply LayerNorm after the residual connection
        fused = self.norm(whisper_emb + gated_attn_output)
        
        return fused
    


class GatedBiCrossAttentionFusion(nn.Module):
    """
    Bidirectional Gated Cross-Attention: W -> M and M -> W.
    """
    def __init__(self, whisper_dim=1280, mhubert_dim=768, num_heads=8):
        super().__init__()
        # --- Path 1: whisper attends to mhubert ---
        self.cross_attn_w2m = nn.MultiheadAttention(
            embed_dim=whisper_dim, kdim=mhubert_dim, vdim=mhubert_dim,
            num_heads=num_heads, batch_first=True
        )
        self.gating_linear_w2m = nn.Linear(whisper_dim, whisper_dim) # Gate for the W->M path
        self.norm_w2m = nn.LayerNorm(whisper_dim)

        # --- Path 2: mhubert attends to whisper ---
        self.cross_attn_m2w = nn.MultiheadAttention(
            embed_dim=mhubert_dim, kdim=whisper_dim, vdim=whisper_dim,
            num_heads=num_heads, batch_first=True
        )
        self.gating_linear_m2w = nn.Linear(mhubert_dim, mhubert_dim) # Gate for the M->W path
        self.norm_m2w = nn.LayerNorm(mhubert_dim)

    def forward(self, whisper_emb, mhubert_emb, whisper_mask=None, mhubert_mask=None):
        # --- Path 1: whisper attends to mhubert ---
        key_padding_mask_m = (mhubert_mask == 0) if mhubert_mask is not None else None
        if key_padding_mask_m is None:
            raise ValueError("mhubert_mask must be provided for M->W attention")
        attn_output_w2m, _ = self.cross_attn_w2m(
            query=whisper_emb,
            key=mhubert_emb,
            value=mhubert_emb,
            key_padding_mask=key_padding_mask_m
        )
        # Apply gating
        gate_w2m = torch.sigmoid(self.gating_linear_w2m(attn_output_w2m))
        fused_whisper = self.norm_w2m(whisper_emb + (gate_w2m * attn_output_w2m))

        # --- Path 2: mhubert attends to whisper ---
        key_padding_mask_w = (mhubert_mask == 0) if mhubert_mask is not None else None
        if key_padding_mask_w is None:
            raise ValueError("mhubert_mask must be provided for M->W attention")

        attn_output_m2w, _ = self.cross_attn_m2w(
            query=mhubert_emb,
            key=whisper_emb,
            value=whisper_emb,
            key_padding_mask=key_padding_mask_w
        )
        # Apply gating
        gate_m2w = torch.sigmoid(self.gating_linear_m2w(attn_output_m2w))
        fused_mhubert = self.norm_m2w(mhubert_emb + (gate_m2w * attn_output_m2w))

        return fused_whisper, fused_mhubert
    

class GatedCrossAttentionFusionWithConcatenation(nn.Module):
    """
    Unidirectional Gated Cross-Attention with concatenation: Whisper queries mHuBERT.
    """
    def __init__(self, query_dim=1280, kv_dim=768, num_heads=8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=query_dim,
            kdim=kv_dim,
            vdim=kv_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.gating_linear = nn.Linear(query_dim + kv_dim, query_dim)  # Adjusted for concatenation
        self.norm = nn.LayerNorm(query_dim)

    def forward(self, whisper_emb, mhubert_emb, whisper_mask=None, mhubert_mask=None):
        key_padding_mask = (mhubert_mask == 0) if mhubert_mask is not None else None
        if key_padding_mask is None:
            raise ValueError("mhubert_mask must be provided for M->W attention")
        
        attn_output, _ = self.cross_attn(
            query=whisper_emb,
            key=mhubert_emb,
            value=mhubert_emb,
            key_padding_mask=key_padding_mask
        )

        # Concatenate the original whisper embedding with the attention output
        concat_output = torch.cat((whisper_emb, attn_output), dim=-1)
        
        # Apply the gating mechanism
        gate = torch.sigmoid(self.gating_linear(concat_output))
        
        # Apply the gate and add the residual connection
        gated_attn_output = gate * attn_output
        
        fused = self.norm(whisper_emb + gated_attn_output)
        
        return fused
    


class ConcatCrossAttentionFusion(nn.Module):
    """
    Unidirectional Concatenation-based Cross-Attention: W -> M.
    Fuses information from mHuBERT into the Whisper embedding.
    """
    def __init__(self, whisper_dim=1280, mhubert_dim=768, num_heads=8):
        super().__init__()
        # --- Whisper attends to mHuBERT ---
        self.cross_attn_w2m = nn.MultiheadAttention(
            embed_dim=whisper_dim,
            kdim=mhubert_dim,
            vdim=mhubert_dim,
            num_heads=num_heads,
            batch_first=True
        )

    def forward(self, whisper_emb, mhubert_emb, mhubert_mask=None):
        """
        Args:
            whisper_emb (Tensor): Whisper embeddings (B, T_w, D_w)
            mhubert_emb (Tensor): mHuBERT embeddings (B, T_m, D_m)
            mhubert_mask (Tensor): mHuBERT padding mask (B, T_m)

        Returns:
            Tensor: Fused whisper embeddings (B, T_w, D_w)
        """
        key_padding_mask_m = (mhubert_mask == 0) if mhubert_mask is not None else None

        attn_output_w2m, _ = self.cross_attn_w2m(
            query=whisper_emb,
            key=mhubert_emb,
            value=mhubert_emb,
            key_padding_mask=key_padding_mask_m
        )

        concatenated_features = torch.cat([whisper_emb, mhubert_emb,attn_output_w2m], dim=-1)

        #dim = 1280+768+1280 = 3328

        return concatenated_features
    

class ConcatBiCrossAttentionFusion(nn.Module):
    """
    Bidirectional Concatenation-based Cross-Attention: W -> M and M -> W.
    """
    def __init__(self, whisper_dim=1280, mhubert_dim=768, num_heads=8):
        super().__init__()
        # --- Path 1: whisper attends to mhubert ---
        self.cross_attn_w2m = nn.MultiheadAttention(
            embed_dim=whisper_dim, kdim=mhubert_dim, vdim=mhubert_dim,
            num_heads=num_heads, batch_first=True
        )

        # --- Path 2: mhubert attends to whisper ---
        self.cross_attn_m2w = nn.MultiheadAttention(
            embed_dim=mhubert_dim, kdim=whisper_dim, vdim=whisper_dim,
            num_heads=num_heads, batch_first=True
        )

    def forward(self, whisper_emb, mhubert_emb, whisper_mask=None, mhubert_mask=None):
        """
        Args:
            whisper_emb (Tensor): Whisper embeddings (B, T_w, D_w)
            mhubert_emb (Tensor): mHuBERT embeddings (B, T_m, D_m)
            whisper_mask (Tensor): Whisper padding mask (B, T_w)
            mhubert_mask (Tensor): mHuBERT padding mask (B, T_m)

        Returns:
            Tuple[Tensor, Tensor]: Fused whisper and mHuBERT embeddings.
        """
        # --- Path 1: whisper attends to mhubert ---
        key_padding_mask_m = (mhubert_mask == 0) if mhubert_mask is not None else None
        attn_output_w2m, _ = self.cross_attn_w2m(
            query=whisper_emb,
            key=mhubert_emb,
            value=mhubert_emb,
            key_padding_mask=key_padding_mask_m
        )
        concatenated_w = torch.cat([whisper_emb, attn_output_w2m], dim=-1)

        # --- Path 2: mhubert attends to whisper ---
        attn_output_m2w, _ = self.cross_attn_m2w(
            query=mhubert_emb,
            key=whisper_emb,
            value=whisper_emb,
            key_padding_mask=key_padding_mask_m
        )
        concatenated = torch.cat([whisper_emb,mhubert_emb, attn_output_w2m,attn_output_m2w], dim=-1)
        #dim = 1280+768+1280+768 = 4096
        return concatenated
    


class GatedFusionBlock(nn.Module):
    """
         Self-Attention (on A) -> Gated Cross-Attention (A queries B) -> Feed-Forward Network.
    """
    def __init__(self, dim_a, dim_b, num_heads=8, ffn_expansion=4):
        super().__init__()
        # 1. Self-Attention for sequence A (the Query sequence)
        self.self_attn = nn.MultiheadAttention(dim_a, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim_a)
        
        # 2. Cross-Attention where A queries B
        self.cross_attn = nn.MultiheadAttention(dim_a, num_heads, kdim=dim_b, vdim=dim_b, batch_first=True)
        self.gating_linear = nn.Linear(dim_a, dim_a) # Gate for the cross-attention output
        self.norm2 = nn.LayerNorm(dim_a)
        
        # 3. Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(dim_a, dim_a * ffn_expansion),
            nn.GELU(),
            nn.Linear(dim_a * ffn_expansion, dim_a),
        )
        self.norm3 = nn.LayerNorm(dim_a)

    def forward(self, x_a, x_b, x_a_mask=None, x_b_mask=None):
        """
        Args:
            x_a (torch.Tensor): Query序列, e.g., whisper_emb.
            x_b (torch.Tensor): Key/Value序列, e.g., mhubert_emb.
            x_a_mask (torch.Tensor, optional): x_a的padding mask.
            x_b_mask (torch.Tensor, optional): x_b的padding mask.
        """
        # --- Step 1: Self-Attention on x_a ---
        self_attn_padding_mask = (x_a_mask == 0) if x_a_mask is not None else None
        attn_output, _ = self.self_attn(x_a, x_a, x_a, key_padding_mask=self_attn_padding_mask)
        x_a = self.norm1(x_a + attn_output)
        
        # --- Step 2: Gated Cross-Attention where x_a queries x_b ---
        cross_attn_padding_mask = (x_b_mask == 0) if x_b_mask is not None else None
        attn_output, _ = self.cross_attn(x_a, x_b, x_b, key_padding_mask=cross_attn_padding_mask)
        
        # Apply gating mechanism
        gate = torch.sigmoid(self.gating_linear(attn_output))
        gated_output = gate * attn_output
        
        # Residual connection
        x_a = self.norm2(x_a + gated_output)
        
        # --- Step 3: Feed-Forward Network ---
        ffn_output = self.ffn(x_a)
        x_a = self.norm3(x_a + ffn_output)
        
        return x_a

class DoubleLayeredGatedBiCrossAttention(nn.Module):

    def __init__(self, whisper_dim=1280, mhubert_dim=768, num_heads=8, num_layers=2):
        super().__init__()
        
        # Path 1: W -> M fusion pathway (2 layers)
        self.w2m_fusion_layers = nn.ModuleList(
            [GatedFusionBlock(whisper_dim, mhubert_dim, num_heads) for _ in range(num_layers)]
        )
        
        # Path 2: M -> W fusion pathway (2 layers)
        self.m2w_fusion_layers = nn.ModuleList(
            [GatedFusionBlock(mhubert_dim, whisper_dim, num_heads) for _ in range(num_layers)]
        )
        
        # Final concatenation will be done outside, after getting the two outputs.

    def forward(self, whisper_emb, mhubert_emb, whisper_mask=None, mhubert_mask=None):
        # Create copies to be refined
        fused_whisper = whisper_emb
        fused_mhubert = mhubert_emb

        # --- Sequentially apply the W -> M fusion layers ---
        # The query (fused_whisper) gets refined at each layer, 
        # while the key/value source (mhubert_emb) remains the original.
        for layer in self.w2m_fusion_layers:
            fused_whisper = layer(fused_whisper, mhubert_emb, whisper_mask, mhubert_mask)
            
        # --- Sequentially apply the M -> W fusion layers ---
        # The query (fused_mhubert) gets refined at each layer, 
        # while the key/value source (whisper_emb) remains the original.
        for layer in self.m2w_fusion_layers:
            fused_mhubert = layer(fused_mhubert, whisper_emb, mhubert_mask, whisper_mask)
            
        return fused_whisper, fused_mhubert




class GatedBiCrossAttentionWithConcatenation(nn.Module):
    """
    Bidirectional Gated Cross-Attention: W -> M and M -> W.
    """
    def __init__(self, whisper_dim=1280, mhubert_dim=768, num_heads=8):
        super().__init__()
        # --- Path 1: whisper attends to mhubert ---
        self.cross_attn_w2m = nn.MultiheadAttention(
            embed_dim=whisper_dim, kdim=mhubert_dim, vdim=mhubert_dim,
            num_heads=num_heads, batch_first=True
        )
        self.gating_linear_w2m = nn.Linear(whisper_dim, whisper_dim) # Gate for the W->M path
        self.norm_w2m = nn.LayerNorm(whisper_dim)

        # --- Path 2: mhubert attends to whisper ---
        self.cross_attn_m2w = nn.MultiheadAttention(
            embed_dim=mhubert_dim, kdim=whisper_dim, vdim=whisper_dim,
            num_heads=num_heads, batch_first=True
        )
        self.gating_linear_m2w = nn.Linear(mhubert_dim, mhubert_dim) # Gate for the M->W path
        self.norm_m2w = nn.LayerNorm(mhubert_dim)

    def forward(self, whisper_emb, mhubert_emb, whisper_mask=None, mhubert_mask=None):
        # --- Path 1: whisper attends to mhubert ---
        key_padding_mask_m = (mhubert_mask == 0) if mhubert_mask is not None else None
        if key_padding_mask_m is None:
            raise ValueError("mhubert_mask must be provided for M->W attention")
        attn_output_w2m, _ = self.cross_attn_w2m(
            query=whisper_emb,
            key=mhubert_emb,
            value=mhubert_emb,
            key_padding_mask=key_padding_mask_m
        )
        # Apply gating
        gate_w2m = torch.sigmoid(self.gating_linear_w2m(attn_output_w2m))
        fused_whisper = self.norm_w2m(whisper_emb + (gate_w2m * attn_output_w2m))

        # --- Path 2: mhubert attends to whisper ---
        key_padding_mask_w = (mhubert_mask == 0) if mhubert_mask is not None else None
        if key_padding_mask_w is None:
            raise ValueError("mhubert_mask must be provided for M->W attention")

        attn_output_m2w, _ = self.cross_attn_m2w(
            query=mhubert_emb,
            key=whisper_emb,
            value=whisper_emb,
            key_padding_mask=key_padding_mask_w
        )
        # Apply gating
        gate_m2w = torch.sigmoid(self.gating_linear_m2w(attn_output_m2w))
        fused_mhubert = self.norm_m2w(mhubert_emb + (gate_m2w * attn_output_m2w))
        # --- Concatenation ---
        fused = torch.cat([whisper_emb, mhubert_emb, fused_whisper, fused_mhubert], dim=-1)

        return fused