from typing import *
import torch
import torch.nn as nn
from ..basic import SparseTensor
from ..attention import SparseMultiHeadAttention, SerializeMode
from ...norm import LayerNorm32
from .blocks import SparseFeedForwardNet


class ModulatedSparseTransformerBlock(nn.Module):
    """
    Sparse Transformer block (MSA + FFN) with adaptive layer norm conditioning.
    """
    def __init__(
        self,
        channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "shift_window", "shift_sequence", "shift_order", "swin"] = "full",
        window_size: Optional[int] = None,
        shift_sequence: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        serialize_mode: Optional[SerializeMode] = None,
        use_checkpoint: bool = False,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
        qkv_bias: bool = True,
        share_mod: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.attn = SparseMultiHeadAttention(
            channels,
            num_heads=num_heads,
            attn_mode=attn_mode,
            window_size=window_size,
            shift_sequence=shift_sequence,
            shift_window=shift_window,
            serialize_mode=serialize_mode,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            qk_rms_norm=qk_rms_norm,
        )
        self.mlp = SparseFeedForwardNet(
            channels,
            mlp_ratio=mlp_ratio,
        )
        if not share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(channels, 6 * channels, bias=True)
            )

    def _forward(self, x: SparseTensor, mod: torch.Tensor) -> SparseTensor:
        if self.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(mod).chunk(6, dim=1)
        h = x.replace(self.norm1(x.feats))
        h = h * (1 + scale_msa) + shift_msa
        h = self.attn(h)
        h = h * gate_msa
        x = x + h
        h = x.replace(self.norm2(x.feats))
        h = h * (1 + scale_mlp) + shift_mlp
        h = self.mlp(h)
        h = h * gate_mlp
        x = x + h
        return x

    def forward(self, x: SparseTensor, mod: torch.Tensor) -> SparseTensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, mod, use_reentrant=False)
        else:
            return self._forward(x, mod)


class ModulatedSparseTransformerCrossBlock(nn.Module):
    """
    Sparse Transformer cross-attention block (MSA + MCA + FFN) with adaptive layer norm conditioning.
    """
    def __init__(
        self,
        channels: int,
        ctx_channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "shift_window", "shift_sequence", "shift_order", "swin"] = "full",
        window_size: Optional[int] = None,
        shift_sequence: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        serialize_mode: Optional[SerializeMode] = None,
        use_checkpoint: bool = False,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        qkv_bias: bool = True,
        share_mod: bool = False,

    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm3 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.self_attn = SparseMultiHeadAttention(
            channels,
            num_heads=num_heads,
            type="self",
            attn_mode=attn_mode,
            window_size=window_size,
            shift_sequence=shift_sequence,
            shift_window=shift_window,
            serialize_mode=serialize_mode,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            qk_rms_norm=qk_rms_norm,
        )
        self.cross_attn = SparseMultiHeadAttention(
            channels,
            ctx_channels=ctx_channels,
            num_heads=num_heads,
            type="cross",
            attn_mode="full",
            qkv_bias=qkv_bias,
            qk_rms_norm=qk_rms_norm_cross,
        )
        self.mlp = SparseFeedForwardNet(
            channels,
            mlp_ratio=mlp_ratio,
        )
        if not share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(channels, 6 * channels, bias=True)
            )
    
    def _forward_style(self, x0: SparseTensor, x1: SparseTensor, x2: SparseTensor, x3: SparseTensor, mod: torch.Tensor, context1: torch.Tensor, context2: torch.Tensor, context3: torch.Tensor):
        if self.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(mod).chunk(6, dim=1)
            
        h0 = x0.replace(self.norm1(x0.feats))
        h0 = h0 * (1 + scale_msa) + shift_msa
        
        h1 = x1.replace(self.norm1(x1.feats))
        h1 = h1 * (1 + scale_msa) + shift_msa
        
        h2 = x2.replace(self.norm1(x2.feats))
        h2 = h2 * (1 + scale_msa) + shift_msa
        
        h3 = x3.replace(self.norm1(x3.feats))
        h3 = h3 * (1 + scale_msa) + shift_msa
        
        h0, h1, h2, h3 = self.self_attn.forward_style(h0, h1, h2, h3)
        
        h0 = h0 * gate_msa
        x0 = x0 + h0
        h0 = x0.replace(self.norm2(x0.feats))
        h0 = self.cross_attn(h0, context1)
        x0 = x0 + h0
        h0 = x0.replace(self.norm3(x0.feats))
        h0 = h0 * (1 + scale_mlp) + shift_mlp
        h0 = self.mlp(h0)
        h0 = h0 * gate_mlp
        x0 = x0 + h0
        
        h1 = h1 * gate_msa
        x1 = x1 + h1
        h1 = x1.replace(self.norm2(x1.feats))
        h1 = self.cross_attn(h1, context1)
        x1 = x1 + h1
        h1 = x1.replace(self.norm3(x1.feats))
        h1 = h1 * (1 + scale_mlp) + shift_mlp
        h1 = self.mlp(h1)
        h1 = h1 * gate_mlp
        x1 = x1 + h1
        
        h2 = h2 * gate_msa
        x2 = x2 + h2
        h2 = x2.replace(self.norm2(x2.feats))
        h2 = self.cross_attn(h2, context2)
        x2 = x2 + h2
        h2 = x2.replace(self.norm3(x2.feats))
        h2 = h2 * (1 + scale_mlp) + shift_mlp
        h2 = self.mlp(h2)
        h2 = h2 * gate_mlp
        x2 = x2 + h2
        
        h3 = h3 * gate_msa
        x3 = x3 + h3
        h3 = x3.replace(self.norm2(x3.feats))
        h3 = self.cross_attn(h3, context3)
        x3 = x3 + h3
        h3 = x3.replace(self.norm3(x3.feats))
        h3 = h3 * (1 + scale_mlp) + shift_mlp
        h3 = self.mlp(h3)
        h3 = h3 * gate_mlp
        x3 = x3 + h3
        
        return x0, x1, x2, x3
    
    
    
    def _forward_color(self, x0: SparseTensor, x1: SparseTensor, x2: SparseTensor, x3: SparseTensor, mod: torch.Tensor, context1: torch.Tensor, context2: torch.Tensor, context3: torch.Tensor):
        if self.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(mod).chunk(6, dim=1)
            
        h0 = x0.replace(self.norm1(x0.feats))
        h0 = h0 * (1 + scale_msa) + shift_msa
        
        h1 = x1.replace(self.norm1(x1.feats))
        h1 = h1 * (1 + scale_msa) + shift_msa
        
        h2 = x2.replace(self.norm1(x2.feats))
        h2 = h2 * (1 + scale_msa) + shift_msa
        
        h3 = x3.replace(self.norm1(x3.feats))
        h3 = h3 * (1 + scale_msa) + shift_msa
        
        h0, h1, h2, h3 = self.self_attn.forward_color(h0, h1, h2, h3)
        
        h0 = h0 * gate_msa
        x0 = x0 + h0
        h0 = x0.replace(self.norm2(x0.feats))
        h0 = self.cross_attn(h0, context1)
        x0 = x0 + h0
        h0 = x0.replace(self.norm3(x0.feats))
        h0 = h0 * (1 + scale_mlp) + shift_mlp
        h0 = self.mlp(h0)
        h0 = h0 * gate_mlp
        x0 = x0 + h0
        
        h1 = h1 * gate_msa
        x1 = x1 + h1
        h1 = x1.replace(self.norm2(x1.feats))
        h1 = self.cross_attn(h1, context1)
        x1 = x1 + h1
        h1 = x1.replace(self.norm3(x1.feats))
        h1 = h1 * (1 + scale_mlp) + shift_mlp
        h1 = self.mlp(h1)
        h1 = h1 * gate_mlp
        x1 = x1 + h1
        
        h2 = h2 * gate_msa
        x2 = x2 + h2
        h2 = x2.replace(self.norm2(x2.feats))
        h2 = self.cross_attn(h2, context2)
        x2 = x2 + h2
        h2 = x2.replace(self.norm3(x2.feats))
        h2 = h2 * (1 + scale_mlp) + shift_mlp
        h2 = self.mlp(h2)
        h2 = h2 * gate_mlp
        x2 = x2 + h2
        
        h3 = h3 * gate_msa
        x3 = x3 + h3
        h3 = x3.replace(self.norm2(x3.feats))
        h3 = self.cross_attn(h3, context3)
        x3 = x3 + h3
        h3 = x3.replace(self.norm3(x3.feats))
        h3 = h3 * (1 + scale_mlp) + shift_mlp
        h3 = self.mlp(h3)
        h3 = h3 * gate_mlp
        x3 = x3 + h3
        
        return x0, x1, x2, x3
    
    
    def _forward_both(self, x0: SparseTensor, x1: SparseTensor, x2: SparseTensor, x3: SparseTensor, x4: SparseTensor, mod: torch.Tensor, cond_cnt: torch.Tensor, cond_txu: torch.Tensor, cond_col: torch.Tensor):
        if self.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(mod).chunk(6, dim=1)
            
        h0 = x0.replace(self.norm1(x0.feats))
        h0 = h0 * (1 + scale_msa) + shift_msa
        
        h1 = x1.replace(self.norm1(x1.feats))
        h1 = h1 * (1 + scale_msa) + shift_msa
        
        h2 = x2.replace(self.norm1(x2.feats))
        h2 = h2 * (1 + scale_msa) + shift_msa
        
        h3 = x3.replace(self.norm1(x3.feats))
        h3 = h3 * (1 + scale_msa) + shift_msa
        
        h4 = x4.replace(self.norm1(x4.feats))
        h4 = h4 * (1 + scale_msa) + shift_msa
        
        h0, h1, h2, h3, h4 = self.self_attn.forward_both(h0, h1, h2, h3, h4)
        
        h0 = h0 * gate_msa
        x0 = x0 + h0
        h0 = x0.replace(self.norm2(x0.feats))
        h0 = self.cross_attn(h0, cond_cnt)
        x0 = x0 + h0
        h0 = x0.replace(self.norm3(x0.feats))
        h0 = h0 * (1 + scale_mlp) + shift_mlp
        h0 = self.mlp(h0)
        h0 = h0 * gate_mlp
        x0 = x0 + h0
        
        h1 = h1 * gate_msa
        x1 = x1 + h1
        h1 = x1.replace(self.norm2(x1.feats))
        h1 = self.cross_attn(h1, cond_cnt)
        x1 = x1 + h1
        h1 = x1.replace(self.norm3(x1.feats))
        h1 = h1 * (1 + scale_mlp) + shift_mlp
        h1 = self.mlp(h1)
        h1 = h1 * gate_mlp
        x1 = x1 + h1
        
        h2 = h2 * gate_msa
        x2 = x2 + h2
        h2 = x2.replace(self.norm2(x2.feats))
        h2 = self.cross_attn(h2, cond_txu)
        x2 = x2 + h2
        h2 = x2.replace(self.norm3(x2.feats))
        h2 = h2 * (1 + scale_mlp) + shift_mlp
        h2 = self.mlp(h2)
        h2 = h2 * gate_mlp
        x2 = x2 + h2
        
        h3 = h3 * gate_msa
        x3 = x3 + h3
        h3 = x3.replace(self.norm2(x3.feats))
        h3 = self.cross_attn(h3, cond_cnt)
        x3 = x3 + h3
        h3 = x3.replace(self.norm3(x3.feats))
        h3 = h3 * (1 + scale_mlp) + shift_mlp
        h3 = self.mlp(h3)
        h3 = h3 * gate_mlp
        x3 = x3 + h3
        
        h4 = h4 * gate_msa
        x4 = x4 + h4
        h4 = x4.replace(self.norm2(x4.feats))
        h4 = self.cross_attn(h4, cond_col)
        x4 = x4 + h4
        h4 = x4.replace(self.norm3(x4.feats))
        h4 = h4 * (1 + scale_mlp) + shift_mlp
        h4 = self.mlp(h4)
        h4 = h4 * gate_mlp
        x4 = x4 + h4
        
        return x0, x1, x2, x3, x4
    
    
    
    def _forward_nocond(self, x0: SparseTensor, x1: SparseTensor, x2: SparseTensor, x3: SparseTensor, x4: SparseTensor, mod: torch.Tensor, cond_cnt: torch.Tensor, cond_txu: torch.Tensor, cond_col: torch.Tensor):
        if self.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(mod).chunk(6, dim=1)
            
        h0 = x0.replace(self.norm1(x0.feats))
        h0 = h0 * (1 + scale_msa) + shift_msa
        
        h1 = x1.replace(self.norm1(x1.feats))
        h1 = h1 * (1 + scale_msa) + shift_msa
        
        h2 = x2.replace(self.norm1(x2.feats))
        h2 = h2 * (1 + scale_msa) + shift_msa
        
        h3 = x3.replace(self.norm1(x3.feats))
        h3 = h3 * (1 + scale_msa) + shift_msa
        
        h4 = x4.replace(self.norm1(x4.feats))
        h4 = h4 * (1 + scale_msa) + shift_msa
        
        h0, h1, h2, h3, h4 = self.self_attn.forward_nocond(h0, h1, h2, h3, h4)
        
        h0 = h0 * gate_msa
        x0 = x0 + h0
        h0 = x0.replace(self.norm2(x0.feats))
        h0 = self.cross_attn(h0, cond_txu)
        x0 = x0 + h0
        h0 = x0.replace(self.norm3(x0.feats))
        h0 = h0 * (1 + scale_mlp) + shift_mlp
        h0 = self.mlp(h0)
        h0 = h0 * gate_mlp
        x0 = x0 + h0
        
        h1 = h1 * gate_msa
        x1 = x1 + h1
        h1 = x1.replace(self.norm2(x1.feats))
        h1 = self.cross_attn(h1, cond_txu)
        x1 = x1 + h1
        h1 = x1.replace(self.norm3(x1.feats))
        h1 = h1 * (1 + scale_mlp) + shift_mlp
        h1 = self.mlp(h1)
        h1 = h1 * gate_mlp
        x1 = x1 + h1
        
        h2 = h2 * gate_msa
        x2 = x2 + h2
        h2 = x2.replace(self.norm2(x2.feats))
        h2 = self.cross_attn(h2, cond_txu)
        x2 = x2 + h2
        h2 = x2.replace(self.norm3(x2.feats))
        h2 = h2 * (1 + scale_mlp) + shift_mlp
        h2 = self.mlp(h2)
        h2 = h2 * gate_mlp
        x2 = x2 + h2
        
        h3 = h3 * gate_msa
        x3 = x3 + h3
        h3 = x3.replace(self.norm2(x3.feats))
        h3 = self.cross_attn(h3, cond_col)
        x3 = x3 + h3
        h3 = x3.replace(self.norm3(x3.feats))
        h3 = h3 * (1 + scale_mlp) + shift_mlp
        h3 = self.mlp(h3)
        h3 = h3 * gate_mlp
        x3 = x3 + h3
        
        h4 = h4 * gate_msa
        x4 = x4 + h4
        h4 = x4.replace(self.norm2(x4.feats))
        h4 = self.cross_attn(h4, cond_col)
        x4 = x4 + h4
        h4 = x4.replace(self.norm3(x4.feats))
        h4 = h4 * (1 + scale_mlp) + shift_mlp
        h4 = self.mlp(h4)
        h4 = h4 * gate_mlp
        x4 = x4 + h4
        
        return x0, x1, x2, x3, x4
    
    
    
    def forward(self, x: SparseTensor, mod: torch.Tensor, context: torch.Tensor) -> SparseTensor:
        if self.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(mod).chunk(6, dim=1)
        h = x.replace(self.norm1(x.feats))
        h = h * (1 + scale_msa) + shift_msa
        h = self.self_attn(h)
        h = h * gate_msa
        x = x + h
        h = x.replace(self.norm2(x.feats))
        h = self.cross_attn(h, context)
        x = x + h
        h = x.replace(self.norm3(x.feats))
        h = h * (1 + scale_mlp) + shift_mlp
        h = self.mlp(h)
        h = h * gate_mlp
        x = x + h
        return x