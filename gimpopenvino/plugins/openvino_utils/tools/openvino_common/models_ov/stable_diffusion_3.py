import os
from pathlib import Path
from PIL import Image
import numpy as np
from openvino.runtime import Core
import torch, math, einops
from torch import nn
import math
from typing import Dict, Optional
import numpy as np
import torch.nn as nn
from einops import rearrange, repeat
import tqdm
from transformers import CLIPTokenizer, T5TokenizerFast

# Note: latest 8B-1024 prefers SHIFT=3.0, but other models prefer SHIFT=1.0
SHIFT = 1.0
# Naturally, adjust to the width/height of the model you have
WIDTH = 512
HEIGHT = 512
# Pick your prompt
# PROMPT = "a photo of a cat"
# PROMPT = "A fairy mushroom house in an enchanted bioluminescent forest."
# PROMPT = "Underwater view of a huge fantastical glowing transparent glass whale swimming in the ocean. Surrounded by glowing jellyfish."
# PROMPT = "cyborg man with a visible detailed brain,muscles cable wires, biopunk, cybernetic, cyberpunk white marble bust, canon m50, 100mm, sharp focus, smooth hyperrealism, highly detailed, intricate details. Below the bust is a plaque which reads 'Terminator'"
PROMPT = "A photo of a cat, wearing a collar with an entire universe inside a charm hanging from the collar."
# Recent models prefer lower CFGs, older models prefer higher
CFG_SCALE = 5
# Different models want different step counts but most will be good at 50, albeit that's slow to run
STEPS = 50
# Random seed
import random 
SEED = random.randint(0,100000)
# Actual model file path
MODEL = "ov_int4_models/sd3_denoiser_alpha.xml"
# Output file path
OUTPUT = "output_" + str(SEED) + ".png"

# clip_g, clip_l, t5, sd3 denoiser, vae ]
DEVICE = ["CPU","CPU","CPU","GPU","GPU"]


## Implements MM-DiT
class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding"""
    def __init__(
            self,
            img_size: Optional[int] = 224,
            patch_size: int = 16,
            in_chans: int = 3,
            embed_dim: int = 768,
            flatten: bool = True,
            bias: bool = True,
            strict_img_size: bool = True,
            dynamic_img_pad: bool = False,
            dtype=None,
            device=None,
    ):
        super().__init__()
        self.patch_size = (patch_size, patch_size)
        if img_size is not None:
            self.img_size = (img_size, img_size)
            self.grid_size = tuple([s // p for s, p in zip(self.img_size, self.patch_size)])
            self.num_patches = self.grid_size[0] * self.grid_size[1]
        else:
            self.img_size = None
            self.grid_size = None
            self.num_patches = None

        # flatten spatial dim and transpose to channels last, kept for bwd compat
        self.flatten = flatten
        self.strict_img_size = strict_img_size
        self.dynamic_img_pad = dynamic_img_pad

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias, dtype=dtype, device=device)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        return x


def modulate(x, shift, scale):
    if shift is None:
        shift = torch.zeros_like(scale)
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0, scaling_factor=None, offset=None):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)
    if scaling_factor is not None:
        grid = grid / scaling_factor
    if offset is not None:
        grid = grid - offset
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)
    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product
    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)
    return np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""

    def __init__(self, hidden_size, frequency_embedding_size=256, dtype=None, device=None):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True, dtype=dtype, device=device),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True, dtype=dtype, device=device),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        if torch.is_floating_point(t):
            embedding = embedding.to(dtype=t.dtype)
        return embedding

    def forward(self, t, dtype, **kwargs):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class VectorEmbedder(nn.Module):
    """Embeds a flat vector of dimension input_dim"""

    def __init__(self, input_dim: int, hidden_size: int, dtype=None, device=None):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_size, bias=True, dtype=dtype, device=device),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True, dtype=dtype, device=device),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


#################################################################################
#                                 Core DiT Model                                #
#################################################################################


def split_qkv(qkv, head_dim):
    qkv = qkv.reshape(qkv.shape[0], qkv.shape[1], 3, -1, head_dim).movedim(2, 0)
    return qkv[0], qkv[1], qkv[2]

def optimized_attention(qkv, num_heads):
    return attention(qkv[0], qkv[1], qkv[2], num_heads)

class SelfAttention(nn.Module):
    ATTENTION_MODES = ("xformers", "torch", "torch-hb", "math", "debug")

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_mode: str = "xformers",
        pre_only: bool = False,
        qk_norm: Optional[str] = None,
        rmsnorm: bool = False,
        dtype=None,
        device=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias, dtype=dtype, device=device)
        if not pre_only:
            self.proj = nn.Linear(dim, dim, dtype=dtype, device=device)
        assert attn_mode in self.ATTENTION_MODES
        self.attn_mode = attn_mode
        self.pre_only = pre_only

        if qk_norm == "rms":
            self.ln_q = RMSNorm(self.head_dim, elementwise_affine=True, eps=1.0e-6, dtype=dtype, device=device)
            self.ln_k = RMSNorm(self.head_dim, elementwise_affine=True, eps=1.0e-6, dtype=dtype, device=device)
        elif qk_norm == "ln":
            self.ln_q = nn.LayerNorm(self.head_dim, elementwise_affine=True, eps=1.0e-6, dtype=dtype, device=device)
            self.ln_k = nn.LayerNorm(self.head_dim, elementwise_affine=True, eps=1.0e-6, dtype=dtype, device=device)
        elif qk_norm is None:
            self.ln_q = nn.Identity()
            self.ln_k = nn.Identity()
        else:
            raise ValueError(qk_norm)

    def pre_attention(self, x: torch.Tensor):
        B, L, C = x.shape
        qkv = self.qkv(x)
        q, k, v = split_qkv(qkv, self.head_dim)
        q = self.ln_q(q).reshape(q.shape[0], q.shape[1], -1)
        k = self.ln_k(k).reshape(q.shape[0], q.shape[1], -1)
        return (q, k, v)

    def post_attention(self, x: torch.Tensor) -> torch.Tensor:
        assert not self.pre_only
        x = self.proj(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        (q, k, v) = self.pre_attention(x)
        x = attention(q, k, v, self.num_heads)
        x = self.post_attention(x)
        return x


class RMSNorm(torch.nn.Module):
    def __init__(
        self, dim: int, elementwise_affine: bool = False, eps: float = 1e-6, device=None, dtype=None
    ):
        """
        Initialize the RMSNorm normalization layer.
        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.
        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.
        """
        super().__init__()
        self.eps = eps
        self.learnable_scale = elementwise_affine
        if self.learnable_scale:
            self.weight = nn.Parameter(torch.empty(dim, device=device, dtype=dtype))
        else:
            self.register_parameter("weight", None)

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.
        Args:
            x (torch.Tensor): The input tensor.
        Returns:
            torch.Tensor: The normalized tensor.
        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.
        Args:
            x (torch.Tensor): The input tensor.
        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.
        """
        x = self._norm(x)
        if self.learnable_scale:
            return x * self.weight.to(device=x.device, dtype=x.dtype)
        else:
            return x


class SwiGLUFeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float] = None,
    ):
        """
        Initialize the FeedForward module.

        Args:
            dim (int): Input dimension.
            hidden_dim (int): Hidden dimension of the feedforward layer.
            multiple_of (int): Value to ensure hidden dimension is a multiple of this value.
            ffn_dim_multiplier (float, optional): Custom multiplier for hidden dimension. Defaults to None.

        Attributes:
            w1 (ColumnParallelLinear): Linear transformation for the first layer.
            w2 (RowParallelLinear): Linear transformation for the second layer.
            w3 (ColumnParallelLinear): Linear transformation for the third layer.

        """
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))


class DismantledBlock(nn.Module):
    """A DiT block with gated adaptive layer norm (adaLN) conditioning."""

    ATTENTION_MODES = ("xformers", "torch", "torch-hb", "math", "debug")

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: str = "xformers",
        qkv_bias: bool = False,
        pre_only: bool = False,
        rmsnorm: bool = False,
        scale_mod_only: bool = False,
        swiglu: bool = False,
        qk_norm: Optional[str] = None,
        dtype=None,
        device=None,
        **block_kwargs,
    ):
        super().__init__()
        assert attn_mode in self.ATTENTION_MODES
        if not rmsnorm:
            self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        else:
            self.norm1 = RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias, attn_mode=attn_mode, pre_only=pre_only, qk_norm=qk_norm, rmsnorm=rmsnorm, dtype=dtype, device=device)
        if not pre_only:
            if not rmsnorm:
                self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
            else:
                self.norm2 = RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        if not pre_only:
            if not swiglu:
                self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=nn.GELU(approximate="tanh"), dtype=dtype, device=device)
            else:
                self.mlp = SwiGLUFeedForward(dim=hidden_size, hidden_dim=mlp_hidden_dim, multiple_of=256)
        self.scale_mod_only = scale_mod_only
        if not scale_mod_only:
            n_mods = 6 if not pre_only else 2
        else:
            n_mods = 4 if not pre_only else 1
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, n_mods * hidden_size, bias=True, dtype=dtype, device=device))
        self.pre_only = pre_only

    def pre_attention(self, x: torch.Tensor, c: torch.Tensor):
        assert x is not None, "pre_attention called with None input"
        if not self.pre_only:
            if not self.scale_mod_only:
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
            else:
                shift_msa = None
                shift_mlp = None
                scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(4, dim=1)
            qkv = self.attn.pre_attention(modulate(self.norm1(x), shift_msa, scale_msa))
            return qkv, (x, gate_msa, shift_mlp, scale_mlp, gate_mlp)
        else:
            if not self.scale_mod_only:
                shift_msa, scale_msa = self.adaLN_modulation(c).chunk(2, dim=1)
            else:
                shift_msa = None
                scale_msa = self.adaLN_modulation(c)
            qkv = self.attn.pre_attention(modulate(self.norm1(x), shift_msa, scale_msa))
            return qkv, None

    def post_attention(self, attn, x, gate_msa, shift_mlp, scale_mlp, gate_mlp):
        assert not self.pre_only
        x = x + gate_msa.unsqueeze(1) * self.attn.post_attention(attn)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        assert not self.pre_only
        (q, k, v), intermediates = self.pre_attention(x, c)
        attn = attention(q, k, v, self.attn.num_heads)
        return self.post_attention(attn, *intermediates)


def block_mixing(context, x, context_block, x_block, c):
    assert context is not None, "block_mixing called with None context"
    context_qkv, context_intermediates = context_block.pre_attention(context, c)

    x_qkv, x_intermediates = x_block.pre_attention(x, c)

    o = []
    for t in range(3):
        o.append(torch.cat((context_qkv[t], x_qkv[t]), dim=1))
    q, k, v = tuple(o)

    attn = attention(q, k, v, x_block.attn.num_heads)
    context_attn, x_attn = (attn[:, : context_qkv[0].shape[1]], attn[:, context_qkv[0].shape[1] :])

    if not context_block.pre_only:
        context = context_block.post_attention(context_attn, *context_intermediates)
    else:
        context = None
    x = x_block.post_attention(x_attn, *x_intermediates)
    return context, x


class JointBlock(nn.Module):
    """just a small wrapper to serve as a fsdp unit"""

    def __init__(self, *args, **kwargs):
        super().__init__()
        pre_only = kwargs.pop("pre_only")
        qk_norm = kwargs.pop("qk_norm", None)
        self.context_block = DismantledBlock(*args, pre_only=pre_only, qk_norm=qk_norm, **kwargs)
        self.x_block = DismantledBlock(*args, pre_only=False, qk_norm=qk_norm, **kwargs)

    def forward(self, *args, **kwargs):
        return block_mixing(*args, context_block=self.context_block, x_block=self.x_block, **kwargs)


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size: int, patch_size: int, out_channels: int, total_out_channels: Optional[int] = None, dtype=None, device=None):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        self.linear = (
            nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True, dtype=dtype, device=device)
            if (total_out_channels is None)
            else nn.Linear(hidden_size, total_out_channels, bias=True, dtype=dtype, device=device)
        )
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True, dtype=dtype, device=device))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class MMDiT(nn.Module):
    """Diffusion model with a Transformer backbone."""

    def __init__(
        self,
        input_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 4,
        depth: int = 28,
        mlp_ratio: float = 4.0,
        learn_sigma: bool = False,
        adm_in_channels: Optional[int] = None,
        context_embedder_config: Optional[Dict] = None,
        register_length: int = 0,
        attn_mode: str = "torch",
        rmsnorm: bool = False,
        scale_mod_only: bool = False,
        swiglu: bool = False,
        out_channels: Optional[int] = None,
        pos_embed_scaling_factor: Optional[float] = None,
        pos_embed_offset: Optional[float] = None,
        pos_embed_max_size: Optional[int] = None,
        num_patches = None,
        qk_norm: Optional[str] = None,
        qkv_bias: bool = True,
        dtype = None,
        device = None,
    ):
        super().__init__()
        print(f"mmdit initializing with: {input_size=}, {patch_size=}, {in_channels=}, {depth=}, {mlp_ratio=}, {learn_sigma=}, {adm_in_channels=}, {context_embedder_config=}, {register_length=}, {attn_mode=}, {rmsnorm=}, {scale_mod_only=}, {swiglu=}, {out_channels=}, {pos_embed_scaling_factor=}, {pos_embed_offset=}, {pos_embed_max_size=}, {num_patches=}, {qk_norm=}, {qkv_bias=}, {dtype=}, {device=}")
        self.dtype = dtype
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        default_out_channels = in_channels * 2 if learn_sigma else in_channels
        self.out_channels = out_channels if out_channels is not None else default_out_channels
        self.patch_size = patch_size
        self.pos_embed_scaling_factor = pos_embed_scaling_factor
        self.pos_embed_offset = pos_embed_offset
        self.pos_embed_max_size = pos_embed_max_size

        # apply magic --> this defines a head_size of 64
        hidden_size = 64 * depth
        num_heads = depth

        self.num_heads = num_heads

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True, strict_img_size=self.pos_embed_max_size is None, dtype=dtype, device=device)
        self.t_embedder = TimestepEmbedder(hidden_size, dtype=dtype, device=device)

        if adm_in_channels is not None:
            assert isinstance(adm_in_channels, int)
            self.y_embedder = VectorEmbedder(adm_in_channels, hidden_size, dtype=dtype, device=device)

        self.context_embedder = nn.Identity()
        if context_embedder_config is not None:
            if context_embedder_config["target"] == "torch.nn.Linear":
                self.context_embedder = nn.Linear(**context_embedder_config["params"], dtype=dtype, device=device)

        self.register_length = register_length
        if self.register_length > 0:
            self.register = nn.Parameter(torch.randn(1, register_length, hidden_size, dtype=dtype, device=device))

        # num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        # just use a buffer already
        if num_patches is not None:
            self.register_buffer(
                "pos_embed",
                torch.zeros(1, num_patches, hidden_size, dtype=dtype, device=device),
            )
        else:
            self.pos_embed = None

        self.joint_blocks = nn.ModuleList(
            [
                JointBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, attn_mode=attn_mode, pre_only=i == depth - 1, rmsnorm=rmsnorm, scale_mod_only=scale_mod_only, swiglu=swiglu, qk_norm=qk_norm, dtype=dtype, device=device)
                for i in range(depth)
            ]
        )

        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels, dtype=dtype, device=device)

    def cropped_pos_embed(self, hw):
        assert self.pos_embed_max_size is not None
        p = self.x_embedder.patch_size[0]
        h, w = hw
        # patched size
        h = h // p
        w = w // p
        assert h <= self.pos_embed_max_size, (h, self.pos_embed_max_size)
        assert w <= self.pos_embed_max_size, (w, self.pos_embed_max_size)
        top = (self.pos_embed_max_size - h) // 2
        left = (self.pos_embed_max_size - w) // 2
        spatial_pos_embed = rearrange(
            self.pos_embed,
            "1 (h w) c -> 1 h w c",
            h=self.pos_embed_max_size,
            w=self.pos_embed_max_size,
        )
        spatial_pos_embed = spatial_pos_embed[:, top : top + h, left : left + w, :]
        spatial_pos_embed = rearrange(spatial_pos_embed, "1 h w c -> 1 (h w) c")
        return spatial_pos_embed

    def unpatchify(self, x, hw=None):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        if hw is None:
            h = w = int(x.shape[1] ** 0.5)
        else:
            h, w = hw
            h = h // p
            w = w // p
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def forward_core_with_concat(self, x: torch.Tensor, c_mod: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.register_length > 0:
            context = torch.cat((repeat(self.register, "1 ... -> b ...", b=x.shape[0]), context if context is not None else torch.Tensor([]).type_as(x)), 1)

        # context is B, L', D
        # x is B, L, D
        for block in self.joint_blocks:
            context, x = block(context, x, c=c_mod)

        x = self.final_layer(x, c_mod)  # (N, T, patch_size ** 2 * out_channels)
        return x

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: Optional[torch.Tensor] = None, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        hw = x.shape[-2:]
        x = self.x_embedder(x) + self.cropped_pos_embed(hw)
        c = self.t_embedder(t, dtype=x.dtype)  # (N, D)
        if y is not None:
            y = self.y_embedder(y)  # (N, D)
            c = c + y  # (N, D)

        context = self.context_embedder(context)

        x = self.forward_core_with_concat(x, c, context)

        x = self.unpatchify(x, hw=hw)  # (N, out_channels, H, W)
        return x


class CoreMMDITWrapper(MMDiT):
    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, context: Optional[torch.Tensor] = None, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        return super().forward(x, timesteps, context=context, y=y)




#################################################################################################
### Core/Utility
#################################################################################################


def attention(q, k, v, heads, mask=None):
    """Convenience wrapper around a basic attention operation"""
    b, _, dim_head = q.shape
    dim_head //= heads
    q, k, v = map(lambda t: t.view(b, -1, heads, dim_head).transpose(1, 2), (q, k, v))
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)
    return out.transpose(1, 2).reshape(b, -1, heads * dim_head)


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks"""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, dtype=None, device=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias, dtype=dtype, device=device)
        self.act = act_layer
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias, dtype=dtype, device=device)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


#################################################################################################
### CLIP
#################################################################################################
class CLIPAttention(torch.nn.Module):
    def __init__(self, embed_dim, heads, dtype, device):
        super().__init__()
        self.heads = heads
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True, dtype=dtype, device=device)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True, dtype=dtype, device=device)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True, dtype=dtype, device=device)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True, dtype=dtype, device=device)

    def forward(self, x, mask=None):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        out = attention(q, k, v, self.heads, mask)
        return self.out_proj(out)


ACTIVATIONS = {
    "quick_gelu": lambda a: a * torch.sigmoid(1.702 * a),
    "gelu": torch.nn.functional.gelu,
}

class CLIPLayer(torch.nn.Module):
    def __init__(self, embed_dim, heads, intermediate_size, intermediate_activation, dtype, device):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(embed_dim, dtype=dtype, device=device)
        self.self_attn = CLIPAttention(embed_dim, heads, dtype, device)
        self.layer_norm2 = nn.LayerNorm(embed_dim, dtype=dtype, device=device)
        #self.mlp = CLIPMLP(embed_dim, intermediate_size, intermediate_activation, dtype, device)
        self.mlp = Mlp(embed_dim, intermediate_size, embed_dim, act_layer=ACTIVATIONS[intermediate_activation], dtype=dtype, device=device)

    def forward(self, x, mask=None):
        x += self.self_attn(self.layer_norm1(x), mask)
        x += self.mlp(self.layer_norm2(x))
        return x


class CLIPEncoder(torch.nn.Module):
    def __init__(self, num_layers, embed_dim, heads, intermediate_size, intermediate_activation, dtype, device):
        super().__init__()
        self.layers = torch.nn.ModuleList([CLIPLayer(embed_dim, heads, intermediate_size, intermediate_activation, dtype, device) for i in range(num_layers)])

    def forward(self, x, mask=None, intermediate_output=None):
        if intermediate_output is not None:
            if intermediate_output < 0:
                intermediate_output = len(self.layers) + intermediate_output
        intermediate = None
        for i, l in enumerate(self.layers):
            x = l(x, mask)
            if i == intermediate_output:
                intermediate = x.clone()
        return x, intermediate


class CLIPEmbeddings(torch.nn.Module):
    def __init__(self, embed_dim, vocab_size=49408, num_positions=77, dtype=None, device=None):
        super().__init__()
        self.token_embedding = torch.nn.Embedding(vocab_size, embed_dim, dtype=dtype, device=device)
        self.position_embedding = torch.nn.Embedding(num_positions, embed_dim, dtype=dtype, device=device)

    def forward(self, input_tokens):
        return self.token_embedding(input_tokens) + self.position_embedding.weight


class CLIPTextModel_(torch.nn.Module):
    def __init__(self, config_dict, dtype, device):
        num_layers = config_dict["num_hidden_layers"]
        embed_dim = config_dict["hidden_size"]
        heads = config_dict["num_attention_heads"]
        intermediate_size = config_dict["intermediate_size"]
        intermediate_activation = config_dict["hidden_act"]
        super().__init__()
        self.embeddings = CLIPEmbeddings(embed_dim, dtype=torch.float32, device=device)
        self.encoder = CLIPEncoder(num_layers, embed_dim, heads, intermediate_size, intermediate_activation, dtype, device)
        self.final_layer_norm = nn.LayerNorm(embed_dim, dtype=dtype, device=device)

    def forward(self, input_tokens, intermediate_output=None, final_layer_norm_intermediate=True):
        x = self.embeddings(input_tokens)
        causal_mask = torch.empty(x.shape[1], x.shape[1], dtype=x.dtype, device=x.device).fill_(float("-inf")).triu_(1)
        x, i = self.encoder(x, mask=causal_mask, intermediate_output=intermediate_output)
        x = self.final_layer_norm(x)
        if i is not None and final_layer_norm_intermediate:
            i = self.final_layer_norm(i)
        pooled_output = x[torch.arange(x.shape[0], device=x.device), input_tokens.to(dtype=torch.int, device=x.device).argmax(dim=-1),]
        return x, i, pooled_output


class CLIPTextModel(torch.nn.Module):
    def __init__(self, config_dict, dtype, device):
        super().__init__()
        self.num_layers = config_dict["num_hidden_layers"]
        self.text_model = CLIPTextModel_(config_dict, dtype, device)
        embed_dim = config_dict["hidden_size"]
        self.text_projection = nn.Linear(embed_dim, embed_dim, bias=False, dtype=dtype, device=device)
        self.text_projection.weight.copy_(torch.eye(embed_dim))
        self.dtype = dtype

    def get_input_embeddings(self):
        return self.text_model.embeddings.token_embedding

    def set_input_embeddings(self, embeddings):
        self.text_model.embeddings.token_embedding = embeddings

    def forward(self, *args, **kwargs):
        x = self.text_model(*args, **kwargs)
        out = self.text_projection(x[2])
        return (x[0], x[1], out, x[2])


class SDTokenizer:
    def __init__(self, max_length=77, pad_with_end=True, tokenizer=None, has_start_token=True, pad_to_max_length=True, min_length=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.min_length = min_length
        empty = self.tokenizer('')["input_ids"]
        if has_start_token:
            self.tokens_start = 1
            self.start_token = empty[0]
            self.end_token = empty[1]
        else:
            self.tokens_start = 0
            self.start_token = None
            self.end_token = empty[0]
        self.pad_with_end = pad_with_end
        self.pad_to_max_length = pad_to_max_length
        vocab = self.tokenizer.get_vocab()
        self.inv_vocab = {v: k for k, v in vocab.items()}
        self.max_word_length = 8


    def tokenize_with_weights(self, text:str):
        """Tokenize the text, with weight values - presume 1.0 for all and ignore other features here. The details aren't relevant for a reference impl, and weights themselves has weak effect on SD3."""
        if self.pad_with_end:
            pad_token = self.end_token
        else:
            pad_token = 0
        batch = []
        if self.start_token is not None:
            batch.append((self.start_token, 1.0))
        to_tokenize = text.replace("\n", " ").split(' ')
        to_tokenize = [x for x in to_tokenize if x != ""]
        for word in to_tokenize:
            batch.extend([(t, 1) for t in self.tokenizer(word)["input_ids"][self.tokens_start:-1]])
        batch.append((self.end_token, 1.0))
        if self.pad_to_max_length:
            batch.extend([(pad_token, 1.0)] * (self.max_length - len(batch)))
        if self.min_length is not None and len(batch) < self.min_length:
            batch.extend([(pad_token, 1.0)] * (self.min_length - len(batch)))
        return [batch]


class SDXLClipGTokenizer(SDTokenizer):
    def __init__(self, tokenizer):
        super().__init__(pad_with_end=False, tokenizer=tokenizer)

class SD3Tokenizer:
    def __init__(self):
        clip_tokenizer = CLIPTokenizer.from_pretrained(Path.home() / "openvino-ai-plugins-gimp" / "weights" / "stable-diffusion-ov" / "stable-diffusion-3.0" / "clip-vit-large-patch14")
        self.clip_l = SDTokenizer(tokenizer=clip_tokenizer)
        self.clip_g = SDXLClipGTokenizer(clip_tokenizer)
        self.t5xxl = T5XXLTokenizer()

    def tokenize_with_weights(self, text:str):
        out = {}
        out["g"] = self.clip_g.tokenize_with_weights(text)
        out["l"] = self.clip_l.tokenize_with_weights(text)
        out["t5xxl"] = self.t5xxl.tokenize_with_weights(text)
        return out


class ClipTokenWeightEncoder:
    def encode_token_weights(self, token_weight_pairs):
        tokens = list(map(lambda a: a[0], token_weight_pairs[0]))
        out, pooled = self([tokens])
        if pooled is not None:
            first_pooled = pooled[0:1].cpu()
        else:
            first_pooled = pooled
        output = [out[0:1]]
        return torch.cat(output, dim=-2).cpu(), first_pooled


class SDClipModel(torch.nn.Module, ClipTokenWeightEncoder):
    """Uses the CLIP transformer encoder for text (from huggingface)"""
    LAYERS = ["last", "pooled", "hidden"]
    def __init__(self, device="cpu", max_length=77, layer="last", layer_idx=None, textmodel_json_config=None, dtype=None, model_class=CLIPTextModel,
                 special_tokens={"start": 49406, "end": 49407, "pad": 49407}, layer_norm_hidden_state=True, return_projected_pooled=True):
        super().__init__()
        assert layer in self.LAYERS
        self.transformer = model_class(textmodel_json_config, dtype, device)
        self.num_layers = self.transformer.num_layers
        self.max_length = max_length
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False
        self.layer = layer
        self.layer_idx = None
        self.special_tokens = special_tokens
        self.logit_scale = torch.nn.Parameter(torch.tensor(4.6055))
        self.layer_norm_hidden_state = layer_norm_hidden_state
        self.return_projected_pooled = return_projected_pooled
        if layer == "hidden":
            assert layer_idx is not None
            assert abs(layer_idx) < self.num_layers
            self.set_clip_options({"layer": layer_idx})
        self.options_default = (self.layer, self.layer_idx, self.return_projected_pooled)

    def set_clip_options(self, options):
        layer_idx = options.get("layer", self.layer_idx)
        self.return_projected_pooled = options.get("projected_pooled", self.return_projected_pooled)
        if layer_idx is None or abs(layer_idx) > self.num_layers:
            self.layer = "last"
        else:
            self.layer = "hidden"
            self.layer_idx = layer_idx

    def forward(self, tokens):
        backup_embeds = self.transformer.get_input_embeddings()
        device = backup_embeds.weight.device
        tokens = torch.LongTensor(tokens).to(device)
        outputs = self.transformer(tokens, intermediate_output=self.layer_idx, final_layer_norm_intermediate=self.layer_norm_hidden_state)
        self.transformer.set_input_embeddings(backup_embeds)
        if self.layer == "last":
            z = outputs[0]
        else:
            z = outputs[1]
        pooled_output = None
        if len(outputs) >= 3:
            if not self.return_projected_pooled and len(outputs) >= 4 and outputs[3] is not None:
                pooled_output = outputs[3].float()
            elif outputs[2] is not None:
                pooled_output = outputs[2].float()
        return z.float(), pooled_output


class SDXLClipG(SDClipModel):
    """Wraps the CLIP-G model into the SD-CLIP-Model interface"""
    def __init__(self, config, device="cpu", layer="penultimate", layer_idx=None, dtype=None):
        if layer == "penultimate":
            layer="hidden"
            layer_idx=-2
        super().__init__(device=device, layer=layer, layer_idx=layer_idx, textmodel_json_config=config, dtype=dtype, special_tokens={"start": 49406, "end": 49407, "pad": 0}, layer_norm_hidden_state=False)


class T5XXLModel(SDClipModel):
    """Wraps the T5-XXL model into the SD-CLIP-Model interface for convenience"""
    def __init__(self, config, device="cpu", layer="last", layer_idx=None, dtype=None):
        super().__init__(device=device, layer=layer, layer_idx=layer_idx, textmodel_json_config=config, dtype=dtype, special_tokens={"end": 1, "pad": 0}, model_class=T5)


#################################################################################################
### T5 implementation, for the T5-XXL text encoder portion, largely pulled from upstream impl
#################################################################################################
class T5XXLTokenizer(SDTokenizer):
    """Wraps the T5 Tokenizer from HF into the SDTokenizer interface"""
    def __init__(self):
        super().__init__(pad_with_end=False, tokenizer=T5TokenizerFast.from_pretrained(Path.home() / "openvino-ai-plugins-gimp" / "weights" / "stable-diffusion-ov" / "stable-diffusion-3.0" / "t5-v1_1-xxl"), has_start_token=False, pad_to_max_length=False, max_length=99999999, min_length=77)


class T5LayerNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-6, dtype=None, device=None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size, dtype=dtype, device=device))
        self.variance_epsilon = eps

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight.to(device=x.device, dtype=x.dtype) * x


class T5DenseGatedActDense(torch.nn.Module):
    def __init__(self, model_dim, ff_dim, dtype, device):
        super().__init__()
        self.wi_0 = nn.Linear(model_dim, ff_dim, bias=False, dtype=dtype, device=device)
        self.wi_1 = nn.Linear(model_dim, ff_dim, bias=False, dtype=dtype, device=device)
        self.wo = nn.Linear(ff_dim, model_dim, bias=False, dtype=dtype, device=device)

    def forward(self, x):
        hidden_gelu = torch.nn.functional.gelu(self.wi_0(x), approximate="tanh")
        hidden_linear = self.wi_1(x)
        x = hidden_gelu * hidden_linear
        x = self.wo(x)
        return x


class T5LayerFF(torch.nn.Module):
    def __init__(self, model_dim, ff_dim, dtype, device):
        super().__init__()
        self.DenseReluDense = T5DenseGatedActDense(model_dim, ff_dim, dtype, device)
        self.layer_norm = T5LayerNorm(model_dim, dtype=dtype, device=device)

    def forward(self, x):
        forwarded_states = self.layer_norm(x)
        forwarded_states = self.DenseReluDense(forwarded_states)
        x += forwarded_states
        return x


class T5Attention(torch.nn.Module):
    def __init__(self, model_dim, inner_dim, num_heads, relative_attention_bias, dtype, device):
        super().__init__()
        # Mesh TensorFlow initialization to avoid scaling before softmax
        self.q = nn.Linear(model_dim, inner_dim, bias=False, dtype=dtype, device=device)
        self.k = nn.Linear(model_dim, inner_dim, bias=False, dtype=dtype, device=device)
        self.v = nn.Linear(model_dim, inner_dim, bias=False, dtype=dtype, device=device)
        self.o = nn.Linear(inner_dim, model_dim, bias=False, dtype=dtype, device=device)
        self.num_heads = num_heads
        self.relative_attention_bias = None
        if relative_attention_bias:
            self.relative_attention_num_buckets = 32
            self.relative_attention_max_distance = 128
            self.relative_attention_bias = torch.nn.Embedding(self.relative_attention_num_buckets, self.num_heads, device=device)

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on

        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        # now relative_position is in the range [0, inf)
        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1))
        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets

    def compute_bias(self, query_length, key_length, device):
        """Compute binned relative position bias"""
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=True,
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        return values

    def forward(self, x, past_bias=None):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        if self.relative_attention_bias is not None:
            past_bias = self.compute_bias(x.shape[1], x.shape[1], x.device)
        if past_bias is not None:
            mask = past_bias
        out = attention(q, k * ((k.shape[-1] / self.num_heads) ** 0.5), v, self.num_heads, mask)
        return self.o(out), past_bias


class T5LayerSelfAttention(torch.nn.Module):
    def __init__(self, model_dim, inner_dim, ff_dim, num_heads, relative_attention_bias, dtype, device):
        super().__init__()
        self.SelfAttention = T5Attention(model_dim, inner_dim, num_heads, relative_attention_bias, dtype, device)
        self.layer_norm = T5LayerNorm(model_dim, dtype=dtype, device=device)

    def forward(self, x, past_bias=None):
        output, past_bias = self.SelfAttention(self.layer_norm(x), past_bias=past_bias)
        x += output
        return x, past_bias


class T5Block(torch.nn.Module):
    def __init__(self, model_dim, inner_dim, ff_dim, num_heads, relative_attention_bias, dtype, device):
        super().__init__()
        self.layer = torch.nn.ModuleList()
        self.layer.append(T5LayerSelfAttention(model_dim, inner_dim, ff_dim, num_heads, relative_attention_bias, dtype, device))
        self.layer.append(T5LayerFF(model_dim, ff_dim, dtype, device))

    def forward(self, x, past_bias=None):
        x, past_bias = self.layer[0](x, past_bias)
        x = self.layer[-1](x)
        return x, past_bias


class T5Stack(torch.nn.Module):
    def __init__(self, num_layers, model_dim, inner_dim, ff_dim, num_heads, vocab_size, dtype, device):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(vocab_size, model_dim, device=device)
        self.block = torch.nn.ModuleList([T5Block(model_dim, inner_dim, ff_dim, num_heads, relative_attention_bias=(i == 0), dtype=dtype, device=device) for i in range(num_layers)])
        self.final_layer_norm = T5LayerNorm(model_dim, dtype=dtype, device=device)

    def forward(self, input_ids, intermediate_output=None, final_layer_norm_intermediate=True):
        intermediate = None
        x = self.embed_tokens(input_ids)
        past_bias = None
        for i, l in enumerate(self.block):
            x, past_bias = l(x, past_bias)
            if i == intermediate_output:
                intermediate = x.clone()
        x = self.final_layer_norm(x)
        if intermediate is not None and final_layer_norm_intermediate:
            intermediate = self.final_layer_norm(intermediate)
        return x, intermediate


class T5(torch.nn.Module):
    def __init__(self, config_dict, dtype, device):
        super().__init__()
        self.num_layers = config_dict["num_layers"]
        self.encoder = T5Stack(self.num_layers, config_dict["d_model"], config_dict["d_model"], config_dict["d_ff"], config_dict["num_heads"], config_dict["vocab_size"], dtype, device)
        self.dtype = dtype

    def get_input_embeddings(self):
        return self.encoder.embed_tokens

    def set_input_embeddings(self, embeddings):
        self.encoder.embed_tokens = embeddings

    def forward(self, *args, **kwargs):
        return self.encoder(*args, **kwargs)

#################################################################################################
### MMDiT Model Wrapping
#################################################################################################


class ModelSamplingDiscreteFlow(torch.nn.Module):
    """Helper for sampler scheduling (ie timestep/sigma calculations) for Discrete Flow models"""
    def __init__(self, shift=1.0):
        super().__init__()
        self.shift = shift
        timesteps = 1000
        ts = self.sigma(torch.arange(1, timesteps + 1, 1))
        self.register_buffer('sigmas', ts)

    @property
    def sigma_min(self):
        return self.sigmas[0]

    @property
    def sigma_max(self):
        return self.sigmas[-1]

    def timestep(self, sigma):
        return sigma * 1000

    def sigma(self, timestep: torch.Tensor):
        timestep = timestep / 1000.0
        if self.shift == 1.0:
            return timestep
        return self.shift * timestep / (1 + (self.shift - 1) * timestep)

    def calculate_denoised(self, sigma, model_output, model_input):
        sigma = sigma.view(sigma.shape[:1] + (1,) * (model_output.ndim - 1))
        return model_input - model_output * sigma

    def noise_scaling(self, sigma, noise, latent_image, max_denoise=False):
        return sigma * noise + (1.0 - sigma) * latent_image


class BaseModel(torch.nn.Module):
    """Wrapper around the core MM-DiT model"""
    def __init__(self, shift=1.0, device=None, dtype=torch.float32, file=None, prefix=""):
        super().__init__()
        # Important configuration values can be quickly determined by checking shapes in the source file
        # Some of these will vary between models (eg 2B vs 8B primarily differ in their depth, but also other details change)
        patch_size = file.get_tensor(f"{prefix}x_embedder.proj.weight").shape[2]
        depth = file.get_tensor(f"{prefix}x_embedder.proj.weight").shape[0] // 64
        num_patches = file.get_tensor(f"{prefix}pos_embed").shape[1]
        pos_embed_max_size = round(math.sqrt(num_patches))
        adm_in_channels = file.get_tensor(f"{prefix}y_embedder.mlp.0.weight").shape[1]
        context_shape = file.get_tensor(f"{prefix}context_embedder.weight").shape
        context_embedder_config = {
            "target": "torch.nn.Linear",
            "params": {
                "in_features": context_shape[1],
                "out_features": context_shape[0]
            }
        }
        self.diffusion_model = MMDiT(input_size=None, pos_embed_scaling_factor=None, pos_embed_offset=None, pos_embed_max_size=pos_embed_max_size, patch_size=patch_size, in_channels=16, depth=depth, num_patches=num_patches, adm_in_channels=adm_in_channels, context_embedder_config=context_embedder_config, device=device, dtype=dtype)
        self.model_sampling = ModelSamplingDiscreteFlow(shift=shift)

    def apply_model(self, x, sigma, c_crossattn=None, y=None):
        dtype = self.get_dtype()
        timestep = self.model_sampling.timestep(sigma).float()
        model_output = self.diffusion_model(x.to(dtype), timestep, context=c_crossattn.to(dtype), y=y.to(dtype)).float()
        return self.model_sampling.calculate_denoised(sigma, model_output, x)

    def forward(self, *args, **kwargs):
        return self.apply_model(*args, **kwargs)

    def get_dtype(self):
        return self.diffusion_model.dtype


class CFGDenoiser(torch.nn.Module):
    """Helper for applying CFG Scaling to diffusion outputs"""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, timestep, cond, uncond, cond_scale):
        # Run cond and uncond in a batch together
        batched = self.model.apply_model(torch.cat([x, x]), torch.cat([timestep, timestep]), c_crossattn=torch.cat([cond["c_crossattn"], uncond["c_crossattn"]]), y=torch.cat([cond["y"], uncond["y"]]))
        # Then split and apply CFG Scaling
        pos_out, neg_out = batched.chunk(2)
        scaled = neg_out + (pos_out - neg_out) * cond_scale
        return scaled


class SD3LatentFormat:
    """Latents are slightly shifted from center - this class must be called after VAE Decode to correct for the shift"""
    def __init__(self):
        self.scale_factor = 1.5305
        self.shift_factor = 0.0609

    def process_in(self, latent):
        return (latent - self.shift_factor) * self.scale_factor

    def process_out(self, latent):
        return (latent / self.scale_factor) + self.shift_factor


#################################################################################################
### K-Diffusion Sampling
#################################################################################################


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    return x[(...,) + (None,) * dims_to_append]


def to_d(x, sigma, denoised):
    """Converts a denoiser output to a Karras ODE derivative."""
    return (x - denoised) / append_dims(sigma, x.ndim)


@torch.no_grad()
@torch.autocast("cuda", dtype=torch.float16)
def sample_euler(model, x, sigmas, callback, callback_userdata, extra_args=None):
    """Implements Algorithm 2 (Euler steps) from Karras et al. (2022)."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in range(len(sigmas) - 1):
        sigma_hat = sigmas[i]
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)
        dt = sigmas[i + 1] - sigma_hat
        # Euler method
        x = x + d * dt
    return x


#################################################################################################
### VAE
#################################################################################################


def Normalize(in_channels, num_groups=32, dtype=torch.float32, device=None):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True, dtype=dtype, device=device)


class ResnetBlock(torch.nn.Module):
    def __init__(self, *, in_channels, out_channels=None, dtype=torch.float32, device=None):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = Normalize(in_channels, dtype=dtype, device=device)
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dtype=dtype, device=device)
        self.norm2 = Normalize(out_channels, dtype=dtype, device=device)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, dtype=dtype, device=device)
        if self.in_channels != self.out_channels:
            self.nin_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dtype=dtype, device=device)
        else:
            self.nin_shortcut = None
        self.swish = torch.nn.SiLU(inplace=True)

    def forward(self, x):
        hidden = x
        hidden = self.norm1(hidden)
        hidden = self.swish(hidden)
        hidden = self.conv1(hidden)
        hidden = self.norm2(hidden)
        hidden = self.swish(hidden)
        hidden = self.conv2(hidden)
        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        return x + hidden


class AttnBlock(torch.nn.Module):
    def __init__(self, in_channels, dtype=torch.float32, device=None):
        super().__init__()
        self.norm = Normalize(in_channels, dtype=dtype, device=device)
        self.q = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, dtype=dtype, device=device)
        self.k = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, dtype=dtype, device=device)
        self.v = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, dtype=dtype, device=device)
        self.proj_out = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, dtype=dtype, device=device)

    def forward(self, x):
        hidden = self.norm(x)
        q = self.q(hidden)
        k = self.k(hidden)
        v = self.v(hidden)
        b, c, h, w = q.shape
        q, k, v = map(lambda x: einops.rearrange(x, "b c h w -> b 1 (h w) c").contiguous(), (q, k, v))
        hidden = torch.nn.functional.scaled_dot_product_attention(q, k, v)  # scale is dim ** -0.5 per default
        hidden = einops.rearrange(hidden, "b 1 (h w) c -> b c h w", h=h, w=w, c=c, b=b)
        hidden = self.proj_out(hidden)
        return x + hidden


class Upsample(torch.nn.Module):
    def __init__(self, in_channels, dtype=torch.float32, device=None):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, dtype=dtype, device=device)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)
        return x


class VAEDecoder(torch.nn.Module):
    def __init__(self, ch=128, out_ch=3, ch_mult=(1, 2, 4, 4), num_res_blocks=2, resolution=256, z_channels=16, dtype=torch.float32, device=None):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1, dtype=dtype, device=device)
        # middle
        self.mid = torch.nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, dtype=dtype, device=device)
        self.mid.attn_1 = AttnBlock(block_in, dtype=dtype, device=device)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, dtype=dtype, device=device)
        # upsampling
        self.up = torch.nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = torch.nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, dtype=dtype, device=device))
                block_in = block_out
            up = torch.nn.Module()
            up.block = block
            if i_level != 0:
                up.upsample = Upsample(block_in, dtype=dtype, device=device)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order
        # end
        self.norm_out = Normalize(block_in, dtype=dtype, device=device)
        self.conv_out = torch.nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1, dtype=dtype, device=device)
        self.swish = torch.nn.SiLU(inplace=True)

    def forward(self, z):
        # z to block_in
        hidden = self.conv_in(z)
        # middle
        hidden = self.mid.block_1(hidden)
        hidden = self.mid.attn_1(hidden)
        hidden = self.mid.block_2(hidden)
        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                hidden = self.up[i_level].block[i_block](hidden)
            if i_level != 0:
                hidden = self.up[i_level].upsample(hidden)
        # end
        hidden = self.norm_out(hidden)
        hidden = self.swish(hidden)
        hidden = self.conv_out(hidden)
        return hidden


class SDVAE(torch.nn.Module):
    """Note that the VAE Encoder is not included in our current reference SD3 models. Might be added on release. Not needed for most gens anyway, only for img2img (Init Image), so for this codebase we'll just ignore it, and implement only the decoder."""
    def __init__(self, dtype=torch.float32, device=None):
        super().__init__()
        self.decoder = VAEDecoder(dtype=dtype, device=device)

    @torch.autocast("cuda", dtype=torch.float16)
    def decode(self, latent):
        return self.decoder(latent)



class StableDiffusionThreeEngine:
    def __init__(
            self,
            shift=1.0,
            model=None,
            device=["CPU","CPU","CPU","GPU","GPU"]
    ):
        
        self.core = Core()
        self.core.set_property({'CACHE_DIR': os.path.join(model, 'cache')}) #adding caching to reduce init time
        
        print("Loading tokenizers...")
        # NOTE: if you need a reference impl for a high performance CLIP tokenizer instead of just using the HF transformers one,
        # check https://github.com/Stability-AI/StableSwarmUI/blob/master/src/Utils/CliplikeTokenizer.cs
        # (T5 tokenizer is different though)
        self.tokenizer = SD3Tokenizer()
        
        print("Loading OpenCLIP bigG...")
        self.clip_g = self.core.compile_model(os.path.join(model, "clip_g.xml"), device[0], {"INFERENCE_PRECISION_HINT": "f32"})
        print("Loading OpenAI CLIP L...")
        self.clip_l = self.core.compile_model(os.path.join(model, "clip_l.xml"), device[1], {"INFERENCE_PRECISION_HINT": "f32"})
        print("Loading Google T5-v1-XXL...")
        self.t5xxl = self.core.compile_model(os.path.join(model, "t5xxl.xml"), device[2],  {"INFERENCE_PRECISION_HINT": "f32"})
        print("Loading SD3 model...")
        self.sd3 = self.core.compile_model(os.path.join(model, "sd3_denoiser_alpha.xml"), device[3] )
        print("Loading VAE model...")
        self.vae = self.core.compile_model(os.path.join(model, "vae.xml"), device[4])
        print("Models loaded.")
        self.model_sampling = ModelSamplingDiscreteFlow(shift=shift)

    def get_empty_latent(self, width, height):
        print("Prep an empty latent...")
        return torch.ones(1, 16, height // 8, width // 8, device="cpu") * 0.0609

    def get_sigmas(self, sampling, steps):
        start = sampling.timestep(sampling.sigma_max)
        end = sampling.timestep(sampling.sigma_min)
        timesteps = torch.linspace(start, end, steps)
        sigs = []
        for x in range(len(timesteps)):
            ts = timesteps[x]
            sigs.append(sampling.sigma(ts))
        sigs += [0.0]
        return torch.FloatTensor(sigs)

    def get_noise(self, seed, latent):
        generator = torch.manual_seed(seed)
        return torch.randn(latent.size(), dtype=torch.float32, layout=latent.layout, generator=generator, device="cpu").to(latent.dtype)

    def get_cond(self, prompt):
        print("Encode prompt...")
        tokens = self.tokenizer.tokenize_with_weights(prompt)
        l_out, l_pooled, g_out, g_pooled, t5_out = self.encode_token_weights(tokens)
        lg_out = torch.cat([l_out, g_out], dim=-1)
        lg_out = torch.nn.functional.pad(lg_out, (0, 4096 - lg_out.shape[-1]))
        return torch.cat([lg_out, t5_out], dim=-2), torch.cat((l_pooled, g_pooled), dim=-1)

    def max_denoise(self, sigmas):
        max_sigma = float(self.model_sampling.sigma_max)
        sigma = float(sigmas[0])
        return math.isclose(max_sigma, sigma, rel_tol=1e-05) or sigma > max_sigma

    def fix_cond(self, cond):
        cond, pooled = (cond[0], cond[1])
        return { "c_crossattn": cond, "y": pooled }

    def do_sampling(self, latent, seed, conditioning, neg_cond, steps, cfg_scale, callback, callback_userdata) -> torch.Tensor:
        print("Sampling...")
        init_latent = latent
        noise = self.get_noise(seed, init_latent)
        sigmas = self.get_sigmas(self.model_sampling, steps)
        conditioning = self.fix_cond(conditioning)
        neg_cond = self.fix_cond(neg_cond)
        extra_args = { "cond": conditioning, "uncond": neg_cond, "cond_scale": cfg_scale }
        noise_scaled = self.model_sampling.noise_scaling(sigmas[0], noise, latent, self.max_denoise(sigmas))
        latent = self.sample_euler(noise_scaled, sigmas, callback, callback_userdata, extra_args=extra_args)
        latent = SD3LatentFormat().process_out(latent)
        print("Sampling done")
        return latent

    def vae_decode(self, latent) -> Image.Image:
        print("Decoding latent to image...")
        image = self.vae(latent)[0]
        image = np.clip((image + 1.0) / 2.0, a_min=0.0, a_max=1.0)[0]
        decoded_np = 255.0 * np.moveaxis(image, 0, 2)
        decoded_np = decoded_np.astype(np.uint8)
        out_image = Image.fromarray(decoded_np)
        print("Decoded")
        return out_image

    def __call__(
            self, 
            prompt,
            negative_prompt = "", 
            width = 512, 
            height = 512, 
            num_inference_steps = 40, 
            guidance_scale = 5, 
            seed = None,
            callback = None,
            callback_userdata = None
            ):
        
        if seed is not None:
            torch.manual_seed(seed)

        latent = self.get_empty_latent(width, height)
        conditioning = self.get_cond_pt(prompt)
        neg_cond = self.get_cond_pt(negative_prompt)
        sampled_latent = self.do_sampling(latent, seed, conditioning, neg_cond, num_inference_steps, guidance_scale, callback, callback_userdata)
        image = self.vae_decode(sampled_latent)
        
        return image
            

    def encode_token_weights(self, model, token_weight_pairs):
        tokens = list(map(lambda a: a[0], token_weight_pairs[0]))
        outs = model(torch.LongTensor([tokens]))
        out = torch.from_numpy(outs[0])
        pooled = None
        if len(outs) > 1:
            pooled = torch.from_numpy(outs[1])

        if pooled is not None:
            first_pooled = pooled[0:1].cpu()
        else:
            first_pooled = pooled
        output = [out[0:1]]
        return torch.cat(output, dim=-2).cpu(), first_pooled

    def sample_euler(self, x, sigmas, callback, callback_userdata, extra_args=None):
        """Implements Algorithm 2 (Euler steps) from Karras et al. (2022)."""
        extra_args = {} if extra_args is None else extra_args
        s_in = x.new_ones([x.shape[0]])
        for i in tqdm.tqdm(range(len(sigmas) - 1)):
            sigma_hat = sigmas[i]
            denoised = self.denoiser(x, sigma_hat * s_in, **extra_args)
            d = to_d(x, sigma_hat, denoised)
            dt = sigmas[i + 1] - sigma_hat
            # Euler method
            x = x + d * dt
        return x

    def denoiser(self, x, timestep, cond, uncond, cond_scale):
        sigma = torch.cat([timestep, timestep])
        timestep_ = self.model_sampling.timestep(sigma).float()
        context=torch.cat([cond["c_crossattn"], uncond["c_crossattn"]])
        y=torch.cat([cond["y"], uncond["y"]])
        model_output = torch.from_numpy(self.sd3([torch.cat([x, x]), timestep_, y, context])[0])
        batched = self.model_sampling.calculate_denoised(sigma, model_output, x)
        pos_out, neg_out = batched[0], batched[1]
        scaled = neg_out + (pos_out - neg_out) * cond_scale
        return scaled

    def get_cond_pt(self, prompt):
        print("Encode prompt...")
        tokens = self.tokenizer.tokenize_with_weights(prompt)
        l_out, l_pooled = self.encode_token_weights(self.clip_l, tokens["l"])
        g_out, g_pooled = self.encode_token_weights(self.clip_g, tokens["g"])
        t5_out, t5_pooled = self.encode_token_weights(self.t5xxl, tokens["t5xxl"])
        lg_out = torch.cat([l_out, g_out], dim=-1)
        lg_out = torch.nn.functional.pad(lg_out, (0, 4096 - lg_out.shape[-1]))
        return torch.cat([lg_out, t5_out], dim=-2), torch.cat((l_pooled, g_pooled), dim=-1)
        

@torch.no_grad()
def main(prompt=PROMPT, width=WIDTH, height=HEIGHT, steps=STEPS, cfg_scale=CFG_SCALE, shift=SHIFT, model=MODEL, seed=SEED, output=OUTPUT, device=DEVICE):
    inferencer = StableDiffusionThreeEngine(model, shift, device)
    inferencer.load(m)
    inferencer.gen_image(prompt, width, height, steps, cfg_scale, seed, output)

