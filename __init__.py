"""
comfyui-sage-guard
==================
Non-invasive SageAttention compatibility guard for ComfyUI v0.28.0+.

Automatically detects incompatible attention patterns:
- GQA (Grouped Query Attention): q_heads != k_heads
- Non-standard head_dim: not in {64, 96, 128}
- Complex attention masks
- Non-CUDA devices

And falls back to native PyTorch manual attention implementation
without modifying ComfyUI core or virtual environment packages.

Compatible with:
- ComfyUI >= v0.28.0 (GQA on all attention backends)
- SageAttention >= 1.0.0
- PyTorch >= 2.0

Author: love530love
Repository: https://github.com/love530love/comfyui-sage-guard
License: MIT
"""

import torch
import torch.nn.functional as F

__version__ = "1.0.0"

# ============ Configuration ============
SAGE_SUPPORTED_HEAD_DIMS = {64, 96, 128}
VERBOSE = True
_LOGGED_TYPES = set()


def _log(msg):
    """Log with deduplication: same message type only prints once."""
    if not VERBOSE:
        return
    msg_type = msg.split("：")[0] if "：" in msg else msg[:25]
    if msg_type not in _LOGGED_TYPES:
        _LOGGED_TYPES.add(msg_type)
        print(f"\033[94m[SageGuard]\033[0m {msg}")


# ============ Pure PyTorch Manual Attention Fallback ============
def _fallback_attention(
    q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kwargs
):
    """
    Pure PyTorch manual attention implementation.
    Handles GQA by repeating k/v heads to match q heads.
    """
    q_heads, k_heads = q.size(1), k.size(1)

    # GQA: repeat k/v to match q heads
    if q_heads != k_heads:
        repeat_factor = q_heads // k_heads
        k = k.repeat_interleave(repeat_factor, dim=1)
        v = v.repeat_interleave(repeat_factor, dim=1)
        _log(
            f"GQA fallback: k/v repeated {repeat_factor}x (q_heads={q_heads}, k_heads={k_heads})"
        )

    if scale is None:
        scale = q.size(-1) ** -0.5

    scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    if attn_mask is not None:
        scores = scores + attn_mask

    if is_causal:
        seq_len_q = scores.size(-2)
        seq_len_k = scores.size(-1)
        causal_mask = torch.triu(
            torch.ones(seq_len_q, seq_len_k, device=scores.device, dtype=torch.bool),
            diagonal=1,
        )
        scores = scores.masked_fill(causal_mask, float("-inf"))

    attn = torch.softmax(scores, dim=-1)
    if dropout_p > 0:
        attn = torch.nn.functional.dropout(attn, p=dropout_p, training=True)

    return torch.matmul(attn, v)


# ============ Compatibility Detection ============
def _should_fallback(q, k, v, attn_mask, **kwargs):
    """Determine if fallback to native PyTorch is needed."""
    if q.device.type != "cuda":
        _log(f"Non-CUDA device: {q.device.type}")
        return True

    if attn_mask is not None:
        _log("Attention mask detected, using fallback")
        return True

    head_dim = q.size(-1)
    if head_dim not in SAGE_SUPPORTED_HEAD_DIMS:
        _log(f"Unsupported head_dim: {head_dim}, using fallback")
        return True

    if q.dim() >= 2 and k.dim() >= 2 and q.size(1) != k.size(1):
        _log(f"GQA detected: q_heads={q.size(1)}, k_heads={k.size(1)}, using fallback")
        return True

    return False


# ============ SageGuard Wrapper ============
class SageGuardWrapper:
    def __init__(self):
        self._current_sdpa = F.scaled_dot_product_attention
        self._is_patched = False

    def __call__(
        self,
        q,
        k,
        v,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
        **kwargs,
    ):
        if _should_fallback(q, k, v, attn_mask, **kwargs):
            return _fallback_attention(
                q, k, v, attn_mask, dropout_p, is_causal, scale, **kwargs
            )

        try:
            return self._current_sdpa(
                q, k, v, attn_mask, dropout_p, is_causal, scale, **kwargs
            )
        except RuntimeError as e:
            err = str(e)
            if "size of tensor" in err or "head_dim" in err or "unsupported" in err:
                _log(f"SageAttention error, falling back: {err[:60]}")
                return _fallback_attention(
                    q, k, v, attn_mask, dropout_p, is_causal, scale, **kwargs
                )
            raise

    def patch(self):
        if self._is_patched:
            return
        self._current_sdpa = F.scaled_dot_product_attention
        F.scaled_dot_product_attention = self
        self._is_patched = True
        _log("F.scaled_dot_product_attention patched successfully")


# ============ Initialize ============
_guard = SageGuardWrapper()
_guard.patch()

# Sync with comfy.ops
try:
    import comfy.ops

    comfy.ops.scaled_dot_product_attention = _guard
    _log("comfy.ops synchronized")
except ImportError:
    pass

# Sync with sageattention.core
try:
    import sageattention.core as sage_core

    sage_core.sageattn = _guard
    _log("sageattention.core synchronized")
except ImportError:
    pass

# ============ Node Registration ============
# This is a ghost node - no UI, patches loaded automatically
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
