# SageAttention Guard for ComfyUI

[![Version](https://img.shields.io/badge/version-1.0.0-blue)](https://github.com/love530love/comfyui-sage-guard)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-%3E%3Dv0.28.0-green)](https://github.com/comfyanonymous/ComfyUI)

Non-invasive compatibility guard for SageAttention in ComfyUI v0.28.0+.

## Problem

ComfyUI v0.28.0 introduced GQA (Grouped Query Attention) on all attention backends.
SageAttention's CUDA kernel does not support GQA structures (e.g., q_heads=32, k_heads=8),
causing `RuntimeError: The size of tensor a (32) must match the size of tensor b (8)`.

## Solution

This ghost node automatically detects incompatible attention patterns and falls back
to native PyTorch SDPA without modifying ComfyUI core or virtual environment packages.

## Features

- ✅ GQA auto-detection and fallback
- ✅ Non-standard head_dim support (not in {64, 96, 128})
- ✅ Complex attention mask handling
- ✅ Non-CUDA device support
- ✅ Zero UI footprint (ghost node)
- ✅ Non-invasive (no core modifications)

## Installation

### Method 1: ComfyUI Manager
Search for "SageAttention Guard" in ComfyUI Manager and install.

### Method 2: Git Clone
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/love530love/comfyui-sage-guard.git
```

### Method 3: Download
Download and extract to `ComfyUI/custom_nodes/comfyui-sage-guard/`.

## Compatibility

| ComfyUI Version | Status |
|----------------|--------|
| v0.28.0+ | ✅ Fully supported |
| v0.27.x | ✅ Compatible (GQA not used) |

## Related

- [ComfyUI-FixFlashAttnSchema](https://github.com/love530love/ComfyUI-FixFlashAttnSchema) - Fixes Flash Attention schema registration bug

## License

MIT
