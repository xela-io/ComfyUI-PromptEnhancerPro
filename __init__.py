"""
Prompt Enhancer Pro - ComfyUI Custom Node

LLM-powered prompt enhancement using Ollama with intelligent VRAM management.

## Nodes

1. **Prompt Enhancer Pro** - Main node for prompt enhancement
   - Multiple enhancement modes (Character Builder, Freeform, Scene Builder, Portrait)
   - Seed support for reproducible results
   - Automatic VRAM management

2. **Prompt Enhancer Pro (Advanced)** - Extended version with additional controls
   - Negative prompt generation
   - Language selection
   - Verbosity control

3. **Ollama Connection Checker** - Utility to verify Ollama connectivity
   - Lists available models
   - Connection status

4. **Ollama Model Manager** - VRAM management utility
   - Load/unload models manually
   - Check loaded models
   - Free VRAM before image generation

## Installation

1. Copy this folder to `ComfyUI/custom_nodes/`
2. Install dependencies: `pip install aiohttp`
3. Ensure Ollama is running: `ollama serve`

## Requirements

- Python 3.11+
- Ollama installed and running
- Recommended models: qwen2.5:7b-instruct, llama3.2:8b, mistral:7b

## Usage

1. Add "Prompt Enhancer Pro" node to your workflow
2. Connect your base prompt input
3. Select enhancement mode
4. Choose Ollama model
5. Enable "Unload After" to free VRAM for image generation

## VRAM Management

The node automatically unloads the LLM after generation when `unload_after` is enabled.
This ensures maximum VRAM is available for subsequent image generation.

For manual control, use the "Ollama Model Manager" node.

## Author

Created by xela-io
https://github.com/xela-io
"""

from .nodes.prompt_enhancer import (
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

__version__ = "1.0.0"
WEB_DIRECTORY = None
