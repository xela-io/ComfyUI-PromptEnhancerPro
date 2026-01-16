"""
Utility modules for Prompt Enhancer Pro
"""

from .ollama_client import OllamaClient, OllamaResponse, run_async
from .vram_manager import VRAMManager, ModelInfo
from .templates import (
    get_template,
    build_system_prompt,
    list_built_in_templates,
    list_external_templates,
    get_all_available_templates,
    load_template_from_file,
    get_model_preset,
    list_model_presets,
    MODEL_PRESETS,
    BUILT_IN_TEMPLATES,
)

__all__ = [
    "OllamaClient",
    "OllamaResponse",
    "VRAMManager",
    "ModelInfo",
    "run_async",
    "get_template",
    "build_system_prompt",
    "list_built_in_templates",
    "list_external_templates",
    "get_all_available_templates",
    "load_template_from_file",
    "get_model_preset",
    "list_model_presets",
    "MODEL_PRESETS",
    "BUILT_IN_TEMPLATES",
]
