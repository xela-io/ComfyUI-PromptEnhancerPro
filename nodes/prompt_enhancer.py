"""
Prompt Enhancer Pro - Main Node Implementation

ComfyUI Custom Node for LLM-powered prompt enhancement using Ollama.
Supports multiple enhancement modes, VRAM management, and reproducible generation.
"""

import random
import json
import urllib.request
import urllib.error
import os
import gc
from typing import Tuple, Optional, List

# ComfyUI model management for VRAM control
try:
    import comfy.model_management as model_management
    HAS_MODEL_MANAGEMENT = True
except ImportError:
    HAS_MODEL_MANAGEMENT = False
    print("[PromptEnhancerPro] Warning: comfy.model_management not available")

from ..utils import (
    OllamaClient,
    VRAMManager,
    build_system_prompt,
    list_built_in_templates,
    list_model_presets,
)


# Default Ollama URL - can be overridden via environment or config
DEFAULT_OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://host.docker.internal:11434")

# Cache for available models (refreshed on each node load)
_cached_models: List[str] = []
_cache_initialized: bool = False


def get_ollama_models(ollama_url: str = None, timeout: int = 2) -> List[str]:
    """
    Fetch available models from Ollama synchronously.
    """
    global _cached_models, _cache_initialized

    if _cache_initialized and _cached_models:
        return _cached_models

    fallback_models = [
        "qwen2.5:7b-instruct",
        "qwen2.5:14b-instruct",
        "llama3.2:8b",
        "mistral:7b",
        "gemma2:9b",
    ]

    urls_to_try = []
    if ollama_url:
        urls_to_try.append(ollama_url)
    urls_to_try.extend([
        DEFAULT_OLLAMA_URL,
        "http://host.docker.internal:11434",
        "http://localhost:11434",
        "http://172.17.0.1:11434",
        "http://172.18.0.1:11434",
    ])

    seen = set()
    urls_to_try = [u for u in urls_to_try if not (u in seen or seen.add(u))]

    for url in urls_to_try:
        try:
            api_url = f"{url.rstrip('/')}/api/tags"
            req = urllib.request.Request(api_url, method="GET")
            req.add_header("Accept", "application/json")

            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    models = [m.get("name", "") for m in data.get("models", []) if m.get("name")]

                    if models:
                        _cached_models = sorted(models)
                        _cache_initialized = True
                        print(f"[PromptEnhancerPro] Found {len(models)} Ollama models at {url}")
                        return _cached_models

        except urllib.error.URLError:
            continue
        except Exception:
            continue

    print("[PromptEnhancerPro] Ollama not reachable, using fallback model list")
    return fallback_models


def refresh_model_cache(ollama_url: str = DEFAULT_OLLAMA_URL) -> List[str]:
    """Force refresh the model cache."""
    global _cached_models, _cache_initialized
    _cache_initialized = False
    _cached_models = []
    return get_ollama_models(ollama_url)


class PromptEnhancerProError(Exception):
    """Custom exception for Prompt Enhancer Pro errors"""
    pass


def get_effective_seed(seed_input: int) -> int:
    """Get effective seed value."""
    if seed_input == -1:
        return random.randint(0, 2147483647)
    return seed_input


def free_comfyui_vram() -> bool:
    """
    Free ComfyUI's VRAM by unloading all models.

    Returns:
        True if VRAM was freed successfully, False otherwise
    """
    if not HAS_MODEL_MANAGEMENT:
        print("[PromptEnhancerPro] Cannot free VRAM: model_management not available")
        return False

    try:
        print("[PromptEnhancerPro] Freeing ComfyUI VRAM...")

        # Unload all models from VRAM
        model_management.unload_all_models()

        # Clear CUDA cache
        model_management.soft_empty_cache()

        # Force Python garbage collection
        gc.collect()

        # Try to get VRAM info
        try:
            free_vram = model_management.get_free_memory() / (1024**3)
            print(f"[PromptEnhancerPro] VRAM freed. Available: {free_vram:.1f} GB")
        except:
            print("[PromptEnhancerPro] VRAM freed successfully")

        return True

    except Exception as e:
        print(f"[PromptEnhancerPro] Error freeing VRAM: {e}")
        return False


class PromptEnhancerPro:
    """
    Main Prompt Enhancer Pro Node
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        enhancement_modes = ["character_builder", "freeform", "scene_builder", "portrait", "custom_template"]
        available_models = get_ollama_models()

        return {
            "required": {
                "base_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Enter your base prompt here..."
                }),
                "enhancement_mode": (enhancement_modes, {
                    "default": "character_builder"
                }),
                "model_name": (available_models, {
                    "default": available_models[0] if available_models else "qwen2.5:7b-instruct"
                }),
            },
            "optional": {
                "custom_template": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Enter custom system prompt when using 'custom_template' mode..."
                }),
                "context_info": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Additional context (format, style preferences, etc.)"
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 2147483647,
                    "step": 1,
                    "display": "number"
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "display": "slider"
                }),
                "max_tokens": ("INT", {
                    "default": 500,
                    "min": 50,
                    "max": 2000,
                    "step": 50,
                    "display": "number"
                }),
                "unload_after": ("BOOLEAN", {
                    "default": True,
                    "label_on": "Unload Model",
                    "label_off": "Keep Loaded"
                }),
                "free_vram_before": ("BOOLEAN", {
                    "default": True,
                    "label_on": "Free VRAM First",
                    "label_off": "Keep Models"
                }),
                "ollama_url": ("STRING", {
                    "default": "http://host.docker.internal:11434",
                    "placeholder": "Ollama API URL"
                }),
                "timeout": ("INT", {
                    "default": 120,
                    "min": 30,
                    "max": 600,
                    "step": 10,
                    "display": "number"
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "INT")
    RETURN_NAMES = ("enhanced_prompt", "original_prompt", "model_used", "generation_seed")
    FUNCTION = "enhance_prompt"
    CATEGORY = "prompt/enhancement"
    OUTPUT_NODE = False

    def enhance_prompt(
        self,
        base_prompt: str,
        enhancement_mode: str,
        model_name: str,
        custom_template: str = "",
        context_info: str = "",
        seed: int = -1,
        temperature: float = 0.7,
        max_tokens: int = 500,
        unload_after: bool = True,
        free_vram_before: bool = True,
        ollama_url: str = "http://host.docker.internal:11434",
        timeout: int = 120,
    ) -> Tuple[str, str, str, int]:
        """Enhance a prompt using Ollama LLM."""

        if not base_prompt or not base_prompt.strip():
            print("[PromptEnhancerPro] Warning: Empty prompt provided")
            return ("", "", model_name, seed if seed >= 0 else 0)

        # Free ComfyUI VRAM before Ollama request if enabled
        if free_vram_before:
            free_comfyui_vram()

        effective_seed = get_effective_seed(seed)

        system_prompt = build_system_prompt(
            mode=enhancement_mode,
            custom_template=custom_template if custom_template else None,
            context_info=context_info if context_info else None,
        )

        client = OllamaClient(base_url=ollama_url)

        try:
            # Check connection first
            connected, conn_msg = client.check_connection()
            if not connected:
                print(f"[PromptEnhancerPro] Warning: {conn_msg}")
                print("[PromptEnhancerPro] Returning original prompt as fallback")
                return (base_prompt, base_prompt, model_name, effective_seed)

            keep_alive = "0" if unload_after else "5m"

            print(f"[PromptEnhancerPro] Enhancing prompt with {model_name}...")
            response = client.generate(
                model=model_name,
                prompt=base_prompt,
                system=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=effective_seed,
                keep_alive=keep_alive,
                timeout=timeout,
            )

            if response.success:
                enhanced = response.content.strip()
                if not enhanced:
                    print("[PromptEnhancerPro] Warning: Empty response from LLM")
                    return (base_prompt, base_prompt, response.model, effective_seed)

                print(f"[PromptEnhancerPro] Enhancement complete (seed: {effective_seed})")

                if unload_after:
                    print(f"[PromptEnhancerPro] Model {model_name} will be unloaded (keep_alive=0)")

                return (enhanced, base_prompt, response.model, effective_seed)
            else:
                error_msg = response.error or "Unknown error"
                print(f"[PromptEnhancerPro] Error: {error_msg}")
                print("[PromptEnhancerPro] Returning original prompt as fallback")
                return (base_prompt, base_prompt, model_name, effective_seed)

        except Exception as e:
            print(f"[PromptEnhancerPro] Exception: {str(e)}")
            print("[PromptEnhancerPro] Returning original prompt as fallback")
            return (base_prompt, base_prompt, model_name, effective_seed)


class PromptEnhancerProAdvanced(PromptEnhancerPro):
    """Advanced version with additional controls."""

    @classmethod
    def INPUT_TYPES(cls):
        base_inputs = PromptEnhancerPro.INPUT_TYPES()

        base_inputs["optional"]["negative_prompt_mode"] = ("BOOLEAN", {
            "default": False,
            "label_on": "Also Generate Negative",
            "label_off": "Positive Only"
        })
        base_inputs["optional"]["language"] = (["auto", "english", "german"], {
            "default": "auto"
        })
        base_inputs["optional"]["verbosity"] = (["concise", "normal", "detailed"], {
            "default": "normal"
        })

        return base_inputs

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "INT")
    RETURN_NAMES = ("enhanced_prompt", "negative_prompt", "original_prompt", "model_used", "generation_seed")
    FUNCTION = "enhance_prompt_advanced"
    CATEGORY = "prompt/enhancement"

    def enhance_prompt_advanced(
        self,
        base_prompt: str,
        enhancement_mode: str,
        model_name: str,
        custom_template: str = "",
        context_info: str = "",
        seed: int = -1,
        temperature: float = 0.7,
        max_tokens: int = 500,
        unload_after: bool = True,
        free_vram_before: bool = True,
        ollama_url: str = "http://host.docker.internal:11434",
        timeout: int = 120,
        negative_prompt_mode: bool = False,
        language: str = "auto",
        verbosity: str = "normal",
    ) -> Tuple[str, str, str, str, int]:
        """Advanced prompt enhancement with negative prompt generation."""

        context_parts = []
        if context_info:
            context_parts.append(context_info)

        if language != "auto":
            lang_instruction = {
                "english": "Output the enhanced prompt in English.",
                "german": "Output the enhanced prompt in German (Deutsch).",
            }
            context_parts.append(lang_instruction.get(language, ""))

        if verbosity == "concise":
            context_parts.append("Keep the output very concise, under 100 words.")
        elif verbosity == "detailed":
            context_parts.append("Provide a detailed, comprehensive prompt with rich descriptions.")

        combined_context = "\n".join(context_parts) if context_parts else ""

        enhanced, original, model, eff_seed = self.enhance_prompt(
            base_prompt=base_prompt,
            enhancement_mode=enhancement_mode,
            model_name=model_name,
            custom_template=custom_template,
            context_info=combined_context,
            seed=seed,
            temperature=temperature,
            max_tokens=max_tokens,
            unload_after=False if negative_prompt_mode else unload_after,
            free_vram_before=free_vram_before,
            ollama_url=ollama_url,
            timeout=timeout,
        )

        negative_prompt = ""

        if negative_prompt_mode and enhanced != original:
            neg_system = build_system_prompt(
                mode="negative_prompt_helper",
                context_info=f"Original prompt: {base_prompt}\nEnhanced prompt: {enhanced}"
            )

            client = OllamaClient(base_url=ollama_url)
            try:
                neg_seed = (eff_seed + 1) % 2147483647

                response = client.generate(
                    model=model_name,
                    prompt=f"Generate a negative prompt for: {enhanced}",
                    system=neg_system,
                    temperature=temperature * 0.8,
                    max_tokens=200,
                    seed=neg_seed,
                    keep_alive="0" if unload_after else "5m",
                    timeout=timeout,
                )

                if response.success and response.content:
                    negative_prompt = response.content.strip()
                    print(f"[PromptEnhancerPro] Negative prompt generated")

            except Exception as e:
                print(f"[PromptEnhancerPro] Error generating negative prompt: {e}")

        return (enhanced, negative_prompt, original, model, eff_seed)


class OllamaConnectionChecker:
    """Utility node to check Ollama connection status."""

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ollama_url": ("STRING", {
                    "default": "http://host.docker.internal:11434",
                    "placeholder": "Ollama API URL"
                }),
            }
        }

    RETURN_TYPES = ("BOOLEAN", "STRING", "STRING")
    RETURN_NAMES = ("is_connected", "status_message", "available_models")
    FUNCTION = "check_connection"
    CATEGORY = "prompt/enhancement"
    OUTPUT_NODE = True

    def check_connection(
        self,
        ollama_url: str = "http://host.docker.internal:11434",
    ) -> Tuple[bool, str, str]:
        """Check if Ollama is reachable and list available models."""
        client = OllamaClient(base_url=ollama_url)

        try:
            connected, message = client.check_connection()

            if connected:
                success, models = client.list_models()
                if success:
                    models_str = ", ".join(models) if models else "No models installed"
                    return (True, "Connected to Ollama", models_str)
                return (True, "Connected but could not list models", "")
            else:
                return (False, message, "")

        except Exception as e:
            return (False, f"Error: {str(e)}", "")


class OllamaModelManager:
    """Node for managing Ollama model VRAM allocation."""

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        available_models = get_ollama_models()

        return {
            "required": {
                "action": (["check_loaded", "load_model", "unload_model", "unload_all", "refresh_models"],),
                "model_name": (available_models, {
                    "default": available_models[0] if available_models else "qwen2.5:7b-instruct"
                }),
            },
            "optional": {
                "ollama_url": ("STRING", {
                    "default": "http://host.docker.internal:11434"
                }),
                "keep_alive": ("STRING", {
                    "default": "5m",
                    "placeholder": "e.g., 5m, 1h, 0"
                }),
            }
        }

    RETURN_TYPES = ("BOOLEAN", "STRING", "STRING")
    RETURN_NAMES = ("success", "message", "loaded_models")
    FUNCTION = "manage_model"
    CATEGORY = "prompt/enhancement"
    OUTPUT_NODE = True

    def manage_model(
        self,
        action: str,
        model_name: str,
        ollama_url: str = "http://host.docker.internal:11434",
        keep_alive: str = "5m",
    ) -> Tuple[bool, str, str]:
        """Manage Ollama model loading/unloading."""
        manager = VRAMManager(ollama_url=ollama_url)

        try:
            if action == "check_loaded":
                success, models = manager.get_loaded_models()
                if success:
                    if models:
                        model_names = [m.name for m in models]
                        return (True, f"Loaded: {len(models)} model(s)", ", ".join(model_names))
                    return (True, "No models currently loaded", "")
                return (False, "Could not check loaded models", "")

            elif action == "load_model":
                success, message = manager.load_model(model_name, keep_alive)
                if success:
                    _, models = manager.get_loaded_models()
                    model_names = [m.name for m in models] if models else []
                    return (True, message, ", ".join(model_names))
                return (False, message, "")

            elif action == "unload_model":
                success, message = manager.unload_model(model_name)
                if success:
                    _, models = manager.get_loaded_models()
                    model_names = [m.name for m in models] if models else []
                    return (True, message, ", ".join(model_names))
                return (False, message, "")

            elif action == "unload_all":
                success, unloaded = manager.unload_all_models()
                if success:
                    if unloaded:
                        return (True, f"Unloaded: {', '.join(unloaded)}", "")
                    return (True, "No models were loaded", "")
                return (False, "Could not unload models", "")

            elif action == "refresh_models":
                models = refresh_model_cache(ollama_url)
                if models:
                    return (True, f"Refreshed: {len(models)} models found", ", ".join(models))
                return (False, "No models found or Ollama not reachable", "")

            else:
                return (False, f"Unknown action: {action}", "")

        except Exception as e:
            return (False, f"Error: {str(e)}", "")


# ============================================================================
# Context File Loader
# ============================================================================

# Directory for context files
CONTEXT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "context")


def get_context_files() -> List[str]:
    """Get list of available context files."""
    if not os.path.exists(CONTEXT_DIR):
        os.makedirs(CONTEXT_DIR, exist_ok=True)
        return ["(no files)"]

    files = []
    for f in os.listdir(CONTEXT_DIR):
        if f.endswith((".txt", ".md")):
            files.append(f)

    return sorted(files) if files else ["(no files)"]


class ContextFileLoader:
    """
    Load context from a text file to use with Prompt Enhancer Pro.

    Place .txt or .md files in the 'context/' folder within the node directory.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        context_files = get_context_files()

        return {
            "required": {
                "file_name": (context_files, {
                    "default": context_files[0] if context_files else "(no files)"
                }),
            },
            "optional": {
                "additional_context": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Additional context to append..."
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("context",)
    FUNCTION = "load_context"
    CATEGORY = "prompt/enhancement"

    def load_context(
        self,
        file_name: str,
        additional_context: str = "",
    ) -> Tuple[str]:
        """Load context from file."""

        if file_name == "(no files)":
            print("[ContextFileLoader] No context files found in context/ folder")
            return (additional_context,)

        file_path = os.path.join(CONTEXT_DIR, file_name)

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()

            print(f"[ContextFileLoader] Loaded {file_name} ({len(content)} chars)")

            if additional_context:
                content = f"{content}\n\n{additional_context}"

            return (content,)

        except Exception as e:
            print(f"[ContextFileLoader] Error loading {file_name}: {e}")
            return (additional_context,)


# ============================================================================
# Node Registration
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "PromptEnhancerPro": PromptEnhancerPro,
    "PromptEnhancerProAdvanced": PromptEnhancerProAdvanced,
    "OllamaConnectionChecker": OllamaConnectionChecker,
    "OllamaModelManager": OllamaModelManager,
    "ContextFileLoader": ContextFileLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptEnhancerPro": "Prompt Enhancer Pro",
    "PromptEnhancerProAdvanced": "Prompt Enhancer Pro (Advanced)",
    "OllamaConnectionChecker": "Ollama Connection Checker",
    "OllamaModelManager": "Ollama Model Manager",
    "ContextFileLoader": "Context File Loader",
}
