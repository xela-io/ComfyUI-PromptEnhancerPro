"""
Prompt Enhancer Pro - Main Node Implementation

ComfyUI Custom Node for LLM-powered prompt enhancement using Ollama.
Supports multiple enhancement modes, VRAM management, and reproducible generation.
"""

import random
import json
import urllib.request
import urllib.error
from typing import Tuple, Optional, List

from ..utils import (
    OllamaClient,
    VRAMManager,
    run_async,
    build_system_prompt,
    list_built_in_templates,
    list_model_presets,
)


# Default Ollama URL - can be overridden via environment or config
# Use host.docker.internal for Docker containers, localhost for native
import os
DEFAULT_OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# Cache for available models (refreshed on each node load)
_cached_models: List[str] = []
_cache_initialized: bool = False


def get_ollama_models(ollama_url: str = None, timeout: int = 2) -> List[str]:
    """
    Fetch available models from Ollama synchronously.

    Called at node load time to populate the dropdown.
    Uses a short timeout to avoid blocking ComfyUI startup.
    Tries multiple endpoints (localhost, Docker gateway, etc.)

    Args:
        ollama_url: Ollama API endpoint (None = auto-detect)
        timeout: Request timeout in seconds (short for startup)

    Returns:
        List of model names, or fallback list if Ollama unreachable
    """
    global _cached_models, _cache_initialized

    # Return cache if already initialized
    if _cache_initialized and _cached_models:
        return _cached_models

    # Fallback models if Ollama is not reachable
    fallback_models = [
        "qwen2.5:7b-instruct",
        "qwen2.5:14b-instruct",
        "llama3.2:8b",
        "mistral:7b",
        "gemma2:9b",
    ]

    # URLs to try (in order of preference)
    urls_to_try = []
    if ollama_url:
        urls_to_try.append(ollama_url)
    urls_to_try.extend([
        DEFAULT_OLLAMA_URL,
        "http://host.docker.internal:11434",  # Docker with extra_hosts
        "http://localhost:11434",  # Native installation
        "http://172.17.0.1:11434",  # Docker bridge gateway (Linux)
        "http://172.18.0.1:11434",  # Alternative Docker network
    ])

    # Remove duplicates while preserving order
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
            continue  # Try next URL
        except Exception:
            continue  # Try next URL

    # Return fallback if Ollama not available
    print("[PromptEnhancerPro] Ollama not reachable, using fallback model list")
    return fallback_models


def refresh_model_cache(ollama_url: str = DEFAULT_OLLAMA_URL) -> List[str]:
    """
    Force refresh the model cache.

    Can be called to update the model list without restarting ComfyUI.
    """
    global _cached_models, _cache_initialized
    _cache_initialized = False
    _cached_models = []
    return get_ollama_models(ollama_url)


class PromptEnhancerProError(Exception):
    """Custom exception for Prompt Enhancer Pro errors"""
    pass


def get_effective_seed(seed_input: int) -> int:
    """
    Get effective seed value.

    Args:
        seed_input: User-provided seed (-1 for random)

    Returns:
        Valid seed value
    """
    if seed_input == -1:
        return random.randint(0, 2147483647)
    return seed_input


class PromptEnhancerPro:
    """
    Main Prompt Enhancer Pro Node

    Enhances prompts using Ollama LLM with built-in templates
    and automatic VRAM management.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        enhancement_modes = ["character_builder", "freeform", "scene_builder", "portrait", "custom_template"]

        # Fetch available Ollama models dynamically
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
        ollama_url: str = "http://localhost:11434",
        timeout: int = 120,
    ) -> Tuple[str, str, str, int]:
        """
        Enhance a prompt using Ollama LLM.

        Args:
            base_prompt: The original prompt to enhance
            enhancement_mode: Type of enhancement to apply
            model_name: Ollama model to use
            custom_template: Custom system prompt (for custom_template mode)
            context_info: Additional context to include
            seed: Random seed (-1 for random)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            unload_after: Whether to unload model after generation
            ollama_url: Ollama API endpoint
            timeout: Request timeout in seconds

        Returns:
            Tuple of (enhanced_prompt, original_prompt, model_used, seed_used)
        """
        # Handle empty prompt
        if not base_prompt or not base_prompt.strip():
            print("[PromptEnhancerPro] Warning: Empty prompt provided")
            return ("", "", model_name, seed if seed >= 0 else 0)

        # Get effective seed
        effective_seed = get_effective_seed(seed)

        # Build system prompt
        system_prompt = build_system_prompt(
            mode=enhancement_mode,
            custom_template=custom_template if custom_template else None,
            context_info=context_info if context_info else None,
        )

        # Create client
        client = OllamaClient(base_url=ollama_url)

        try:
            # Check connection first
            connected, conn_msg = run_async(client.check_connection())
            if not connected:
                print(f"[PromptEnhancerPro] Warning: {conn_msg}")
                print("[PromptEnhancerPro] Returning original prompt as fallback")
                return (base_prompt, base_prompt, model_name, effective_seed)

            # Determine keep_alive based on unload_after setting
            keep_alive = "0" if unload_after else "5m"

            # Generate enhanced prompt
            print(f"[PromptEnhancerPro] Enhancing prompt with {model_name}...")
            response = run_async(client.generate(
                model=model_name,
                prompt=base_prompt,
                system=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=effective_seed,
                keep_alive=keep_alive,
                timeout=timeout,
            ))

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

        finally:
            # Clean up client session
            try:
                run_async(client.close())
            except Exception:
                pass


class PromptEnhancerProAdvanced(PromptEnhancerPro):
    """
    Advanced version with additional controls and batch support.
    """

    @classmethod
    def INPUT_TYPES(cls):
        base_inputs = PromptEnhancerPro.INPUT_TYPES()

        # Add advanced options
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
        ollama_url: str = "http://localhost:11434",
        timeout: int = 120,
        negative_prompt_mode: bool = False,
        language: str = "auto",
        verbosity: str = "normal",
    ) -> Tuple[str, str, str, str, int]:
        """
        Advanced prompt enhancement with negative prompt generation.
        """
        # Build context with language and verbosity preferences
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

        # Get enhanced prompt using base class method
        enhanced, original, model, eff_seed = self.enhance_prompt(
            base_prompt=base_prompt,
            enhancement_mode=enhancement_mode,
            model_name=model_name,
            custom_template=custom_template,
            context_info=combined_context,
            seed=seed,
            temperature=temperature,
            max_tokens=max_tokens,
            unload_after=False if negative_prompt_mode else unload_after,  # Keep loaded if generating negative
            ollama_url=ollama_url,
            timeout=timeout,
        )

        negative_prompt = ""

        # Generate negative prompt if requested
        if negative_prompt_mode and enhanced != original:
            from ..utils import build_system_prompt

            neg_system = build_system_prompt(
                mode="negative_prompt_helper",
                context_info=f"Original prompt: {base_prompt}\nEnhanced prompt: {enhanced}"
            )

            client = OllamaClient(base_url=ollama_url)
            try:
                # Use different seed for negative prompt
                neg_seed = (eff_seed + 1) % 2147483647

                response = run_async(client.generate(
                    model=model_name,
                    prompt=f"Generate a negative prompt for: {enhanced}",
                    system=neg_system,
                    temperature=temperature * 0.8,  # Slightly lower temp for negative
                    max_tokens=200,
                    seed=neg_seed,
                    keep_alive="0" if unload_after else "5m",
                    timeout=timeout,
                ))

                if response.success and response.content:
                    negative_prompt = response.content.strip()
                    print(f"[PromptEnhancerPro] Negative prompt generated")

            except Exception as e:
                print(f"[PromptEnhancerPro] Error generating negative prompt: {e}")
            finally:
                try:
                    run_async(client.close())
                except Exception:
                    pass

        return (enhanced, negative_prompt, original, model, eff_seed)


class OllamaConnectionChecker:
    """
    Utility node to check Ollama connection status.
    """

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
        ollama_url: str = "http://localhost:11434",
    ) -> Tuple[bool, str, str]:
        """
        Check if Ollama is reachable and list available models.
        """
        client = OllamaClient(base_url=ollama_url)

        try:
            connected, message = run_async(client.check_connection())

            if connected:
                success, models = run_async(client.list_models())
                if success:
                    models_str = ", ".join(models) if models else "No models installed"
                    return (True, "Connected to Ollama", models_str)
                return (True, "Connected but could not list models", "")
            else:
                return (False, message, "")

        except Exception as e:
            return (False, f"Error: {str(e)}", "")

        finally:
            try:
                run_async(client.close())
            except Exception:
                pass


class OllamaModelManager:
    """
    Node for managing Ollama model VRAM allocation.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        # Fetch available Ollama models dynamically
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
        ollama_url: str = "http://localhost:11434",
        keep_alive: str = "5m",
    ) -> Tuple[bool, str, str]:
        """
        Manage Ollama model loading/unloading.
        """
        manager = VRAMManager(ollama_url=ollama_url)

        try:
            if action == "check_loaded":
                success, models = run_async(manager.get_loaded_models())
                if success:
                    if models:
                        model_names = [m.name for m in models]
                        return (True, f"Loaded: {len(models)} model(s)", ", ".join(model_names))
                    return (True, "No models currently loaded", "")
                return (False, "Could not check loaded models", "")

            elif action == "load_model":
                success, message = run_async(manager.load_model(model_name, keep_alive))
                if success:
                    _, models = run_async(manager.get_loaded_models())
                    model_names = [m.name for m in models] if models else []
                    return (True, message, ", ".join(model_names))
                return (False, message, "")

            elif action == "unload_model":
                success, message = run_async(manager.unload_model(model_name))
                if success:
                    _, models = run_async(manager.get_loaded_models())
                    model_names = [m.name for m in models] if models else []
                    return (True, message, ", ".join(model_names))
                return (False, message, "")

            elif action == "unload_all":
                success, unloaded = run_async(manager.unload_all_models())
                if success:
                    if unloaded:
                        return (True, f"Unloaded: {', '.join(unloaded)}", "")
                    return (True, "No models were loaded", "")
                return (False, "Could not unload models", "")

            elif action == "refresh_models":
                # Refresh the model cache
                models = refresh_model_cache(ollama_url)
                if models:
                    return (True, f"Refreshed: {len(models)} models found", ", ".join(models))
                return (False, "No models found or Ollama not reachable", "")

            else:
                return (False, f"Unknown action: {action}", "")

        except Exception as e:
            return (False, f"Error: {str(e)}", "")

        finally:
            try:
                run_async(manager.close())
            except Exception:
                pass


# ============================================================================
# Node Registration
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "PromptEnhancerPro": PromptEnhancerPro,
    "PromptEnhancerProAdvanced": PromptEnhancerProAdvanced,
    "OllamaConnectionChecker": OllamaConnectionChecker,
    "OllamaModelManager": OllamaModelManager,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptEnhancerPro": "Prompt Enhancer Pro",
    "PromptEnhancerProAdvanced": "Prompt Enhancer Pro (Advanced)",
    "OllamaConnectionChecker": "Ollama Connection Checker",
    "OllamaModelManager": "Ollama Model Manager",
}
