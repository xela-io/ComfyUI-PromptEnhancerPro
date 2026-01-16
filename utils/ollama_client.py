"""
Ollama API Client for Prompt Enhancer Pro

Synchronous HTTP client for communicating with Ollama API.
Uses urllib for maximum compatibility with ComfyUI's event loop.
"""

import json
import urllib.request
import urllib.error
from typing import Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class OllamaResponse:
    """Response from Ollama API"""
    success: bool
    content: str
    model: str
    error: Optional[str] = None
    total_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    eval_count: Optional[int] = None


class OllamaClient:
    """
    Synchronous client for Ollama API communication.
    Uses urllib to avoid event loop issues with ComfyUI.
    """

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip("/")

    def check_connection(self, timeout: int = 5) -> Tuple[bool, str]:
        """
        Check if Ollama is reachable.

        Returns:
            Tuple of (success, message)
        """
        try:
            req = urllib.request.Request(
                f"{self.base_url}/api/tags",
                method="GET"
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status == 200:
                    return True, "Ollama is running"
                return False, f"Ollama returned status {resp.status}"
        except urllib.error.URLError as e:
            return False, f"Cannot connect to Ollama: {e.reason}"
        except Exception as e:
            return False, f"Error: {str(e)}"

    def list_models(self, timeout: int = 10) -> Tuple[bool, List[str]]:
        """
        List available models.

        Returns:
            Tuple of (success, list of model names)
        """
        try:
            req = urllib.request.Request(
                f"{self.base_url}/api/tags",
                method="GET"
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    models = [m.get("name", "") for m in data.get("models", []) if m.get("name")]
                    return True, models
                return False, []
        except Exception as e:
            print(f"[PromptEnhancerPro] Error listing models: {e}")
            return False, []

    def check_model_loaded(self, model_name: str, timeout: int = 5) -> bool:
        """
        Check if a model is currently loaded in VRAM.
        """
        try:
            req = urllib.request.Request(
                f"{self.base_url}/api/ps",
                method="GET"
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    loaded_models = [m.get("name", "") for m in data.get("models", [])]
                    return any(model_name in m for m in loaded_models)
                return False
        except Exception:
            return False

    def generate(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 500,
        seed: Optional[int] = None,
        keep_alive: str = "5m",
        timeout: int = 120
    ) -> OllamaResponse:
        """
        Generate text using Ollama.

        Args:
            model: Model name (e.g., "qwen2.5:7b-instruct")
            prompt: User prompt
            system: Optional system prompt
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
            seed: Random seed for reproducibility
            keep_alive: How long to keep model loaded (e.g., "5m", "0" to unload)
            timeout: Request timeout in seconds

        Returns:
            OllamaResponse with generation result
        """
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
            "keep_alive": keep_alive,
        }

        if system:
            payload["system"] = system

        if seed is not None and seed >= 0:
            payload["options"]["seed"] = seed

        try:
            req = urllib.request.Request(
                f"{self.base_url}/api/generate",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST"
            )

            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    return OllamaResponse(
                        success=True,
                        content=data.get("response", ""),
                        model=data.get("model", model),
                        total_duration=data.get("total_duration"),
                        prompt_eval_count=data.get("prompt_eval_count"),
                        eval_count=data.get("eval_count"),
                    )
                else:
                    return OllamaResponse(
                        success=False,
                        content="",
                        model=model,
                        error=f"HTTP {resp.status}"
                    )

        except urllib.error.HTTPError as e:
            error_body = ""
            try:
                error_body = e.read().decode("utf-8")
            except:
                pass
            return OllamaResponse(
                success=False,
                content="",
                model=model,
                error=f"HTTP {e.code}: {error_body}"
            )
        except urllib.error.URLError as e:
            return OllamaResponse(
                success=False,
                content="",
                model=model,
                error=f"Connection error: {e.reason}"
            )
        except TimeoutError:
            return OllamaResponse(
                success=False,
                content="",
                model=model,
                error=f"Request timed out after {timeout}s"
            )
        except Exception as e:
            return OllamaResponse(
                success=False,
                content="",
                model=model,
                error=str(e)
            )

    def close(self):
        """No-op for compatibility. Synchronous client doesn't need cleanup."""
        pass
