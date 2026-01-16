"""
Ollama API Client for Prompt Enhancer Pro

Async HTTP client for communicating with Ollama's OpenAI-compatible API.
Handles model generation, status checks, and error handling.
"""

import asyncio
import json
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    import urllib.request
    import urllib.error

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
    Async client for Ollama API communication.

    Uses aiohttp if available, falls back to urllib for synchronous requests.
    """

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip("/")
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> "aiohttp.ClientSession":
        """Get or create aiohttp session"""
        if not AIOHTTP_AVAILABLE:
            raise RuntimeError("aiohttp not available")
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=120)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self):
        """Close the HTTP session"""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def check_connection(self) -> Tuple[bool, str]:
        """
        Check if Ollama is reachable.

        Returns:
            Tuple of (success, message)
        """
        try:
            if AIOHTTP_AVAILABLE:
                session = await self._get_session()
                async with session.get(f"{self.base_url}/api/tags") as resp:
                    if resp.status == 200:
                        return True, "Ollama is running"
                    return False, f"Ollama returned status {resp.status}"
            else:
                req = urllib.request.Request(f"{self.base_url}/api/tags")
                with urllib.request.urlopen(req, timeout=10) as resp:
                    if resp.status == 200:
                        return True, "Ollama is running"
                    return False, f"Ollama returned status {resp.status}"
        except Exception as e:
            return False, f"Cannot connect to Ollama: {str(e)}"

    async def list_models(self) -> Tuple[bool, list]:
        """
        List available models.

        Returns:
            Tuple of (success, list of model names)
        """
        try:
            if AIOHTTP_AVAILABLE:
                session = await self._get_session()
                async with session.get(f"{self.base_url}/api/tags") as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        models = [m["name"] for m in data.get("models", [])]
                        return True, models
                    return False, []
            else:
                req = urllib.request.Request(f"{self.base_url}/api/tags")
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read().decode())
                    models = [m["name"] for m in data.get("models", [])]
                    return True, models
        except Exception as e:
            print(f"[PromptEnhancerPro] Error listing models: {e}")
            return False, []

    async def check_model_loaded(self, model_name: str) -> bool:
        """
        Check if a model is currently loaded in VRAM.

        Args:
            model_name: Name of the model to check

        Returns:
            True if model is loaded, False otherwise
        """
        try:
            if AIOHTTP_AVAILABLE:
                session = await self._get_session()
                async with session.get(f"{self.base_url}/api/ps") as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        loaded_models = [m.get("name", "") for m in data.get("models", [])]
                        return any(model_name in m for m in loaded_models)
                    return False
            else:
                req = urllib.request.Request(f"{self.base_url}/api/ps")
                with urllib.request.urlopen(req, timeout=10) as resp:
                    data = json.loads(resp.read().decode())
                    loaded_models = [m.get("name", "") for m in data.get("models", [])]
                    return any(model_name in m for m in loaded_models)
        except Exception:
            return False

    async def generate(
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
            if AIOHTTP_AVAILABLE:
                session = await self._get_session()
                custom_timeout = aiohttp.ClientTimeout(total=timeout)
                async with session.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=custom_timeout
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return OllamaResponse(
                            success=True,
                            content=data.get("response", ""),
                            model=data.get("model", model),
                            total_duration=data.get("total_duration"),
                            prompt_eval_count=data.get("prompt_eval_count"),
                            eval_count=data.get("eval_count"),
                        )
                    else:
                        error_text = await resp.text()
                        return OllamaResponse(
                            success=False,
                            content="",
                            model=model,
                            error=f"HTTP {resp.status}: {error_text}"
                        )
            else:
                # Synchronous fallback
                req = urllib.request.Request(
                    f"{self.base_url}/api/generate",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST"
                )
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    data = json.loads(resp.read().decode())
                    return OllamaResponse(
                        success=True,
                        content=data.get("response", ""),
                        model=data.get("model", model),
                        total_duration=data.get("total_duration"),
                        prompt_eval_count=data.get("prompt_eval_count"),
                        eval_count=data.get("eval_count"),
                    )
        except asyncio.TimeoutError:
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

    async def generate_chat(
        self,
        model: str,
        messages: list,
        temperature: float = 0.7,
        max_tokens: int = 500,
        seed: Optional[int] = None,
        keep_alive: str = "5m",
        timeout: int = 120
    ) -> OllamaResponse:
        """
        Generate text using Ollama's chat endpoint (OpenAI-compatible).

        Args:
            model: Model name
            messages: List of message dicts [{"role": "system/user/assistant", "content": "..."}]
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            seed: Random seed for reproducibility
            keep_alive: How long to keep model loaded
            timeout: Request timeout in seconds

        Returns:
            OllamaResponse with generation result
        """
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
            "keep_alive": keep_alive,
        }

        if seed is not None and seed >= 0:
            payload["options"]["seed"] = seed

        try:
            if AIOHTTP_AVAILABLE:
                session = await self._get_session()
                custom_timeout = aiohttp.ClientTimeout(total=timeout)
                async with session.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                    timeout=custom_timeout
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        message = data.get("message", {})
                        return OllamaResponse(
                            success=True,
                            content=message.get("content", ""),
                            model=data.get("model", model),
                            total_duration=data.get("total_duration"),
                            prompt_eval_count=data.get("prompt_eval_count"),
                            eval_count=data.get("eval_count"),
                        )
                    else:
                        error_text = await resp.text()
                        return OllamaResponse(
                            success=False,
                            content="",
                            model=model,
                            error=f"HTTP {resp.status}: {error_text}"
                        )
            else:
                req = urllib.request.Request(
                    f"{self.base_url}/api/chat",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST"
                )
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    data = json.loads(resp.read().decode())
                    message = data.get("message", {})
                    return OllamaResponse(
                        success=True,
                        content=message.get("content", ""),
                        model=data.get("model", model),
                        total_duration=data.get("total_duration"),
                        prompt_eval_count=data.get("prompt_eval_count"),
                        eval_count=data.get("eval_count"),
                    )
        except asyncio.TimeoutError:
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


def run_async(coro):
    """
    Run an async coroutine from sync context.

    Handles the case where we're already in an event loop (ComfyUI)
    vs when we need to create a new one.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None:
        # Already in async context - create a new task
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result()
    else:
        # No running loop - use asyncio.run
        return asyncio.run(coro)
