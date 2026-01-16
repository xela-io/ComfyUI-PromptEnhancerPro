"""
VRAM Manager for Prompt Enhancer Pro

Manages Ollama model loading and unloading to optimize VRAM usage.
Critical for workflows where LLM and image generation share GPU memory.
"""

import asyncio
import json
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    import urllib.request
    import urllib.error


@dataclass
class ModelInfo:
    """Information about a loaded model"""
    name: str
    size: int
    digest: str
    expires_at: Optional[str] = None
    size_vram: Optional[int] = None


class VRAMManager:
    """
    Manages Ollama model VRAM allocation.

    Provides methods to:
    - Check which models are loaded
    - Pre-load models into VRAM
    - Unload models to free VRAM
    - Monitor VRAM usage
    """

    def __init__(self, ollama_url: str = "http://localhost:11434"):
        self.ollama_url = ollama_url.rstrip("/")
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> "aiohttp.ClientSession":
        """Get or create aiohttp session"""
        if not AIOHTTP_AVAILABLE:
            raise RuntimeError("aiohttp not available")
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self):
        """Close the HTTP session"""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def get_loaded_models(self) -> Tuple[bool, List[ModelInfo]]:
        """
        Get list of currently loaded models.

        Uses /api/ps endpoint to check running models.

        Returns:
            Tuple of (success, list of ModelInfo)
        """
        try:
            if AIOHTTP_AVAILABLE:
                session = await self._get_session()
                async with session.get(f"{self.ollama_url}/api/ps") as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        models = []
                        for m in data.get("models", []):
                            models.append(ModelInfo(
                                name=m.get("name", ""),
                                size=m.get("size", 0),
                                digest=m.get("digest", ""),
                                expires_at=m.get("expires_at"),
                                size_vram=m.get("size_vram"),
                            ))
                        return True, models
                    return False, []
            else:
                req = urllib.request.Request(f"{self.ollama_url}/api/ps")
                with urllib.request.urlopen(req, timeout=10) as resp:
                    data = json.loads(resp.read().decode())
                    models = []
                    for m in data.get("models", []):
                        models.append(ModelInfo(
                            name=m.get("name", ""),
                            size=m.get("size", 0),
                            digest=m.get("digest", ""),
                            expires_at=m.get("expires_at"),
                            size_vram=m.get("size_vram"),
                        ))
                    return True, models
        except Exception as e:
            print(f"[VRAMManager] Error getting loaded models: {e}")
            return False, []

    async def is_model_loaded(self, model_name: str) -> bool:
        """
        Check if a specific model is currently loaded.

        Args:
            model_name: Name of the model to check

        Returns:
            True if model is loaded, False otherwise
        """
        success, models = await self.get_loaded_models()
        if not success:
            return False
        return any(model_name in m.name for m in models)

    async def load_model(
        self,
        model_name: str,
        keep_alive: str = "5m"
    ) -> Tuple[bool, str]:
        """
        Pre-load a model into VRAM.

        Sends a minimal generate request to trigger model loading.

        Args:
            model_name: Name of the model to load
            keep_alive: How long to keep model loaded

        Returns:
            Tuple of (success, message)
        """
        payload = {
            "model": model_name,
            "prompt": "",  # Empty prompt just to load the model
            "stream": False,
            "keep_alive": keep_alive,
        }

        try:
            if AIOHTTP_AVAILABLE:
                session = await self._get_session()
                timeout = aiohttp.ClientTimeout(total=300)  # Loading can take a while
                async with session.post(
                    f"{self.ollama_url}/api/generate",
                    json=payload,
                    timeout=timeout
                ) as resp:
                    if resp.status == 200:
                        return True, f"Model {model_name} loaded successfully"
                    else:
                        error_text = await resp.text()
                        return False, f"Failed to load model: {error_text}"
            else:
                req = urllib.request.Request(
                    f"{self.ollama_url}/api/generate",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST"
                )
                with urllib.request.urlopen(req, timeout=300) as resp:
                    if resp.status == 200:
                        return True, f"Model {model_name} loaded successfully"
                    return False, "Failed to load model"
        except Exception as e:
            return False, f"Error loading model: {str(e)}"

    async def unload_model(self, model_name: str) -> Tuple[bool, str]:
        """
        Unload a model from VRAM.

        Uses keep_alive=0 to immediately unload the model.

        Args:
            model_name: Name of the model to unload

        Returns:
            Tuple of (success, message)
        """
        payload = {
            "model": model_name,
            "prompt": "",
            "stream": False,
            "keep_alive": "0",  # Immediately unload
        }

        try:
            if AIOHTTP_AVAILABLE:
                session = await self._get_session()
                async with session.post(
                    f"{self.ollama_url}/api/generate",
                    json=payload
                ) as resp:
                    if resp.status == 200:
                        return True, f"Model {model_name} unloaded from VRAM"
                    else:
                        error_text = await resp.text()
                        return False, f"Failed to unload model: {error_text}"
            else:
                req = urllib.request.Request(
                    f"{self.ollama_url}/api/generate",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST"
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    if resp.status == 200:
                        return True, f"Model {model_name} unloaded from VRAM"
                    return False, "Failed to unload model"
        except Exception as e:
            return False, f"Error unloading model: {str(e)}"

    async def unload_all_models(self) -> Tuple[bool, List[str]]:
        """
        Unload all currently loaded models.

        Returns:
            Tuple of (success, list of unloaded model names)
        """
        success, models = await self.get_loaded_models()
        if not success:
            return False, []

        unloaded = []
        for model in models:
            ok, _ = await self.unload_model(model.name)
            if ok:
                unloaded.append(model.name)

        return True, unloaded

    async def get_vram_usage(self) -> Dict[str, any]:
        """
        Get current VRAM usage information.

        Returns:
            Dictionary with VRAM usage details
        """
        success, models = await self.get_loaded_models()
        if not success:
            return {"error": "Could not get model info"}

        total_vram = 0
        model_details = []

        for model in models:
            vram = model.size_vram or model.size
            total_vram += vram
            model_details.append({
                "name": model.name,
                "vram_bytes": vram,
                "vram_gb": round(vram / (1024**3), 2) if vram else 0,
            })

        return {
            "total_vram_bytes": total_vram,
            "total_vram_gb": round(total_vram / (1024**3), 2) if total_vram else 0,
            "models": model_details,
            "model_count": len(models),
        }


def run_async(coro):
    """
    Run an async coroutine from sync context.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result()
    else:
        return asyncio.run(coro)
