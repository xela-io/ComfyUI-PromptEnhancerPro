"""
VRAM Manager for Prompt Enhancer Pro

Manages Ollama model loading and unloading to optimize VRAM usage.
Synchronous implementation for ComfyUI compatibility.
"""

import json
import urllib.request
import urllib.error
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass


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
    Synchronous implementation using urllib.
    """

    def __init__(self, ollama_url: str = "http://localhost:11434"):
        self.ollama_url = ollama_url.rstrip("/")

    def get_loaded_models(self, timeout: int = 10) -> Tuple[bool, List[ModelInfo]]:
        """
        Get list of currently loaded models.
        """
        try:
            req = urllib.request.Request(
                f"{self.ollama_url}/api/ps",
                method="GET"
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
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
        except Exception as e:
            print(f"[VRAMManager] Error getting loaded models: {e}")
            return False, []

    def is_model_loaded(self, model_name: str) -> bool:
        """Check if a specific model is currently loaded."""
        success, models = self.get_loaded_models()
        if not success:
            return False
        return any(model_name in m.name for m in models)

    def load_model(self, model_name: str, keep_alive: str = "5m", timeout: int = 300) -> Tuple[bool, str]:
        """
        Pre-load a model into VRAM.
        """
        payload = {
            "model": model_name,
            "prompt": "",
            "stream": False,
            "keep_alive": keep_alive,
        }

        try:
            req = urllib.request.Request(
                f"{self.ollama_url}/api/generate",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status == 200:
                    return True, f"Model {model_name} loaded successfully"
                return False, f"Failed to load model: HTTP {resp.status}"
        except Exception as e:
            return False, f"Error loading model: {str(e)}"

    def unload_model(self, model_name: str, timeout: int = 30) -> Tuple[bool, str]:
        """
        Unload a model from VRAM using keep_alive=0.
        """
        payload = {
            "model": model_name,
            "prompt": "",
            "stream": False,
            "keep_alive": "0",
        }

        try:
            req = urllib.request.Request(
                f"{self.ollama_url}/api/generate",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status == 200:
                    return True, f"Model {model_name} unloaded from VRAM"
                return False, f"Failed to unload model: HTTP {resp.status}"
        except Exception as e:
            return False, f"Error unloading model: {str(e)}"

    def unload_all_models(self) -> Tuple[bool, List[str]]:
        """Unload all currently loaded models."""
        success, models = self.get_loaded_models()
        if not success:
            return False, []

        unloaded = []
        for model in models:
            ok, _ = self.unload_model(model.name)
            if ok:
                unloaded.append(model.name)

        return True, unloaded

    def get_vram_usage(self) -> Dict[str, Any]:
        """Get current VRAM usage information."""
        success, models = self.get_loaded_models()
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

    def close(self):
        """No-op for compatibility."""
        pass
