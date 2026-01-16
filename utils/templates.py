"""
Built-in Templates for Prompt Enhancer Pro

Contains system prompts and templates for different enhancement modes.
"""

import os
from typing import Dict, Optional, List
from pathlib import Path

# Directory containing external template files
TEMPLATES_DIR = Path(__file__).parent.parent / "templates"


# ============================================================================
# Built-in System Prompts
# ============================================================================

CHARACTER_BUILDER_TEMPLATE = """Du bist ein Experte für Stable Diffusion / FLUX Prompting.
Deine Aufgabe ist es, Benutzerprompts in detaillierte, bildgenerierende Prompts zu erweitern.

Strukturiere den Output wie folgt:
1. **Subject**: Hauptmotiv mit präzisen Details
2. **Appearance**: Physische Merkmale, Kleidung, Accessoires
3. **Pose & Expression**: Körperhaltung, Gesichtsausdruck, Blickrichtung
4. **Environment**: Setting, Beleuchtung, Atmosphäre
5. **Technical**: Kamerawinkel, Bildstil, Qualitätstags

Regeln:
- Verwende Komma-separierte Tags
- Priorisiere wichtige Elemente am Anfang
- Füge Qualitätstags hinzu: masterpiece, best quality, highly detailed
- Vermeide Negationen im Prompt (nutze separate negative prompts)
- Halte den Output unter 200 Wörtern
- Gib NUR den verbesserten Prompt aus, keine Erklärungen oder Strukturüberschriften"""


FREEFORM_TEMPLATE = """You are an expert at crafting prompts for AI image generation (Stable Diffusion, FLUX, Midjourney).

Your task is to enhance the user's prompt by:
1. Adding more descriptive details
2. Including relevant style and quality tags
3. Specifying lighting, composition, and mood where appropriate
4. Keeping the core intent of the original prompt

Rules:
- Output ONLY the enhanced prompt, no explanations
- Use comma-separated tags
- Prioritize important elements at the beginning
- Add quality tags: masterpiece, best quality, highly detailed
- Keep output under 150 words
- Maintain the language of the input (German input = German output)"""


SCENE_BUILDER_TEMPLATE = """You are a specialist for creating detailed scene descriptions for AI image generation.

Focus on:
1. **Setting**: Location, time of day, weather conditions
2. **Atmosphere**: Mood, lighting quality, color palette
3. **Details**: Background elements, props, environmental storytelling
4. **Technical**: Camera angle, lens type, depth of field

Rules:
- Create immersive, cinematic scene descriptions
- Use comma-separated tags
- Add atmosphere and lighting details
- Include quality tags: masterpiece, best quality, highly detailed
- Output ONLY the enhanced prompt
- Keep under 150 words"""


PORTRAIT_TEMPLATE = """You are an expert at creating portrait prompts for AI image generation.

Focus on:
1. **Subject**: Person description, age, ethnicity (if specified)
2. **Face**: Expression, gaze direction, makeup if applicable
3. **Hair**: Style, color, length, texture
4. **Clothing**: Outfit details, accessories
5. **Lighting**: Portrait lighting setup (Rembrandt, butterfly, etc.)
6. **Background**: Simple or contextual backdrop

Rules:
- Create flattering, professional portrait descriptions
- Specify lighting setup when appropriate
- Add quality tags: masterpiece, best quality, highly detailed, sharp focus
- Output ONLY the enhanced prompt
- Keep under 150 words"""


NEGATIVE_PROMPT_HELPER = """Based on the user's prompt, generate an appropriate negative prompt that will help avoid common issues.

Consider:
- Anatomy issues (extra limbs, deformed hands, etc.)
- Quality issues (blurry, low quality, artifacts)
- Style conflicts (if photorealistic: avoid anime, cartoon; if anime: avoid realistic)
- Composition issues (bad cropping, watermarks, text)

Output ONLY the negative prompt as comma-separated tags.
Keep it focused and relevant to the input."""


# ============================================================================
# Template Registry
# ============================================================================

BUILT_IN_TEMPLATES: Dict[str, str] = {
    "character_builder": CHARACTER_BUILDER_TEMPLATE,
    "freeform": FREEFORM_TEMPLATE,
    "scene_builder": SCENE_BUILDER_TEMPLATE,
    "portrait": PORTRAIT_TEMPLATE,
    "negative_prompt_helper": NEGATIVE_PROMPT_HELPER,
}


# ============================================================================
# Template Management Functions
# ============================================================================

def get_template(mode: str, custom_template: Optional[str] = None) -> str:
    """
    Get the appropriate system prompt template.

    Args:
        mode: Enhancement mode (character_builder, custom_template, freeform, etc.)
        custom_template: Custom template string (used when mode is "custom_template")

    Returns:
        System prompt string
    """
    if mode == "custom_template" and custom_template:
        return custom_template.strip()

    return BUILT_IN_TEMPLATES.get(mode, FREEFORM_TEMPLATE)


def list_built_in_templates() -> List[str]:
    """
    Get list of available built-in template names.

    Returns:
        List of template names
    """
    return list(BUILT_IN_TEMPLATES.keys())


def load_template_from_file(filename: str) -> Optional[str]:
    """
    Load a template from the templates directory.

    Args:
        filename: Name of the template file (with or without .txt extension)

    Returns:
        Template content or None if not found
    """
    if not filename.endswith(".txt"):
        filename = f"{filename}.txt"

    filepath = TEMPLATES_DIR / filename

    if filepath.exists():
        try:
            return filepath.read_text(encoding="utf-8").strip()
        except Exception as e:
            print(f"[PromptEnhancerPro] Error loading template {filename}: {e}")
            return None

    return None


def list_external_templates() -> List[str]:
    """
    List all template files in the templates directory.

    Returns:
        List of template filenames (without .txt extension)
    """
    if not TEMPLATES_DIR.exists():
        return []

    templates = []
    for file in TEMPLATES_DIR.glob("*.txt"):
        templates.append(file.stem)

    return sorted(templates)


def get_all_available_templates() -> List[str]:
    """
    Get list of all available templates (built-in + external).

    Returns:
        Combined list of template names
    """
    built_in = list_built_in_templates()
    external = list_external_templates()

    # Mark external templates with [file] suffix
    external_marked = [f"{t} [file]" for t in external if t not in built_in]

    return built_in + external_marked


def build_system_prompt(
    mode: str,
    custom_template: Optional[str] = None,
    context_info: Optional[str] = None
) -> str:
    """
    Build the complete system prompt with optional context.

    Args:
        mode: Enhancement mode
        custom_template: Custom template for custom_template mode
        context_info: Additional context to append

    Returns:
        Complete system prompt
    """
    base_template = get_template(mode, custom_template)

    if context_info and context_info.strip():
        return f"{base_template}\n\nAdditional Context:\n{context_info.strip()}"

    return base_template


# ============================================================================
# Model Presets
# ============================================================================

MODEL_PRESETS: Dict[str, Dict[str, any]] = {
    "qwen2.5:7b-instruct": {
        "name": "Qwen 2.5 7B Instruct",
        "description": "Fast, good quality prompts",
        "recommended_temp": 0.7,
        "max_tokens": 500,
    },
    "qwen2.5:14b-instruct": {
        "name": "Qwen 2.5 14B Instruct",
        "description": "Higher quality, slower",
        "recommended_temp": 0.7,
        "max_tokens": 500,
    },
    "llama3.2:8b": {
        "name": "Llama 3.2 8B",
        "description": "Meta's latest, balanced",
        "recommended_temp": 0.7,
        "max_tokens": 500,
    },
    "mistral:7b": {
        "name": "Mistral 7B",
        "description": "Fast and efficient",
        "recommended_temp": 0.7,
        "max_tokens": 500,
    },
    "gemma2:9b": {
        "name": "Gemma 2 9B",
        "description": "Google's model, creative",
        "recommended_temp": 0.8,
        "max_tokens": 500,
    },
}


def get_model_preset(model_name: str) -> Optional[Dict[str, any]]:
    """
    Get preset configuration for a model.

    Args:
        model_name: Model name to look up

    Returns:
        Preset dict or None
    """
    # Check exact match first
    if model_name in MODEL_PRESETS:
        return MODEL_PRESETS[model_name]

    # Check partial match (model name without tag)
    base_name = model_name.split(":")[0]
    for preset_name, preset in MODEL_PRESETS.items():
        if preset_name.startswith(base_name):
            return preset

    return None


def list_model_presets() -> List[str]:
    """
    Get list of available model presets.

    Returns:
        List of model names
    """
    return list(MODEL_PRESETS.keys())
