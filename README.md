# Prompt Enhancer Pro

A professional ComfyUI custom node for LLM-powered prompt enhancement using Ollama. Features intelligent VRAM management, dynamic model detection, and seamless Docker integration.

![ComfyUI](https://img.shields.io/badge/ComfyUI-Custom%20Node-blue)
![Python](https://img.shields.io/badge/Python-3.10%2B-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Features

- **Dynamic Model Detection**: Automatically detects available Ollama models via dropdown
- **Multiple Enhancement Modes**: Character Builder, Freeform, Scene Builder, Portrait, Custom Template
- **Smart VRAM Management**: Automatic ComfyUI VRAM clearing before Ollama + model unloading after
- **Context File Support**: Load context from external files for consistent style/format preferences
- **Reproducible Results**: Seed support for deterministic prompt generation
- **Negative Prompt Generation**: Advanced mode generates matching negative prompts
- **Docker Ready**: Auto-detection of Ollama endpoints (localhost, Docker gateway, host.docker.internal)

## Nodes

### 1. Prompt Enhancer Pro

Main node for LLM-powered prompt enhancement.

![Prompt Enhancer Pro Node](https://via.placeholder.com/400x300?text=Prompt+Enhancer+Pro)

#### Inputs

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| `base_prompt` | STRING | - | Your original prompt to enhance |
| `enhancement_mode` | DROPDOWN | character_builder | Enhancement template to use |
| `model_name` | DROPDOWN | (auto-detected) | Ollama model - dynamically populated |
| `custom_template` | STRING | - | Custom system prompt (for custom_template mode) |
| `context_info` | STRING | - | Additional context (format, style preferences) |
| `seed` | INT | -1 | Random seed (-1 = random, ≥0 = fixed) |
| `temperature` | FLOAT | 0.7 | Sampling temperature (0.0-2.0) |
| `max_tokens` | INT | 500 | Maximum tokens to generate (50-2000) |
| `unload_after` | BOOLEAN | True | Unload model after generation to free VRAM |
| `free_vram_before` | BOOLEAN | True | Free ComfyUI VRAM before Ollama request (prevents VRAM conflicts) |
| `ollama_url` | STRING | host.docker.internal:11434 | Ollama API endpoint |
| `timeout` | INT | 120 | Request timeout in seconds |

#### Outputs

| Output | Type | Description |
|--------|------|-------------|
| `enhanced_prompt` | STRING | The LLM-enhanced prompt |
| `original_prompt` | STRING | Original input (for comparison) |
| `model_used` | STRING | Actual model that was used |
| `generation_seed` | INT | Seed used (useful for reproduction) |

---

### 2. Prompt Enhancer Pro (Advanced)

Extended version with negative prompt generation and language controls.

#### Additional Inputs

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| `negative_prompt_mode` | BOOLEAN | False | Also generate a negative prompt |
| `language` | DROPDOWN | auto | Output language (auto/english/german) |
| `verbosity` | DROPDOWN | normal | Output detail level (concise/normal/detailed) |

#### Additional Outputs

| Output | Type | Description |
|--------|------|-------------|
| `negative_prompt` | STRING | Generated negative prompt |

---

### 3. Ollama Connection Checker

Utility node to verify Ollama connectivity and list available models.

#### Outputs

| Output | Type | Description |
|--------|------|-------------|
| `is_connected` | BOOLEAN | Connection status |
| `status_message` | STRING | Status or error message |
| `available_models` | STRING | Comma-separated list of installed models |

---

### 4. Ollama Model Manager

VRAM management utility for manual model control.

#### Inputs

| Input | Type | Description |
|-------|------|-------------|
| `action` | DROPDOWN | Operation to perform |
| `model_name` | DROPDOWN | Model to manage (auto-detected) |
| `keep_alive` | STRING | Duration to keep model loaded (e.g., "5m", "1h", "0") |

#### Actions

| Action | Description |
|--------|-------------|
| `check_loaded` | List models currently in VRAM |
| `load_model` | Pre-load a model into VRAM |
| `unload_model` | Remove specific model from VRAM |
| `unload_all` | Clear all models from VRAM |
| `refresh_models` | Refresh the model dropdown list |

---

### 5. Context File Loader

Load context from external files to use with Prompt Enhancer Pro.

#### Inputs

| Input | Type | Description |
|-------|------|-------------|
| `file_name` | DROPDOWN | Context file from `context/` folder (auto-detected) |
| `additional_context` | STRING | Extra context to append |

#### Outputs

| Output | Type | Description |
|--------|------|-------------|
| `context` | STRING | Combined context (file content + additional) |

#### Usage

1. Place `.txt` or `.md` files in the `context/` folder
2. Add **Context File Loader** node
3. Select a file from the dropdown
4. Connect `context` output to `context_info` input of Prompt Enhancer Pro

```
[Context File Loader] ──context──► [Prompt Enhancer Pro (context_info)]
```

---

## Enhancement Modes

| Mode | Best For | Description |
|------|----------|-------------|
| `character_builder` | Characters/Portraits | Structured output with subject, appearance, pose, environment, technical details |
| `freeform` | General prompts | Flexible enhancement with quality tags |
| `scene_builder` | Environments | Focus on setting, atmosphere, lighting |
| `portrait` | Portrait photography | Professional portrait descriptions with lighting setup |
| `custom_template` | Custom needs | Use your own system prompt |

### Character Builder Output Structure

```
1. Subject: Main subject with precise details
2. Appearance: Physical features, clothing, accessories
3. Pose & Expression: Body posture, facial expression, gaze
4. Environment: Setting, lighting, atmosphere
5. Technical: Camera angle, style, quality tags
```

---

## Installation

### Prerequisites

1. **Ollama** installed and running on host:
   ```bash
   # Arch Linux
   sudo pacman -S ollama

   # Start Ollama
   systemctl --user start ollama
   # Or: ollama serve
   ```

2. **Install at least one model**:
   ```bash
   ollama pull qwen2.5:7b-instruct
   # Or other models:
   ollama pull llama3.2:8b
   ollama pull mistral:7b
   ```

### Node Installation

#### Option A: Manual Installation

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/xela-io/ComfyUI-PromptEnhancerPro.git prompt_enhancer_pro
pip install aiohttp
```

#### Option B: ComfyUI Manager

Search for "Prompt Enhancer Pro" in ComfyUI Manager and install.

### Docker Configuration

If running ComfyUI in Docker, add these to your `docker-compose.yml`:

```yaml
services:
  comfyui:
    extra_hosts:
      - "host.docker.internal:host-gateway"
    environment:
      - OLLAMA_HOST=http://host.docker.internal:11434
```

**Firewall (UFW)**: Allow Docker networks to access Ollama:
```bash
sudo ufw allow from 172.16.0.0/12 to any port 11434
```

---

## Usage

### Basic Workflow

```
[Your Prompt] → [Prompt Enhancer Pro] → [CLIP Text Encode] → [KSampler]
```

1. Add **Prompt Enhancer Pro** node
2. Enter your base prompt (e.g., "a woman on the beach")
3. Select enhancement mode and model
4. Connect `enhanced_prompt` output to CLIP Text Encode
5. Enable `unload_after` to free VRAM before image generation

### Example

**Input:**
```
eine Frau am Strand
```

**Output (character_builder mode):**
```
1girl, woman, standing on sandy beach, ocean waves in background,
golden hour sunset lighting, warm orange and pink sky,
long flowing brown hair, white sundress, barefoot,
looking at viewer, serene peaceful expression, slight smile,
wide shot, shallow depth of field, cinematic composition,
masterpiece, best quality, highly detailed, photorealistic,
natural lighting, atmospheric, professional photography
```

### Reproducible Results

Use a fixed seed (≥0) to get the same enhanced prompt every time:

```
base_prompt: "a cat sleeping"
seed: 42
→ Always produces the same enhanced prompt
```

---

## VRAM Management

### Automatic (Recommended)

Enable `unload_after` (default: True) to automatically free VRAM after prompt generation. The LLM is unloaded via `keep_alive: 0` before image generation starts.

### Manual Control

Use **Ollama Model Manager** for fine-grained control:

1. `check_loaded` - See what's using VRAM
2. `unload_all` - Clear all LLM models before large image batches
3. `load_model` - Pre-load models for faster first inference

### Typical Workflow for Limited VRAM

```
[Prompt Enhancer Pro (unload_after=True)]
    → [CLIP Text Encode]
    → [Load Checkpoint]
    → [KSampler]
```

---

## Recommended Models

| Model | VRAM | Speed | Quality | Best For |
|-------|------|-------|---------|----------|
| `qwen2.5:7b-instruct` | ~5GB | Fast | Good | General use |
| `qwen2.5:14b-instruct` | ~9GB | Medium | Better | Higher quality |
| `llama3.2:8b` | ~5GB | Fast | Good | Creative prompts |
| `mistral:7b` | ~4GB | Fast | Good | Efficient |
| `gemma2:9b` | ~6GB | Medium | Good | Balanced |

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_HOST` | http://localhost:11434 | Ollama API endpoint |

### Custom Templates

Add `.txt` files to the `templates/` directory:

```
prompt_enhancer_pro/
└── templates/
    ├── character_builder.txt  (built-in)
    └── my_custom_template.txt (your template)
```

Template files are automatically available in the `custom_template` mode.

---

## Troubleshooting

### "Ollama not reachable"

1. **Check if Ollama is running:**
   ```bash
   curl http://localhost:11434/api/tags
   ```

2. **Start Ollama:**
   ```bash
   ollama serve
   ```

3. **Docker users:** Ensure `extra_hosts` and firewall rules are configured (see Installation)

### "Model not found"

1. **List installed models:**
   ```bash
   ollama list
   ```

2. **Install a model:**
   ```bash
   ollama pull qwen2.5:7b-instruct
   ```

3. **Refresh dropdown:** Use Ollama Model Manager → `refresh_models`

### Slow Generation

- Use smaller models (7B instead of 14B)
- Reduce `max_tokens`
- Check GPU usage: `nvidia-smi`

### Empty or Poor Results

- Increase `temperature` (try 0.8-1.0 for more creativity)
- Increase `max_tokens` (try 800-1000)
- Try a different enhancement mode
- Add context in `context_info` field

### Out of VRAM

1. Enable `unload_after` (should be True by default)
2. Use **Ollama Model Manager** → `unload_all` before image generation
3. Use smaller LLM models

---

## API Reference

### Ollama Endpoints Used

| Endpoint | Purpose |
|----------|---------|
| `GET /api/tags` | List available models |
| `GET /api/ps` | List loaded models (VRAM) |
| `POST /api/generate` | Generate text with `keep_alive` control |

### Error Handling

The node gracefully handles errors with fallback to original prompt:
- Connection failures → Returns original prompt + warning
- Timeouts → Returns original prompt + warning
- Empty responses → Returns original prompt + warning
- Model not found → Clear error message with installation hint

---

## Project Structure

```
prompt_enhancer_pro/
├── __init__.py              # Package registration
├── README.md                # This documentation
├── requirements.txt         # Python dependencies
├── nodes/
│   ├── __init__.py
│   └── prompt_enhancer.py   # Main node implementations
├── utils/
│   ├── __init__.py
│   ├── ollama_client.py     # Sync Ollama API client (urllib)
│   ├── vram_manager.py      # VRAM load/unload management
│   └── templates.py         # Built-in enhancement templates
├── templates/
│   └── character_builder.txt # External template file
└── context/
    └── example_photorealistic.txt  # Example context file
```

---

## Changelog

### v1.1.0 (2026-01-16)
- Added **Context File Loader** node for loading context from external files
- Added `free_vram_before` option to free ComfyUI VRAM before Ollama requests
- Added `context/` folder with example context file
- Prevents VRAM conflicts between ComfyUI models and Ollama

### v1.0.0 (2026-01-16)
- Initial release
- 4 nodes: Enhancer, Advanced Enhancer, Connection Checker, Model Manager
- 5 built-in enhancement templates
- Dynamic Ollama model detection via dropdown
- Full VRAM management with automatic unload
- Docker support with host.docker.internal
- Synchronous HTTP using urllib (stable with ComfyUI event loop)
- Seed support for reproducible results

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Author

**xela-io**
- GitHub: [https://github.com/xela-io](https://github.com/xela-io)

---

## Related Projects

- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) - The base UI
- [Ollama](https://ollama.com/) - Local LLM runtime
- [ComfyUI-CharacterBuilder](https://github.com/xela-io/ComfyUI-CharacterBuilder) - Character description nodes
- [ComfyUI-StyleSelector](https://github.com/xela-io/ComfyUI-StyleSelector) - Style selection nodes
