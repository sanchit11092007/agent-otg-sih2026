"""
image_gen.py - Local AI Image Generation for Agent OTG

Uses the diffusers library with the "stabilityai/sd-turbo" model.
This model is lightweight and fast enough to run without a high-end GPU,
though generation will be noticeably slower on CPU-only machines.

Note on offline operation:
The model weights (approx 2-3 GB) download once on first use and require
internet access for that initial download only. After that, the pipeline
runs completely offline from the local Hugging Face cache.
"""
from __future__ import annotations

import os
import importlib.util
from contextlib import nullcontext
from pathlib import Path
from typing import Any

# Cached pipeline instance at module level (loaded once)
_PIPELINE: Any = None


def _clean_path(path: str) -> str:
    """Clean quotes, whitespace, and resolve relative path into output directory."""
    if not isinstance(path, (str, os.PathLike)) or not path:
        return ""
    p = str(path).strip().strip("<>").strip("&").strip()
    if not p or "\x00" in p:
        return ""
    if (p.startswith('"') and p.endswith('"')) or (p.startswith("'") and p.endswith("'")):
        p = p[1:-1].strip()

    # If it is a bare filename or relative path, save to ~/Downloads/AgentOTG or generated_files
    otg_dir = Path(os.environ.get("USERPROFILE", Path.home())) / "Downloads" / "AgentOTG"
    otg_dir.mkdir(parents=True, exist_ok=True)

    # Image generation is an output-only operation, so discard caller-supplied
    # directories rather than allowing writes outside the artifact folder.
    return str(otg_dir / Path(p).name)


def _get_pipeline():
    """Load and cache the SD-Turbo pipeline on GPU if available, else CPU."""
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE

    missing = [name for name in ("torch", "diffusers", "transformers", "accelerate")
               if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError(
            "Local image generation is unavailable because this Python environment is missing: "
            f"{', '.join(missing)}. Install the image dependencies from requirements.txt "
            "in the same environment used to run ask.py, then restart the app."
        )

    try:
        import torch
        from diffusers import AutoPipelineForText2Image

        # Select torch device and dtype
        if torch.cuda.is_available():
            device = "cuda"
            torch_dtype = torch.float16
        else:
            # Fallback to CPU (generation will be slower on CPU-only machines)
            device = "cpu"
            torch_dtype = torch.float32

        from config import IMAGE_GEN_LOCAL_ONLY, IMAGE_GEN_MODEL

        load_args = {"torch_dtype": torch_dtype, "local_files_only": IMAGE_GEN_LOCAL_ONLY}
        # fp16 variants are useful on CUDA, but not every compatible model
        # publishes one.  Retrying without it makes model selection robust.
        if device == "cuda":
            load_args["variant"] = "fp16"
        # Agent OTG normally blocks outbound traffic.  A first-time model
        # download is the sole intentional exception and only happens when
        # IMAGE_GEN_LOCAL_ONLY=false; later runs use the Hugging Face cache.
        try:
            from offline_guard import allow_external
            download_context = nullcontext() if IMAGE_GEN_LOCAL_ONLY else allow_external()
        except ImportError:
            download_context = nullcontext()
        with download_context:
            try:
                pipe = AutoPipelineForText2Image.from_pretrained(IMAGE_GEN_MODEL, **load_args)
            except Exception:
                load_args.pop("variant", None)
                pipe = AutoPipelineForText2Image.from_pretrained(IMAGE_GEN_MODEL, **load_args)
        pipe.to(device)
        if device == "cpu":
            pipe.enable_attention_slicing()
        _PIPELINE = pipe
        return _PIPELINE
    except Exception as exc:
        raise RuntimeError(
            f"Failed to initialize image generation pipeline: {exc}. "
            "Ensure diffusers, torch, transformers, and accelerate are installed."
        ) from exc


def generate_image(prompt: str, filename: str = "generated.png") -> str:
    """
    Generate an image from a text prompt using local SD-Turbo.
    Saves the image file and returns the saved file path.
    """
    if not prompt or not str(prompt).strip():
        return "❌ Error: Missing image description prompt."

    target_path = _clean_path(filename or "generated.png")
    if not target_path:
        return "❌ Error: Invalid output filename."
    if not target_path.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
        target_path += ".png"

    os.makedirs(os.path.dirname(target_path), exist_ok=True)

    try:
        pipe = _get_pipeline()
        from config import IMAGE_GEN_MODEL, IMAGE_GEN_SIZE
        # SD-Turbo operates in one step; use sensible values for other local
        # diffusion checkpoints without requiring the user to edit code.
        is_turbo = "turbo" in IMAGE_GEN_MODEL.lower()
        result = pipe(prompt=str(prompt).strip(),
                      num_inference_steps=1 if is_turbo else 20,
                      guidance_scale=0.0 if is_turbo else 7.0,
                      width=IMAGE_GEN_SIZE, height=IMAGE_GEN_SIZE)
        image = result.images[0]
        image.save(target_path)
        return f"✅ Image saved to: {target_path}"
    except Exception as exc:
        return (f"❌ Failed to generate image: {exc}. "
                "Check IMAGE_GEN_MODEL, then retry; its weights download once when "
                "IMAGE_GEN_LOCAL_ONLY=false and are cached for later offline use.")
