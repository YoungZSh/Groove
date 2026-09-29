"""Explicit Qwen3.5 vision freezing and a first-update integrity audit."""

import hashlib

import torch


def parameter_group(name):
    name = name.replace("_fsdp_wrapped_module.", "")
    if name.startswith("model.visual.merger."):
        return "merger"
    if name.startswith("model.visual."):
        return "vit"
    return "llm"


def freeze_qwen35_vision(module, *, train_merger=False):
    """Freeze the complete visual subtree, optionally reopening only its merger.

    This deliberately rejects unverified model layouts instead of silently
    accepting a freeze flag that does not affect any parameters.
    """
    if getattr(module.config, "model_type", None) not in {"qwen3_5", "qwen3_5_moe"}:
        raise ValueError("Vision freezing currently supports Qwen3.5 model layouts only")
    core = getattr(module, "model", None)
    visual = getattr(core, "visual", None)
    merger = getattr(visual, "merger", None)
    language = getattr(core, "language_model", None)
    if not all(isinstance(part, torch.nn.Module) for part in (visual, merger, language)):
        raise ValueError("Expected model.visual, model.visual.merger and model.language_model")
    visual.requires_grad_(False)
    if train_merger:
        merger.requires_grad_(True)
    counts = {key: {"total": 0, "trainable": 0} for key in ("vit", "merger", "llm")}
    for name, parameter in module.named_parameters():
        group = parameter_group(name)
        counts[group]["total"] += parameter.numel()
        if parameter.requires_grad:
            counts[group]["trainable"] += parameter.numel()
        expected = group == "llm" or (group == "merger" and train_merger)
        if parameter.requires_grad != expected:
            raise ValueError(f"Unexpected requires_grad for {name}")
    if not all(value["total"] > 0 for value in counts.values()):
        raise ValueError("Each of ViT, merger and LLM must contain parameters")
    return {"freeze_vision_tower": True, "train_vision_merger": train_merger, "parameters": counts}


def _frozen_digest(module):
    digest = hashlib.sha256()
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            digest.update(name.encode())
            digest.update(str(tuple(parameter.shape)).encode())
            raw = parameter.detach().reshape(-1).cpu().contiguous().view(torch.uint8)
            digest.update(raw.numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def capture_vision_update(module):
    """Inspect local FSDP original-parameter shards before one optimizer step."""
    samples = {}
    norms = {key: [] for key in ("vit", "merger", "llm")}
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad and parameter.grad is not None:
            raise RuntimeError(f"Frozen parameter received a gradient: {name}")
        if parameter.grad is not None and parameter.grad.numel():
            norms[parameter_group(name)].append(
                torch.linalg.vector_norm(parameter.grad.detach(), dtype=torch.float32).double().square()
            )
        if parameter.requires_grad and parameter.numel():
            samples[name] = parameter.detach().reshape(-1)[:64].cpu().clone()
    return {
        "frozen_sha256": _frozen_digest(module),
        "samples": samples,
        "gradient_norms": {key: torch.stack(values).sum().sqrt().item() if values else 0.0
                           for key, values in norms.items()},
    }


@torch.no_grad()
def verify_vision_update(module, before):
    """Require byte-identical frozen shards; report trainable sample changes."""
    after_digest = _frozen_digest(module)
    if before["frozen_sha256"] != after_digest:
        raise RuntimeError("Frozen visual parameters changed during the optimizer step")
    changed = {key: 0 for key in ("merger", "llm")}
    for name, parameter in module.named_parameters():
        if name in before["samples"] and not torch.equal(
            before["samples"][name], parameter.detach().reshape(-1)[:64].cpu()
        ):
            changed[parameter_group(name)] += 1
    return {"frozen_weights_unchanged": True, "frozen_gradients_absent": True,
            "frozen_shard_sha256": after_digest, "gradient_norms": before["gradient_norms"],
            "changed_parameter_samples": changed}
