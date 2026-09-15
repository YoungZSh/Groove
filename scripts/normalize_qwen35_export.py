#!/usr/bin/env python3
"""Normalize repeated Qwen3.5 export prefixes using a reference schema.

Writes a new directory and verifies every tensor is unchanged. This is a
compatibility conversion for the Transformers 5.5 save_pretrained output;
FSDP training checkpoints are not modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil


def canonical_key(key: str) -> str:
    if re.match(r"^model\.(?:language_model\.)+visual\.", key):
        return re.sub(r"^model\.(?:language_model\.)+visual\.", "model.visual.", key)
    return re.sub(r"^model\.(?:language_model\.)+", "model.language_model.", key)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    expected = {}
    for file in args.reference.glob("*.safetensors"):
        with safe_open(file, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                # MTP is not part of the trained Student and is not used here.
                if not key.startswith("mtp."):
                    expected[key] = handle.get_slice(key).get_shape()
    assert expected, "Missing reference schema"
    tensors, mapping = {}, {}
    source_file = args.source / "model.safetensors"
    with safe_open(source_file, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            new_key = canonical_key(key)
            assert new_key not in tensors, f"Collision: {key} -> {new_key}"
            assert new_key in expected, f"Unexpected key: {key} -> {new_key}"
            assert handle.get_slice(key).get_shape() == expected[new_key], new_key
            tensors[new_key] = handle.get_tensor(key)
            mapping[key] = new_key
    assert tensors.keys() == expected.keys(), expected.keys() - tensors.keys()
    args.output.mkdir(parents=True, exist_ok=False)
    save_file(tensors, str(args.output / "model.safetensors"), metadata={"format": "pt"})
    for name in ("config.json", "generation_config.json", "processor_config.json",
                 "tokenizer_config.json", "tokenizer.json", "chat_template.jinja"):
        source = args.source / name
        if source.exists():
            shutil.copy2(source, args.output / name)
    with safe_open(args.output / "model.safetensors", framework="pt", device="cpu") as handle:
        for key, original in tensors.items():
            assert torch.equal(original, handle.get_tensor(key)), key
    manifest = {
        "source": str(args.source.resolve()), "reference_schema": str(args.reference.resolve()),
        "tensor_count": len(tensors), "tensor_values_unchanged": True,
        "schema_and_shapes_match_reference_excluding_mtp": True, "key_mapping": mapping,
        "source_sha256": hashlib.sha256(source_file.read_bytes()).hexdigest(),
    }
    (args.output / "export_conversion.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "tensor_count": len(tensors), "all_tensors_equal": True}), flush=True)


if __name__ == "__main__":
    main()
