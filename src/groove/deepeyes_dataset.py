"""Apply the next-run Student prompt without rewriting historical datasets."""

from __future__ import annotations

from copy import deepcopy

from verl.utils.dataset.rl_dataset import RLHFDataset

from groove.deepeyes_prompt import REASONING_SYSTEM_PROMPT


class DeepEyesReasoningDataset(RLHFDataset):
    def _build_messages(self, example: dict, key: str | None = None):
        prompt_key = key or self.prompt_key
        messages = deepcopy(example[prompt_key])
        if not messages or messages[-1]["role"] != "user":
            raise ValueError("DeepEyes Student prompt must end with the original user question")
        if messages[0]["role"] == "system":
            messages[0]["content"] = REASONING_SYSTEM_PROMPT
        else:
            messages.insert(0, {"role": "system", "content": REASONING_SYSTEM_PROMPT})
        updated = dict(example)
        updated[prompt_key] = messages
        # RLHFDataset normalizes image dictionaries in place. Keep the source
        # row reusable without copying the (potentially large) image payloads.
        updated[self.image_key] = [
            dict(image) if isinstance(image, dict) else image
            for image in (example.get(self.image_key) or [])
        ]
        return super()._build_messages(updated, key=prompt_key)
