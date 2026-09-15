"""Student reasoning/answer prompt and Qwen3.5 generation-prefix configuration."""

from __future__ import annotations

from pathlib import Path


REASONING_SYSTEM_PROMPT = (
    "You are a visual question-answering assistant. "
    "First explain your reasoning based on the image and the question in ordinary text. "
    "Then put only the final answer inside a single <answer>...</answer> pair "
    "at the end of your response. Keep the reasoning outside the answer tags. "
    "Do not use think tags or write anything after </answer>."
)


def configure_deepeyes_response(config) -> None:
    """Resolve the opt-in prompt and template before creating any Ray workers.

    The custom dataset transforms only messages in memory. Existing parquet
    files, questions, row order, labels, and image payloads remain untouched.
    The original model template is shared unchanged by rollout and Teacher
    prompt construction, including its empty non-thinking generation prefill.
    """
    from omegaconf import OmegaConf

    response_format = config.data.get("response_format", "original")
    if response_format == "original":
        return
    if response_format != "reasoning_answer":
        raise ValueError(f"Unknown data.response_format: {response_format}")
    if config.data.apply_chat_template_kwargs.get("enable_thinking", False) is not False:
        raise ValueError("reasoning_answer requires enable_thinking=false")
    if config.data.custom_cls.get("path") is not None:
        raise ValueError("reasoning_answer supplies its own DeepEyes dataset class")

    model = config.actor_rollout_ref.model
    if model.get("custom_chat_template") is not None:
        raise ValueError("reasoning_answer uses the original model template unchanged")
    tokenizer_path = Path(model.get("tokenizer_path") or model.path).expanduser()
    native_template = (tokenizer_path / "chat_template.jinja").read_text(encoding="utf-8")
    OmegaConf.update(config, "actor_rollout_ref.model.custom_chat_template", native_template)
    OmegaConf.update(config, "data.custom_cls.path", str(Path(__file__).with_name("deepeyes_dataset.py")))
    OmegaConf.update(config, "data.custom_cls.name", "DeepEyesReasoningDataset")
