"""Model loading, chat formatting, residual capture, and steering hooks."""

from __future__ import annotations

import random
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class LoadedModel:
    model_id: str
    revision: str
    tokenizer: Any
    model: Any
    device: torch.device
    dtype: torch.dtype


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_dtype(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[name]


def load_model(
    model_id: str,
    revision: str,
    *,
    dtype: str = "float32",
    device: str = "cpu",
) -> LoadedModel:
    """Load an instruct model with hidden-state access."""
    torch_dtype = resolve_dtype(dtype)
    torch_device = torch.device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
    model.to(torch_device)
    model.eval()
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return LoadedModel(
        model_id=model_id,
        revision=revision,
        tokenizer=tokenizer,
        model=model,
        device=torch_device,
        dtype=torch_dtype,
    )


def supports_system_role(tokenizer: Any) -> bool:
    """Heuristic: try applying a system message; fall back if template rejects it."""
    messages = [
        {"role": "system", "content": "You are a test."},
        {"role": "user", "content": "Hi"},
    ]
    try:
        tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return True
    except Exception:
        return False


def format_chat(
    tokenizer: Any,
    *,
    user: str,
    system: str | None = None,
    assistant_prefill: str | None = None,
) -> str:
    """Build a chat-formatted prompt ending where the assistant reply begins.

    If the template has no system role, prepend the persona to the user turn.
    Optional assistant_prefill is appended after the generation prompt.
    """
    if system and supports_system_role(tokenizer):
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
    else:
        user_content = f"{system}\n\n{user}" if system else user
        messages = [{"role": "user", "content": user_content}]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    if assistant_prefill:
        text = text + assistant_prefill
    return text


def get_decoder_layers(model: Any) -> torch.nn.ModuleList:
    """Return the transformer block list across common HF architectures."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    if hasattr(model, "model") and hasattr(model.model, "decoder"):
        return model.model.decoder.layers
    raise AttributeError("Could not locate decoder layers on model")


def n_transformer_layers(model: Any) -> int:
    return len(get_decoder_layers(model))


class ResidualAddHook:
    """Add `coefficient * vector` to the residual stream at every sequence position.

    `layer_index` is 0-based over transformer blocks (not including embeddings).
    """

    def __init__(
        self,
        model: Any,
        *,
        layer_index: int,
        vector: torch.Tensor,
        coefficient: float,
    ) -> None:
        layers = get_decoder_layers(model)
        if layer_index < 0 or layer_index >= len(layers):
            raise IndexError(
                f"layer_index {layer_index} out of range for {len(layers)} layers"
            )
        self.layer_index = layer_index
        self.coefficient = float(coefficient)
        self.vector = vector.detach()
        self._handle: Any | None = None
        self._layer = layers[layer_index]

    def _hook(self, _module: Any, _inputs: Any, output: Any) -> Any:
        delta = self.coefficient * self.vector
        if isinstance(output, tuple):
            hidden = output[0]
            hidden = hidden + delta.to(device=hidden.device, dtype=hidden.dtype)
            return (hidden,) + output[1:]
        return output + delta.to(device=output.device, dtype=output.dtype)

    def attach(self) -> None:
        if self._handle is not None:
            return
        self._handle = self._layer.register_forward_hook(self._hook)

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


@contextmanager
def steering_context(
    model: Any,
    *,
    layer_index: int,
    vector: torch.Tensor,
    coefficient: float,
) -> Iterator[ResidualAddHook]:
    hook = ResidualAddHook(
        model,
        layer_index=layer_index,
        vector=vector,
        coefficient=coefficient,
    )
    hook.attach()
    try:
        yield hook
    finally:
        hook.remove()


@torch.inference_mode()
def forward_with_hidden_states(
    loaded: LoadedModel,
    prompt: str,
    *,
    steer_layer: int | None = None,
    steer_vector: torch.Tensor | None = None,
    steer_coefficient: float = 0.0,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    """Return next-token logits and all hidden states for a single prompt.

    Hidden states are a tuple of (n_layers + 1) tensors of shape [1, seq, hidden].
    The last prompt token is at index -1.
    """
    inputs = loaded.tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(loaded.device) for k, v in inputs.items()}

    def _run() -> Any:
        return loaded.model(**inputs, output_hidden_states=True, use_cache=False)

    if steer_layer is not None and steer_vector is not None:
        with steering_context(
            loaded.model,
            layer_index=steer_layer,
            vector=steer_vector,
            coefficient=steer_coefficient,
        ):
            outputs = _run()
    else:
        outputs = _run()
    return outputs.logits, outputs.hidden_states


def next_token_logits(logits: torch.Tensor) -> torch.Tensor:
    """Logits for the token predicted after the prompt (last position)."""
    return logits[0, -1, :]


def last_token_residual(
    hidden_states: tuple[torch.Tensor, ...],
    layer: int,
) -> torch.Tensor:
    """Residual at the last prompt token for a hidden-state index.

    Layer indexing matches `outputs.hidden_states`: 0 is embeddings,
    1..N are post-block residuals. Callers usually want 1..N.
    """
    return hidden_states[layer][0, -1, :].detach()
