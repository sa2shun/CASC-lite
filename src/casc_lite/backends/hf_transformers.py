"""Transformers backend implementation."""
from __future__ import annotations

import logging
from typing import Any, Dict, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from casc_lite.core.entropy import EntropyStats, compute_entropy_metrics

from .base import BaseBackend, BackendError

logger = logging.getLogger(__name__)


class HFTransformersBackend(BaseBackend):
    """Backend powered by Hugging Face Transformers."""

    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        torch_dtype: str | torch.dtype | None = None,
        **model_kwargs: Any,
    ) -> None:
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        if torch_dtype is not None and isinstance(torch_dtype, str):
            torch_dtype = getattr(torch, torch_dtype)
        logger.info("Loading model %s on device %s", model_name, self.device)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            **model_kwargs,
        )
        self.model.to(self.device)
        self.model.eval()

    def _prepare_inputs(self, prompt: str) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        return {key: value.to(self.device) for key, value in encoded.items()}

    @torch.inference_mode()
    def prefix_entropy(
        self,
        prompt: str,
        K: int,
        temperature: float,
        top_p: float | None = None,
        top_k: int | None = None,
        **generate_kwargs: Any,
    ) -> EntropyStats:
        inputs = self._prepare_inputs(prompt)
        max_tokens = max(K, 1)
        generation = self.model.generate(
            **inputs,
            do_sample=False,
            temperature=1.0,
            max_new_tokens=max_tokens,
            min_new_tokens=1,
            num_return_sequences=1,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **generate_kwargs,
        )
        scores = generation.scores
        entropy_stats = compute_entropy_metrics(scores, K)
        logger.debug(
            "Computed entropy %.4f for prompt length=%d",
            entropy_stats.average_entropy,
            inputs["input_ids"].shape[-1],
        )
        return entropy_stats

    def generate_n(
        self,
        prompt: str,
        n: int,
        temperature: float,
        top_p: float | None,
        top_k: int | None,
        max_new_tokens: int,
        min_new_tokens: int,
        **generate_kwargs: Any,
    ) -> Sequence[Dict[str, Any]]:
        if n < 1:
            raise ValueError("n must be >= 1")
        inputs = self._prepare_inputs(prompt)
        generation_kwargs = dict(
            do_sample=True,
            temperature=max(temperature, 1e-4),
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            num_return_sequences=n,
            return_dict_in_generate=True,
            output_scores=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        if top_p is not None:
            generation_kwargs["top_p"] = top_p
        if top_k is not None:
            generation_kwargs["top_k"] = top_k
        generation_kwargs.update(generate_kwargs)

        with torch.no_grad():
            generation = self.model.generate(**inputs, **generation_kwargs)
        sequences = getattr(generation, "sequences", None)
        if sequences is None:
            raise BackendError("Generation returned no sequences")
        prompt_length = inputs["input_ids"].shape[-1]
        outputs: list[Dict[str, Any]] = []
        for idx, sequence in enumerate(sequences):
            sequence = sequence.to("cpu")
            generated_part = sequence[prompt_length:]
            text = self.tokenizer.decode(generated_part, skip_special_tokens=True)
            outputs.append(
                {
                    "text": text,
                    "token_ids": generated_part.tolist(),
                    "num_generated_tokens": len(generated_part),
                    "sample_index": idx,
                }
            )
        return outputs

    def tokenize(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def detokenize(self, token_ids: Sequence[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
