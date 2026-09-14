from __future__ import annotations

import json
import re
from typing import Any

from thai_listwise_v5 import DEFAULT_JINA_LISTWISE


DEFAULT_QWEN35_LISTWISE = "Qwen/Qwen3.5-0.8B"

QWEN35_RANKING_POLICY = """You are a Thai lexical ranking model for fiction writers.

Compare ALL candidate dictionary entries against one another and rank them from best
to worst for replacing the target word.

Priority order:
1. Preserve the intended dictionary sense.
2. Preserve the target grammatical role.
3. Prefer a natural lexical substitute over a merely related word.
4. Among equally valid substitutes, prefer common contemporary Thai before formal,
   literary, archaic, technical, or rare dictionary vocabulary.
5. Keep useful literary/archaic alternatives in the ranking, but below equally
   accurate common alternatives.
6. Penalize associated-only words, cause/effect relations, objects/agents, compounds
   with a different lexical role, and manner/subtype changes that alter the meaning.

Return exactly one JSON array of the requested best candidate integer IDs, ordered
from best to worst. Return no prose, no markdown, and no explanation.
"""


def build_generative_listwise_prompt(
    query_text: str,
    documents: list[str],
    *,
    output_count: int = 15,
) -> str:
    lines = [
        QWEN35_RANKING_POLICY.strip(),
        "",
        "TARGET AND SENSE:",
        query_text.strip(),
        "",
        "CANDIDATES:",
    ]
    for index, document in enumerate(documents):
        lines.append(f"[{index}] {document.strip()}")

    requested = min(max(1, int(output_count)), len(documents))
    lines.extend(
        [
            "",
            f"There are exactly {len(documents)} candidates with IDs 0 through "
            f"{max(0, len(documents) - 1)}.",
            f"Return exactly the best {requested} candidate IDs only.",
            "Output only the JSON array now.",
        ]
    )
    return "\n".join(lines)


def _unique_valid_ids(values: list[Any], candidate_count: int) -> list[int]:
    parsed: list[int] = []
    for value in values:
        if isinstance(value, bool):
            continue
        try:
            index = int(value)
        except (TypeError, ValueError):
            continue
        if 0 <= index < candidate_count and index not in parsed:
            parsed.append(index)
    return parsed


def parse_ranked_candidate_ids(
    output_text: str,
    candidate_count: int,
    *,
    expected_count: int | None = None,
) -> tuple[list[int], bool, str, int]:
    if candidate_count <= 0:
        return [], True, "empty", 0

    target_count = min(
        candidate_count,
        max(1, int(expected_count if expected_count is not None else candidate_count)),
    )
    text = str(output_text or "").strip()

    best: list[int] = []
    strategy = "none"

    # Prefer the largest valid JSON array in the generated output instead of
    # blindly taking the first bracket pair.
    for match in re.finditer(r"\[[^\[\]]*\]", text):
        chunk = match.group(0)
        try:
            raw = json.loads(chunk)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(raw, list):
            continue
        candidate_ids = _unique_valid_ids(raw, candidate_count)
        if len(candidate_ids) > len(best):
            best = candidate_ids
            strategy = "json"

    # Small models sometimes emit singleton labels like [7] [3] [12].
    if len(best) < target_count:
        singleton_ids = _unique_valid_ids(
            re.findall(r"\[(\d+)\]", text),
            candidate_count,
        )
        if len(singleton_ids) > len(best):
            best = singleton_ids
            strategy = "bracket-ids"

    # Last recovery path: collect candidate integers in appearance order from
    # generated output only. This is intentionally marked as recovery.
    if len(best) < target_count:
        numeric_ids = _unique_valid_ids(
            re.findall(r"\d+", text),
            candidate_count,
        )
        if len(numeric_ids) > len(best):
            best = numeric_ids
            strategy = "numeric-recovery"

    explicit_count = min(len(best), target_count)
    selected = best[:target_count]
    complete = explicit_count >= target_count

    # V5's shared ranker interface expects a full permutation. IDs not
    # explicitly ranked by Qwen fall back to the original V2.5 order.
    missing = [index for index in range(candidate_count) if index not in selected]
    full_order = selected + missing

    return full_order, complete, strategy, explicit_count


class Qwen35GenerativeListwiseRanker:
    def __init__(
        self,
        model_id: str = DEFAULT_QWEN35_LISTWISE,
        *,
        device: str | None = None,
        dtype: str | None = None,
        max_new_tokens: int = 192,
        output_count: int = 15,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForMultimodalLM, AutoProcessor
        except (ImportError, AttributeError) as exc:
            raise RuntimeError(
                "Qwen3.5 requires a recent Hugging Face Transformers build with "
                "AutoModelForMultimodalLM. Upgrade Transformers before running V5.1."
            ) from exc

        self.model_id = model_id
        self.max_new_tokens = max(64, int(max_new_tokens))
        self.output_count = max(1, int(output_count))
        self.last_output = ""
        self.last_parse_complete = False
        self.last_parse_strategy = "none"
        self.last_parsed_count = 0
        self.last_requested_count = 0

        if dtype is None:
            resolved_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        else:
            try:
                resolved_dtype = getattr(torch, dtype)
            except AttributeError as exc:
                raise ValueError(f"Unknown torch dtype: {dtype!r}") from exc

        if device:
            device_map: Any = {"": device}
        else:
            device_map = "auto"

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForMultimodalLM.from_pretrained(
            model_id,
            dtype=resolved_dtype,
            device_map=device_map,
        )
        self.model.eval()

    def rank(self, query_text: str, documents: list[str]) -> list[dict[str, Any]]:
        if not documents:
            self.last_output = "[]"
            self.last_parse_complete = True
            self.last_parse_strategy = "empty"
            self.last_parsed_count = 0
            self.last_requested_count = 0
            return []

        requested_count = min(self.output_count, len(documents))
        prompt = build_generative_listwise_prompt(
            query_text,
            documents,
            output_count=requested_count,
        )
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)
        input_length = int(inputs["input_ids"].shape[-1])

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            use_cache=True,
        )

        generated = outputs[0][input_length:]
        output_text = self.processor.decode(
            generated,
            skip_special_tokens=True,
        ).strip()

        order, complete, strategy, explicit_count = parse_ranked_candidate_ids(
            output_text,
            len(documents),
            expected_count=requested_count,
        )

        self.last_output = output_text
        self.last_parse_complete = complete
        self.last_parse_strategy = strategy
        self.last_parsed_count = explicit_count
        self.last_requested_count = requested_count

        return [
            {
                "index": index,
                "rank": rank,
                "score": 1.0 / float(rank),
            }
            for rank, index in enumerate(order, start=1)
        ]


def resolve_v51_ranker(
    name: str,
    *,
    device: str | None = None,
    qwen_dtype: str | None = None,
    qwen_max_new_tokens: int = 192,
    qwen_output_count: int = 15,
):
    key = str(name).strip().lower()

    if key in {"jina", "jina-v3.5", DEFAULT_JINA_LISTWISE.lower()}:
        from thai_listwise_v5 import JinaListwiseRanker

        return (
            "jina-v3.5",
            JinaListwiseRanker(
                DEFAULT_JINA_LISTWISE,
                device=device,
            ),
        )

    if key in {
        "qwen",
        "qwen3.5",
        "qwen3.5-0.8b",
        DEFAULT_QWEN35_LISTWISE.lower(),
    }:
        return (
            "qwen3.5-0.8b",
            Qwen35GenerativeListwiseRanker(
                DEFAULT_QWEN35_LISTWISE,
                device=device,
                dtype=qwen_dtype,
                max_new_tokens=qwen_max_new_tokens,
                output_count=qwen_output_count,
            ),
        )

    raise ValueError(
        f"Unknown V5.1 ranker {name!r}. "
        "Expected jina-v3.5 or qwen3.5-0.8b."
    )
