from __future__ import annotations

import json
import re
from typing import Any

from thai_listwise_v5 import build_listwise_document, build_listwise_query


DEFAULT_GEMMA4_E2B_QAT = "google/gemma-4-E2B-it-qat-mobile-transformers"\nGEMMA4_PROMPT_VERSION = "v5.3.1-hard-lexical-gate"\n
GEMMA4_SYSTEM_PROMPT = """You are a ranking engine for a Thai dictionary/search tool used by fiction writers.

Rank only by lexical usefulness for replacing the target Thai word.

Use a HARD lexical-validity gate before considering style or frequency:
1. A top candidate must preserve the intended dictionary sense.
2. A top candidate must preserve the target grammatical role. Infer the candidate role
   from its headword and dictionary definition when no explicit POS label is given.
3. A top candidate should be able to replace the target in a natural Thai sentence
   without changing who/what/action/property the sentence is talking about.
4. A word that is merely associated with the target MUST rank below a true substitute,
   even if it is very common or semantically close.
5. Strongly demote different parts of speech, cause/effect relations, objects/agents,
   compounds or phrases with a different lexical role, and manner/subtype changes that
   materially alter meaning.
6. Only after lexical validity is satisfied: among equally valid substitutes, prefer
   common contemporary Thai before formal, literary, archaic, technical, or rare
   dictionary vocabulary.
7. Keep useful literary or archaic alternatives lower in the list rather than removing
   them completely.

If fewer than 10 strong substitutes exist, fill the remaining lower positions with the
closest usable alternatives. Never let a related-but-not-substitutable word outrank a
valid substitute just because it is frequent, vivid, or strongly associated.

Follow the requested output format exactly.
"""


def build_gemma_ranking_prompt(
    query_text: str,
    candidates: list[dict[str, Any]],
    *,
    top_k: int = 10,
) -> str:
    requested = min(max(1, int(top_k)), len(candidates))
    lines = [
        "TARGET:",
        query_text.strip(),
        "",
        "CANDIDATES:",
    ]
    for index, candidate in enumerate(candidates):
        lines.append(f"[{index}] {build_listwise_document(candidate)}")

    lines.extend(
        [
            "",
            f"Return exactly the best {requested} candidate IDs from the list above.",
            "Return one JSON array only, ordered best to worst.",
            "Example format: [7, 2, 15, 1]",
            "Do not add prose, markdown, labels, scores, or explanations.",
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


def parse_gemma_ranking(
    output_text: str,
    candidate_count: int,
    *,
    expected_count: int,
) -> tuple[list[int], bool, str]:
    if candidate_count <= 0:
        return [], True, "empty"

    target = min(max(1, int(expected_count)), candidate_count)
    text = str(output_text or "").strip()

    best: list[int] = []
    strategy = "none"

    for match in re.finditer(r"\[[^\[\]]*\]", text):
        chunk = match.group(0)
        try:
            raw = json.loads(chunk)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(raw, list):
            continue
        ids = _unique_valid_ids(raw, candidate_count)
        if len(ids) > len(best):
            best = ids
            strategy = "json"

    if len(best) < target:
        bracket_ids = _unique_valid_ids(
            re.findall(r"\[(\d+)\]", text),
            candidate_count,
        )
        if len(bracket_ids) > len(best):
            best = bracket_ids
            strategy = "bracket-ids"

    selected = best[:target]
    return selected, len(selected) == target, strategy


class Gemma4ListwiseJudge:
    def __init__(
        self,
        model_id: str = DEFAULT_GEMMA4_E2B_QAT,
        *,
        device: str | None = None,
        max_new_tokens: int = 96,
        seed: int = 42,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForMultimodalLM, AutoProcessor
        except (ImportError, AttributeError) as exc:
            raise RuntimeError(
                "Gemma 4 requires a recent Transformers build with "
                "AutoModelForMultimodalLM. Upgrade Transformers and restart "
                "the runtime before running V5.3."
            ) from exc

        self._torch = torch
        self.model_id = model_id
        self.max_new_tokens = max(32, int(max_new_tokens))
        self.seed = int(seed)

        device_map: Any = {"": device} if device else "auto"

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForMultimodalLM.from_pretrained(
            model_id,
            dtype="auto",
            device_map=device_map,
        )
        self.model.eval()

        self.last_output = ""
        self.last_parse_complete = False
        self.last_parse_strategy = "none"
        self.last_explicit_count = 0
        self.last_requested_count = 0

    def rank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        *,
        category: str | None = None,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        if not candidates:
            self.last_output = "[]"
            self.last_parse_complete = True
            self.last_parse_strategy = "empty"
            self.last_explicit_count = 0
            self.last_requested_count = 0
            return []

        query_sense = candidates[0].get("query_sense")
        if not isinstance(query_sense, dict):
            query_sense = None

        query_text = build_listwise_query(
            query,
            query_sense,
            category=category,
        )
        requested = min(max(1, int(top_k)), len(candidates))
        prompt = build_gemma_ranking_prompt(
            query_text,
            candidates,
            top_k=requested,
        )

        messages = [
            {"role": "system", "content": GEMMA4_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            enable_thinking=False,
        ).to(self.model.device)
        input_len = int(inputs["input_ids"].shape[-1])

        self._torch.manual_seed(self.seed)
        if self._torch.cuda.is_available():
            self._torch.cuda.manual_seed_all(self.seed)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=1.0,
            top_p=0.95,
            top_k=64,
            use_cache=True,
        )

        generated = outputs[0][input_len:]
        raw_text = self.processor.decode(
            generated,
            skip_special_tokens=True,
        ).strip()

        ids, complete, strategy = parse_gemma_ranking(
            raw_text,
            len(candidates),
            expected_count=requested,
        )

        self.last_output = raw_text
        self.last_parse_complete = complete
        self.last_parse_strategy = strategy
        self.last_explicit_count = len(ids)
        self.last_requested_count = requested

        return [
            {
                "local_index": index,
                "rank": rank,
                "candidate": dict(candidates[index]),
            }
            for rank, index in enumerate(ids, start=1)
        ]
