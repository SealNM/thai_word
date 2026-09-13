from __future__ import annotations

import unittest

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from thai_lexical_v1 import (
    SearchArtifacts,
    _hierarchical_score,
    _reference_strength,
    _relation_tier,
    search,
    split_tokens,
)


class SenseAwareSearchTests(unittest.TestCase):
    def setUp(self) -> None:
        entries = [
            {
                "word": "ขวบ",
                "definition": "ปี รอบปี",
                "definitions": ["ปี รอบปี"],
                "source_ids": [1],
                "sense_count": 1,
            },
            {
                "word": "ฝน",
                "definition": "น้ำตกจากเมฆ ลับมีด",
                "definitions": ["น้ำตกจากเมฆ", "ลับมีด"],
                "source_ids": [2, 3],
                "sense_count": 2,
            },
            {
                "word": "พิรุณ",
                "definition": "ฝน",
                "definitions": ["ฝน"],
                "source_ids": [4],
                "sense_count": 1,
            },
            {
                "word": "ลับ",
                "definition": "ทำให้คมด้วยการฝน",
                "definitions": ["ทำให้คมด้วยการฝน"],
                "source_ids": [5],
                "sense_count": 1,
            },
        ]
        senses = [
            {"entry_index": 0, "sense_index": 1, "definition": "ปี รอบปี"},
            {"entry_index": 1, "sense_index": 1, "definition": "น้ำตกจากเมฆ"},
            {"entry_index": 1, "sense_index": 2, "definition": "ลับมีด"},
            {"entry_index": 2, "sense_index": 1, "definition": "ฝน"},
            {"entry_index": 3, "sense_index": 1, "definition": "ทำให้คมด้วยการฝน"},
        ]
        tokenized = [
            "ปี รอบปี",
            "น้ำตก เมฆ",
            "ลับ มีด",
            "ฝน",
            "ทำ คม ฝน",
        ]
        vectorizer = TfidfVectorizer(
            tokenizer=split_tokens,
            preprocessor=None,
            token_pattern=None,
            lowercase=False,
            ngram_range=(1, 2),
            dtype=np.float32,
        )
        matrix = vectorizer.fit_transform(tokenized).tocsr()

        self.artifacts = SearchArtifacts(
            entries=entries,
            senses=senses,
            word_to_index={"ขวบ": 0, "ฝน": 1, "พิรุณ": 2, "ลับ": 3},
            vectorizer=vectorizer,
            matrix=sparse.csr_matrix(matrix),
            references=[[], [], [3], [1], [1]],
            reverse_references=[[], [3, 4], [], [2]],
            token_sets=[
                {"ปี", "รอบปี"},
                {"น้ำตก", "เมฆ"},
                {"ลับ", "มีด"},
                {"ฝน"},
                {"ทำ", "คม", "ฝน"},
            ],
            entry_to_senses=[[0], [1, 2], [3], [4]],
            tokenize=lambda text: text.split(),
        )

    def test_default_uses_first_sense(self) -> None:
        results = search(self.artifacts, "ฝน", top_k=3, candidate_pool=4)
        self.assertEqual(results[0]["word"], "พิรุณ")
        self.assertEqual(results[0]["query_sense"]["sense"], 1)

    def test_explicit_second_sense_changes_neighborhood(self) -> None:
        results = search(
            self.artifacts,
            "ฝน",
            top_k=3,
            candidate_pool=4,
            sense=2,
        )
        self.assertEqual(results[0]["word"], "ลับ")
        self.assertEqual(results[0]["query_sense"]["sense"], 2)

    def test_out_of_range_sense_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            search(self.artifacts, "ฝน", sense=3)

    def test_reference_strength_prefers_direct_gloss(self) -> None:
        direct = _reference_strength("ฝน.", "ฝน")
        alternative = _reference_strength("เมฆ, ฝน.", "ฝน")
        subtype = _reference_strength("ฝนเม็ดใหญ่ที่ตกลงมาแรง", "ฝน")
        contextual = _reference_strength("หอบไป เช่น เมฆอุ้มฝน.", "ฝน")
        associated = _reference_strength("เทวดาแห่งฝน.", "ฝน")

        self.assertGreater(direct, alternative)
        self.assertGreater(alternative, subtype)
        self.assertGreater(subtype, contextual)
        self.assertGreater(contextual, associated)

    def test_direct_gloss_uses_synonym_relation_hint(self) -> None:
        results = search(self.artifacts, "ฝน", top_k=3, candidate_pool=4)
        pirun = next(item for item in results if item["word"] == "พิรุณ")
        self.assertEqual(pirun["relation_hint"], "direct_gloss_or_synonym")
        self.assertEqual(pirun["relation_tier"], 5)
        self.assertGreaterEqual(pirun["signals"]["reverse_reference"], 0.999)

    def test_relation_tier_prevents_similarity_from_beating_direct_gloss(self) -> None:
        pure_tier = _relation_tier(
            reverse_strength=1.0,
            forward_strength=0.0,
            candidate_word="พิรุณ",
        )
        alternative_tier = _relation_tier(
            reverse_strength=0.95,
            forward_strength=0.0,
            candidate_word="พลาหก",
        )
        pure_score = _hierarchical_score(
            pure_tier,
            cosine=0.0,
            shared=0.0,
            word_form=0.0,
            exact_headword_query=True,
        )
        alternative_score = _hierarchical_score(
            alternative_tier,
            cosine=1.0,
            shared=1.0,
            word_form=1.0,
            exact_headword_query=True,
        )
        self.assertGreater(pure_score, alternative_score)

    def test_bound_form_is_demoted_one_relation_tier(self) -> None:
        standalone = _relation_tier(
            reverse_strength=1.0,
            forward_strength=0.0,
            candidate_word="พรรษ",
        )
        bound = _relation_tier(
            reverse_strength=1.0,
            forward_strength=0.0,
            candidate_word="พรรษ-",
        )
        self.assertEqual(standalone, 5)
        self.assertEqual(bound, 4)


if __name__ == "__main__":
    unittest.main()
