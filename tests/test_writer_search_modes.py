from __future__ import annotations

import unittest
from unittest.mock import patch

from scripts import search_writer
from thai_writer_search import WriterSearch


class FakeV25:
    def __init__(self, results):
        self.results = [dict(item) for item in results]

    def search(self, query, **kwargs):
        return [dict(item) for item in self.results[: kwargs["top_k"]]]


class GoodLearned:
    category_mode = "omit"

    def score_rows(self, rows):
        return [
            {
                "expected_utility": 2.0,
                "severe_probability": 0.1,
                "safe_score": float(index + 1),
            }
            for index, _ in enumerate(rows)
        ]


class GoodNeural:
    def score_rows(self, rows):
        return [float(index + 1) for index, _ in enumerate(rows)]

    def runtime_info(self):
        return {"loaded": True, "category_mode": "omit"}


class BrokenNeural:
    def score_rows(self, rows):
        raise RuntimeError("neural exploded")

    def runtime_info(self):
        return {"loaded": True, "category_mode": "omit"}


def _result(word):
    return {
        "word": word,
        "score": 0.01,
        "relation_tier": 1,
        "relation_hint": "definition_similar",
        "lexical_form": "standalone",
        "sense_resolution": "selected_sense_supported",
        "lexical_score": 0.1,
        "lexical_rank": 1,
        "dense_similarity": 0.2,
        "dense_rank": 1,
        "query_sense": {"sense": 1, "definition": "query definition"},
        "matched_candidate_sense": {"sense": 1, "definition": word},
        "definition": word,
    }


class WriterSearchModesTests(unittest.TestCase):
    def test_off_mode_returns_v25_results_unchanged(self):
        base = [_result("ก"), _result("ข")]
        searcher = WriterSearch(v25=FakeV25(base), mode="off")

        results = searcher.search("ฝน", top_k=2, rerank_pool=30)

        self.assertEqual(results, base)
        self.assertEqual(searcher.last_reranker_status, "off")
        self.assertIsNone(searcher.last_reranker_error)

    def test_optional_mode_falls_back_exactly_to_v25_on_runtime_error(self):
        base = [_result("ก"), _result("ข")]
        searcher = WriterSearch(
            v25=FakeV25(base),
            learned=GoodLearned(),
            neural=BrokenNeural(),
            mode="optional",
        )

        results = searcher.search("ฝน", top_k=2, rerank_pool=2)

        self.assertEqual(results, base)
        self.assertEqual(searcher.last_reranker_status, "fallback_v25")
        self.assertIn("neural exploded", searcher.last_reranker_error)

    def test_optional_mode_can_fallback_from_initialization_error(self):
        base = [_result("ก")]
        searcher = WriterSearch(
            v25=FakeV25(base),
            mode="optional",
            initialization_error="FileNotFoundError: missing learned artifact",
        )

        results = searcher.search("ฝน", top_k=1, rerank_pool=1)

        self.assertEqual(results, base)
        self.assertEqual(searcher.last_reranker_status, "fallback_v25")
        self.assertIn("missing learned artifact", searcher.last_reranker_error)

    def test_required_mode_propagates_runtime_error(self):
        searcher = WriterSearch(
            v25=FakeV25([_result("ก")]),
            learned=GoodLearned(),
            neural=BrokenNeural(),
            mode="required",
        )

        with self.assertRaisesRegex(RuntimeError, "neural exploded"):
            searcher.search("ฝน", top_k=1, rerank_pool=1)

    def test_required_mode_requires_both_rerankers(self):
        with self.assertRaises(ValueError):
            WriterSearch(v25=FakeV25([]), mode="required")

    def test_optional_success_reports_writer_reranked(self):
        searcher = WriterSearch(
            v25=FakeV25([_result("ก"), _result("ข")]),
            learned=GoodLearned(),
            neural=GoodNeural(),
            mode="optional",
        )

        results = searcher.search("ฝน", top_k=2, rerank_pool=2)

        self.assertEqual(searcher.last_reranker_status, "writer_reranked")
        self.assertTrue(all(item["reranker_status"] == "writer_reranked" for item in results))

    def test_off_mode_ignores_rerank_pool_size(self):
        base = [_result("ก"), _result("ข")]
        searcher = WriterSearch(v25=FakeV25(base), mode="off")

        results = searcher.search("ฝน", top_k=2, rerank_pool=1)

        self.assertEqual(results, base)
        self.assertEqual(searcher.last_reranker_status, "off")

    def test_cli_list_senses_bypasses_writer_search_construction(self):
        args = search_writer.build_parser().parse_args(
            [
                "ฝน",
                "--dense-index",
                "unused-dense",
                "--list-senses",
            ]
        )
        with (
            patch("scripts.search_writer.load_artifacts", return_value="LEXICAL"),
            patch(
                "scripts.search_writer.list_senses",
                return_value=[{"sense": 1, "definition": "น้ำที่ตกจากเมฆ"}],
            ),
            patch("scripts.search_writer.WriterSearch.from_paths") as from_paths,
        ):
            payload = search_writer.run(args)

        self.assertEqual(payload[0]["sense"], 1)
        from_paths.assert_not_called()

    def test_cli_default_mode_is_optional(self):
        args = search_writer.build_parser().parse_args(
            ["ฝน", "--dense-index", "unused-dense"]
        )
        self.assertEqual(args.reranker_mode, "optional")

    def test_invalid_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            WriterSearch(v25=FakeV25([]), mode="automatic")


if __name__ == "__main__":
    unittest.main()
