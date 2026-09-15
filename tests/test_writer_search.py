from __future__ import annotations

import unittest

from thai_writer_search import WRITER_HYBRID_ALPHA, WriterSearch, _rank_normalize


class FakeV25:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def search(self, query, **kwargs):
        self.calls.append((query, kwargs))
        return [dict(item) for item in self.results[: kwargs["top_k"]]]


class FakeLearned:
    category_mode = "omit"

    def __init__(self, scores):
        self.scores = scores
        self.rows_seen = None

    def score_rows(self, rows):
        self.rows_seen = rows
        return [
            {
                "expected_utility": float(score) + 1.0,
                "severe_probability": 0.1,
                "safe_score": float(score),
            }
            for score in self.scores[: len(rows)]
        ]


class FakeNeural:
    def __init__(self, scores):
        self.scores = scores
        self.rows_seen = None

    def score_rows(self, rows):
        self.rows_seen = rows
        return [float(score) for score in self.scores[: len(rows)]]

    def runtime_info(self):
        return {"loaded": False, "category_mode": "omit"}


def _result(word, score, *, query_definition="น้ำที่ตกจากเมฆ"):
    return {
        "word": word,
        "score": score,
        "relation_tier": 1,
        "relation_hint": "definition_similar",
        "lexical_form": "standalone",
        "sense_resolution": "selected_sense_supported",
        "lexical_score": score,
        "lexical_rank": 1,
        "dense_similarity": score,
        "dense_rank": 1,
        "query_sense": {
            "sense": 1,
            "definition": query_definition,
        },
        "matched_candidate_sense": {
            "sense": 1,
            "definition": f"ความหมาย {word}",
        },
        "definition": f"ความหมาย {word}",
    }


class WriterSearchTests(unittest.TestCase):
    def test_locked_alpha_cannot_change(self):
        with self.assertRaises(ValueError):
            WriterSearch(
                v25=FakeV25([]),
                learned=FakeLearned([]),
                neural=FakeNeural([]),
                alpha=0.4,
            )

    def test_rank_normalize_matches_phase3_direction(self):
        values = _rank_normalize([10.0, 20.0, 30.0])
        self.assertEqual(values.tolist(), [0.0, 0.5, 1.0])

    def test_rank_normalize_averages_ties(self):
        values = _rank_normalize([1.0, 1.0, 3.0])
        self.assertEqual(values.tolist(), [0.25, 0.25, 1.0])

    def test_search_reranks_only_v25_pool(self):
        v25_results = [
            _result("ก", 0.9),
            _result("ข", 0.8),
            _result("ค", 0.7),
            _result("ง", 0.6),
        ]
        searcher = WriterSearch(
            v25=FakeV25(v25_results),
            learned=FakeLearned([0.1, 0.2, 0.3]),
            neural=FakeNeural([0.1, 0.2, 0.3]),
        )

        results = searcher.search("ฝน", top_k=2, rerank_pool=3)

        self.assertEqual({item["word"] for item in results}, {"ข", "ค"})
        self.assertNotIn("ง", {item["word"] for item in results})
        self.assertEqual(searcher.v25.calls[0][1]["top_k"], 3)

    def test_hybrid_uses_equal_locked_weights(self):
        v25_results = [_result("ก", 0.9), _result("ข", 0.8), _result("ค", 0.7)]
        searcher = WriterSearch(
            v25=FakeV25(v25_results),
            learned=FakeLearned([3.0, 2.0, 1.0]),
            neural=FakeNeural([1.0, 2.0, 3.0]),
        )

        results = searcher.search("ฝน", top_k=3, rerank_pool=3)

        self.assertEqual(searcher.alpha, WRITER_HYBRID_ALPHA)
        self.assertTrue(all(item["writer_hybrid_score"] == 0.5 for item in results))

    def test_tie_break_uses_original_v25_rank(self):
        v25_results = [_result("ก", 0.9), _result("ข", 0.8), _result("ค", 0.7)]
        searcher = WriterSearch(
            v25=FakeV25(v25_results),
            learned=FakeLearned([3.0, 2.0, 1.0]),
            neural=FakeNeural([1.0, 2.0, 3.0]),
        )

        results = searcher.search("ฝน", top_k=3, rerank_pool=3)

        self.assertEqual([item["word"] for item in results], ["ก", "ข", "ค"])
        self.assertEqual([item["original_v25_rank"] for item in results], [1, 2, 3])

    def test_runtime_rows_have_query_definition_and_no_annotation(self):
        v25_results = [_result("พิรุณ", 0.9)]
        learned = FakeLearned([1.0])
        neural = FakeNeural([1.0])
        searcher = WriterSearch(v25=FakeV25(v25_results), learned=learned, neural=neural)

        results = searcher.search("ฝน", top_k=1, rerank_pool=1)

        self.assertEqual(
            learned.rows_seen[0]["query"]["definition"],
            "น้ำที่ตกจากเมฆ",
        )
        self.assertNotIn("annotation", learned.rows_seen[0])
        self.assertEqual(neural.rows_seen, learned.rows_seen)
        self.assertEqual(results[0]["reranker_status"], "writer_reranked")

    def test_output_keeps_diagnostics(self):
        searcher = WriterSearch(
            v25=FakeV25([_result("พิรุณ", 0.9)]),
            learned=FakeLearned([2.5]),
            neural=FakeNeural([0.7]),
        )

        result = searcher.search("ฝน", top_k=1, rerank_pool=1)[0]

        for key in (
            "final_rank",
            "original_v25_rank",
            "writer_hybrid_score",
            "writer_learned_rank_score",
            "writer_neural_rank_score",
            "writer_learned_safe_score",
            "writer_expected_utility",
            "writer_severe_probability",
            "writer_neural_score",
            "writer_alpha",
        ):
            self.assertIn(key, result)

    def test_rerank_pool_must_cover_top_k(self):
        searcher = WriterSearch(
            v25=FakeV25([]),
            learned=FakeLearned([]),
            neural=FakeNeural([]),
        )
        with self.assertRaises(ValueError):
            searcher.search("ฝน", top_k=10, rerank_pool=5)


if __name__ == "__main__":
    unittest.main()
