from __future__ import annotations

import unittest

from scripts.profile_writer_runtime import _summary, build_parser


class WriterRuntimeProfileTests(unittest.TestCase):
    def test_summary_reports_basic_statistics(self):
        report = _summary([1.0, 2.0, 3.0])

        self.assertEqual(report["count"], 3)
        self.assertEqual(report["mean"], 2.0)
        self.assertEqual(report["median"], 2.0)
        self.assertEqual(report["min"], 1.0)
        self.assertEqual(report["max"], 3.0)

    def test_parser_uses_locked_candidate_pool_default(self):
        args = build_parser().parse_args(
            [
                "ฝน",
                "--dense-index",
                "artifacts/v2/test",
                "--neural-model",
                "artifacts/phase3/model",
            ]
        )

        self.assertEqual(args.rerank_pool, 30)
        self.assertEqual(args.top_k, 10)
        self.assertEqual(args.repeats, 5)

    def test_profiler_requires_explicit_neural_model(self):
        with self.assertRaises(SystemExit):
            build_parser().parse_args(
                [
                    "ฝน",
                    "--dense-index",
                    "artifacts/v2/test",
                ]
            )


if __name__ == "__main__":
    unittest.main()
