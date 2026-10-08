import unittest

from deploy.benchmark_huggingface_space import percentile, quota_capacity, summarize


class HuggingFaceBenchmarkTests(unittest.TestCase):
    def test_percentile_uses_linear_interpolation(self):
        self.assertEqual(percentile([1, 2, 3, 4, 5], 50), 3)
        self.assertAlmostEqual(percentile([1, 2, 3, 4, 5], 95), 4.8)

    def test_capacity_uses_measured_gpu_seconds(self):
        self.assertEqual(quota_capacity(300, 1.2), 250)

    def test_summary_keeps_wall_and_gpu_latency_separate(self):
        samples = [
            {"wall_seconds": 3.0, "gpu_seconds": 1.0},
            {"wall_seconds": 5.0, "gpu_seconds": 2.0},
        ]
        result = summarize(samples)
        self.assertEqual(result["wall_seconds_p50"], 4.0)
        self.assertEqual(result["gpu_seconds_p50"], 1.5)
        self.assertEqual(
            result["estimated_daily_scores"]["free_at_p50_gpu_seconds"], 200
        )
        self.assertEqual(
            result["estimated_daily_scores"]["pro_at_p95_gpu_seconds"], 1230
        )


if __name__ == "__main__":
    unittest.main()
