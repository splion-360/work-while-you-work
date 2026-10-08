import unittest
from types import SimpleNamespace
from unittest.mock import patch

from deploy.huggingface_space_runtime import wait_until_running


class FakeApi:
    def __init__(self, stages):
        self.stages = iter(stages)

    def get_space_runtime(self, _repo_id):
        return SimpleNamespace(stage=next(self.stages))


class HuggingFaceSpaceRuntimeTests(unittest.TestCase):
    @patch("deploy.huggingface_space_runtime.time.sleep")
    def test_waits_until_space_is_running(self, sleep):
        wait_until_running(FakeApi(["BUILDING", "RUNNING"]), "owner/space")
        sleep.assert_called_once_with(5)

    def test_fails_immediately_for_terminal_error(self):
        with self.assertRaisesRegex(RuntimeError, "BUILD_ERROR"):
            wait_until_running(FakeApi(["BUILD_ERROR"]), "owner/space")

    @patch("deploy.huggingface_space_runtime.time.monotonic", side_effect=[0, 1])
    def test_times_out(self, _monotonic):
        with self.assertRaisesRegex(TimeoutError, "before the timeout"):
            wait_until_running(FakeApi([]), "owner/space", timeout=1)


if __name__ == "__main__":
    unittest.main()
