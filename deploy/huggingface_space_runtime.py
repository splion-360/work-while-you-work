import time

FAILURE_STAGES = {"BUILD_ERROR", "RUNTIME_ERROR", "CONFIG_ERROR"}


def wait_until_running(api, repo_id, timeout=600, poll_interval=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        stage = api.get_space_runtime(repo_id).stage
        if stage == "RUNNING":
            return
        if stage in FAILURE_STAGES:
            raise RuntimeError(f"Space failed while starting: {stage}")
        time.sleep(poll_interval)
    raise TimeoutError("Space did not become ready before the timeout")
