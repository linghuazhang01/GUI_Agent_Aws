import contextlib
import datetime
import json
import logging
import os
import sys
import time
from typing import *
from wrapt_timeout_decorator import *

logger = logging.getLogger("desktopenv.experiment")


class _TeeStream:
    def __init__(self, log_file):
        self._log_file = log_file
        self._log_closed = False

    def write(self, data):
        if data:
            if not self._log_closed and not self._log_file.closed:
                try:
                    self._log_file.write(data)
                except ValueError:
                    self._log_closed = True

    def flush(self):
        if not self._log_closed and not self._log_file.closed:
            try:
                self._log_file.flush()
            except ValueError:
                self._log_closed = True

    def close_log(self):
        self._log_closed = True


@contextlib.contextmanager
def _capture_runtime_output(example_result_dir: str):
    runtime_log_path = os.path.join(example_result_dir, "runtime.log")
    root_logger = logging.getLogger()
    handler = logging.FileHandler(runtime_log_path, encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    root_logger.addHandler(handler)

    # Reduce noisy SDK debug logs that include base64 payloads.
    anthropic_logger = logging.getLogger("anthropic")
    httpx_logger = logging.getLogger("httpx")
    httpcore_logger = logging.getLogger("httpcore")
    prev_levels = (
        anthropic_logger.level,
        httpx_logger.level,
        httpcore_logger.level,
    )
    anthropic_logger.setLevel(logging.INFO)
    httpx_logger.setLevel(logging.INFO)
    httpcore_logger.setLevel(logging.INFO)

    with open(runtime_log_path, "a", encoding="utf-8") as log_file:
        stdout_tee = _TeeStream(log_file)
        stderr_tee = _TeeStream(log_file)
        with contextlib.redirect_stdout(stdout_tee), contextlib.redirect_stderr(
            stderr_tee
        ):
            try:
                yield
            finally:
                stdout_tee.close_log()
                stderr_tee.close_log()
                anthropic_logger.setLevel(prev_levels[0])
                httpx_logger.setLevel(prev_levels[1])
                httpcore_logger.setLevel(prev_levels[2])
                root_logger.removeHandler(handler)
                handler.close()


def run_single_example(
    agent, env, example, max_steps, instruction, args, example_result_dir, scores
):
    with _capture_runtime_output(example_result_dir):
        runtime_logger = setup_logger(example, example_result_dir)
        try:
            agent.reset(runtime_logger)
        except Exception:
            agent.reset()

        env.reset(task_config=example)
        time.sleep(60)  # Wait for the environment to be ready
        obs = env._get_obs()  # Get the initial observation

        with open(os.path.join(example_result_dir, "step_0.png"), "wb") as _f:
            _f.write(obs["screenshot"])

        with open(
            os.path.join(example_result_dir, "instruction.txt"), "w", encoding="utf-8"
        ) as f:
            f.write(instruction)

        done = False
        step_idx = 0
        # env.controller.start_recording()
        while not done and step_idx < max_steps:
            response, actions = agent.predict(instruction, obs)
            for action in actions:
                action_timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")
                logger.info("Step %d: %s", step_idx + 1, action)
                obs, reward, done, info = env.step(action, args.sleep_after_execution)

                logger.info("Reward: %.2f", reward)
                logger.info("Done: %s", done)
                # Save screenshot and trajectory information
                with open(
                    os.path.join(
                        example_result_dir, f"step_{step_idx + 1}_{action_timestamp}.png"
                    ),
                    "wb",
                ) as _f:
                    _f.write(obs["screenshot"])

                response.update(
                    {
                        "step_num": step_idx + 1,
                        "action_timestamp": action_timestamp,
                        "action": action,
                        "reward": reward,
                        "done": done,
                        "info": info,
                        "screenshot_file": f"step_{step_idx + 1}_{action_timestamp}.png",
                    }
                )
                with open(
                    os.path.join(example_result_dir, "traj.jsonl"),
                    "a",
                    encoding="utf-8",
                ) as f:
                    f.write(json.dumps(response, ensure_ascii=False))
                    f.write("\n")
                if done:
                    logger.info("The episode is done.")
                    break
            step_idx += 1
        result = env.evaluate()
        logger.info("Result: %.2f", result)
        scores.append(result)
        with open(
            os.path.join(example_result_dir, "result.txt"), "w", encoding="utf-8"
        ) as f:
            f.write(f"{result}\n")
    # env.controller.end_recording(os.path.join(example_result_dir, "recording.mp4"))


def setup_logger(example, example_result_dir):
    runtime_logger = logging.getLogger(f"desktopenv.example.{example['id']}")
    runtime_logger.setLevel(logging.DEBUG)
    return runtime_logger
