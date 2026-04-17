"""Run 10 OSWorld tasks with Qwen3.5-397B-A17B via DashScope.

Usage:
    python osworld_setup/run_qwen_test.py \
        --dashscope_api_key YOUR_KEY \
        --path_to_vm /path/to/vmware.vmx \
        [--test_all_meta_path evaluation_examples/test_all.json] \
        [--max_steps 15]

This script reuses the existing run_muscle_mem_agent_local.py infrastructure
but preconfigures all model settings for Qwen3.5 via DashScope.
"""

import argparse
import datetime
import json
import logging
import os
import sys

from tqdm import tqdm

import osworld_setup.lib_run_single as lib_run_single
from desktop_env.desktop_env import DesktopEnv
from muscle_mem.agents.agent import AgentMm
from muscle_mem.agents.grounding import OSWorldACI

from dotenv import load_dotenv

load_dotenv()

DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_MODEL = "qwen3.5-397b-a17b"

# 10 representative tasks spanning different domains
DEFAULT_10_TASKS = {
    "chrome": [
        "bb5e4c0d-f964-439c-97b6-bdb9747de3f4",
        "7b6c7e24-c58a-49fc-a5bb-d57b80e5b4c3",
    ],
    "libreoffice_calc": [
        "357ef137-7eeb-4c80-a3bb-0951f26a8aff",
        "42e0a640-4f19-4b28-973d-729602b5a4a7",
    ],
    "libreoffice_writer": [
        "0810415c-bde4-4443-9047-d5f70165a697",
        "0a0faba3-5580-44df-965d-f562a99b291c",
    ],
    "vlc": [
        "59f21cfb-0120-4326-b255-a5b827b38967",
        "8ba5ae7a-5ae5-4eab-9fcc-5dd4fe3abf89",
    ],
    "os": [
        "94d95f96-9699-4208-98ba-3c3119edf9c2",
        "bedcedc4-4d72-425e-ad62-21960b11fe0d",
    ],
}

# Logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
datetime_str = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")
os.makedirs("logs", exist_ok=True)
for level, handler_cls, path in [
    (logging.INFO, logging.FileHandler, f"logs/normal-{datetime_str}.log"),
    (logging.DEBUG, logging.FileHandler, f"logs/debug-{datetime_str}.log"),
    (logging.INFO, logging.StreamHandler, None),
]:
    h = handler_cls(path, encoding="utf-8") if path else handler_cls(sys.stdout)
    h.setLevel(level)
    h.setFormatter(logging.Formatter(
        fmt="\x1b[1;33m[%(asctime)s \x1b[31m%(levelname)s \x1b[32m%(module)s/%(lineno)d\x1b[1;33m] \x1b[0m%(message)s"
    ))
    if path and "sdebug" not in path:
        pass  # all loggers
    logger.addHandler(h)

logger = logging.getLogger("desktopenv.experiment")


def config():
    parser = argparse.ArgumentParser(description="Run 10 OSWorld tasks with Qwen3.5 via DashScope")

    # DashScope / model config
    parser.add_argument("--dashscope_api_key", type=str, default=os.getenv("DASHSCOPE_API_KEY", ""))
    parser.add_argument("--model", type=str, default=QWEN_MODEL)
    parser.add_argument("--ground_model", type=str, default=QWEN_MODEL)
    parser.add_argument("--base_url", type=str, default=DASHSCOPE_BASE_URL)

    # Environment config
    parser.add_argument("--path_to_vm", type=str, default=None)
    parser.add_argument("--provider_name", type=str, default="vmware")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--action_space", type=str, default="pyautogui")
    parser.add_argument("--observation_type", type=str, default="screenshot")
    parser.add_argument("--screen_width", type=int, default=1920)
    parser.add_argument("--screen_height", type=int, default=1080)
    parser.add_argument("--sleep_after_execution", type=float, default=3.0)
    parser.add_argument("--max_steps", type=int, default=15)

    # Agent config
    parser.add_argument("--max_trajectory_length", type=int, default=3)
    parser.add_argument("--test_config_base_dir", type=str, default="evaluation_examples")
    parser.add_argument("--grounding_width", type=int, default=1920)
    parser.add_argument("--grounding_height", type=int, default=1080)

    # Output
    parser.add_argument("--result_dir", type=str, default="./results_qwen_test")

    # Task selection
    parser.add_argument("--tasks_json", type=str, default=None,
                        help="Custom JSON file with tasks (same format as test_all.json). "
                             "If not provided, uses built-in DEFAULT_10_TASKS.")

    args = parser.parse_args()

    if not args.dashscope_api_key:
        raise ValueError("Please provide --dashscope_api_key or set DASHSCOPE_API_KEY env var")
    if not args.path_to_vm:
        raise ValueError("Please provide --path_to_vm")

    return args


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = config()

    # Load task list
    if args.tasks_json:
        with open(args.tasks_json, "r", encoding="utf-8") as f:
            test_all_meta = json.load(f)
    else:
        test_all_meta = DEFAULT_10_TASKS

    print(f"Tasks to run: {json.dumps(test_all_meta, indent=2)}")
    total = sum(len(ids) for ids in test_all_meta.values())
    print(f"Total: {total} tasks")

    # Configure engines — all use OpenAI-compatible (DashScope)
    engine_params = {
        "engine_type": "openai",
        "model": args.model,
        "base_url": args.base_url,
        "api_key": args.dashscope_api_key,
        "temperature": 0.0,
    }

    engine_params_for_grounding = {
        "engine_type": "openai",
        "model": args.ground_model,
        "base_url": args.base_url,
        "api_key": args.dashscope_api_key,
        "grounding_width": args.grounding_width,
        "grounding_height": args.grounding_height,
    }

    # Initialize environment
    env = DesktopEnv(
        provider_name=args.provider_name,
        path_to_vm=args.path_to_vm,
        action_space=args.action_space,
        screen_size=(args.screen_width, args.screen_height),
        headless=args.headless,
        os_type="Ubuntu",
        require_a11y_tree=args.observation_type in ["a11y_tree", "screenshot_a11y_tree", "som"],
        enable_proxy=True,
    )

    grounding_agent = OSWorldACI(
        env=env,
        platform="linux",
        engine_params_for_generation=engine_params,
        engine_params_for_grounding=engine_params_for_grounding,
        width=args.screen_width,
        height=args.screen_height,
    )

    agent = AgentMm(
        engine_params,
        grounding_agent,
        platform="linux",
        max_trajectory_length=args.max_trajectory_length,
    )

    scores = []
    for domain in tqdm(test_all_meta, desc="Domain"):
        for example_id in tqdm(test_all_meta[domain], desc="Example", leave=False):
            config_file = os.path.join(
                args.test_config_base_dir, f"examples/{domain}/{example_id}.json"
            )
            if not os.path.exists(config_file):
                logger.warning(f"Config not found: {config_file}, skipping")
                continue

            with open(config_file, "r", encoding="utf-8") as f:
                example = json.load(f)

            instruction = example["instruction"]
            logger.info(f"[Domain]: {domain}  [ID]: {example_id}")
            logger.info(f"[Instruction]: {instruction}")

            example_result_dir = os.path.join(
                args.result_dir, args.model, domain, example_id
            )
            os.makedirs(example_result_dir, exist_ok=True)

            try:
                lib_run_single.run_single_example(
                    agent, env, example, args.max_steps, instruction, args,
                    example_result_dir, scores,
                )
            except Exception as e:
                logger.error(f"Exception in {domain}/{example_id}: {e}")
                if hasattr(env, "controller") and env.controller is not None:
                    env.controller.end_recording(
                        os.path.join(example_result_dir, "recording.mp4")
                    )
                with open(os.path.join(example_result_dir, "traj.jsonl"), "a") as f:
                    f.write(json.dumps({"Error": f"Failed: {domain}/{example_id}: {e}"}))
                    f.write("\n")

            # Reset agent for next task
            agent.reset()

    env.close()
    if scores:
        avg = sum(scores) / len(scores)
        logger.info(f"Average score: {avg} ({sum(scores)}/{len(scores)})")
        print(f"\n{'='*60}")
        print(f"Results: {len(scores)}/{total} tasks completed")
        print(f"Average score: {avg:.4f}")
        print(f"{'='*60}")
    else:
        logger.warning("No completed tasks")


if __name__ == "__main__":
    main()
