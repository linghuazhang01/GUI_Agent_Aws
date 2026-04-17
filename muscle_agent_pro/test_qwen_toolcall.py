"""Quick verification: test Qwen3.5 tool calling via DashScope.

Usage:
    python test_qwen_toolcall.py --api_key YOUR_DASHSCOPE_KEY
    # or
    DASHSCOPE_API_KEY=xxx python test_qwen_toolcall.py
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from muscle_mem.core.engine import LMMEngineOpenAI

DASHSCOPE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL = "qwen3.5-397b-a17b"


def test_basic_call(engine):
    """Test 1: Basic non-tool call."""
    print("\n=== Test 1: Basic call (no tools) ===")
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
        {"role": "user", "content": [{"type": "text", "text": "Say 'hello world' and nothing else."}]},
    ]
    result = engine.generate(messages, temperature=0.0)
    print(f"Result type: {type(result)}")
    print(f"Result: {result}")
    assert isinstance(result, str), "Expected string for non-tool call"
    print("PASS")


def test_tool_call(engine):
    """Test 2: Tool calling with format conversion."""
    print("\n=== Test 2: Tool calling ===")

    # Anthropic-format messages (what Worker produces)
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful weather assistant."}]},
        {"role": "user", "content": [{"type": "text", "text": "What's the weather in Beijing?"}]},
    ]

    # Anthropic-format tool schemas (what ToolRegistry produces)
    tools = [
        {
            "name": "get_weather",
            "description": "Get the weather for a given city",
            "input_schema": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                },
                "required": ["city"],
            },
        }
    ]

    result = engine.generate(
        messages,
        temperature=0.0,
        tools=tools,
        tool_choice={"type": "any"},
    )

    print(f"Result type: {type(result)}")
    print(f"Result: {json.dumps(result, indent=2, ensure_ascii=False)}")

    assert isinstance(result, dict), f"Expected dict for tool call, got {type(result)}"
    assert "content" in result, "Expected 'content' key"
    assert "stop_reason" in result, "Expected 'stop_reason' key"

    # Find tool_use block
    tool_use = None
    for block in result["content"]:
        if block.get("type") == "tool_use":
            tool_use = block
            break

    assert tool_use is not None, "Expected tool_use block in response"
    assert tool_use["name"] == "get_weather", f"Expected tool name 'get_weather', got {tool_use['name']}"
    assert "city" in tool_use["input"], f"Expected 'city' in tool input, got {tool_use['input']}"

    print(f"Tool name: {tool_use['name']}")
    print(f"Tool input: {tool_use['input']}")
    print("PASS")


def test_tool_result_roundtrip(engine):
    """Test 3: Full tool calling roundtrip (call -> result -> continue)."""
    print("\n=== Test 3: Tool result roundtrip ===")

    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful weather assistant. Use the get_weather tool."}]},
        {"role": "user", "content": [{"type": "text", "text": "What's the weather in Beijing?"}]},
    ]

    tools = [
        {
            "name": "get_weather",
            "description": "Get the weather for a given city",
            "input_schema": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                },
                "required": ["city"],
            },
        }
    ]

    # First call: get tool_use
    result1 = engine.generate(messages, temperature=0.0, tools=tools, tool_choice={"type": "any"})
    tool_use = None
    for block in result1["content"]:
        if block.get("type") == "tool_use":
            tool_use = block
            break
    assert tool_use is not None, "First call should return tool_use"

    print(f"Step 1 - Tool call: {tool_use['name']}({tool_use['input']})")

    # Append assistant message with tool_use (Anthropic format, as Worker does)
    messages.append({"role": "assistant", "content": result1["content"]})
    # Append tool_result (Anthropic format, as Worker does)
    messages.append({
        "role": "user",
        "content": [{
            "type": "tool_result",
            "tool_use_id": tool_use["id"],
            "content": "Beijing: sunny, 25°C",
        }],
    })

    # Second call: should produce text response
    result2 = engine.generate(messages, temperature=0.0, tools=tools)
    print(f"Step 2 - Response: {result2}")
    print("PASS")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, default=os.getenv("DASHSCOPE_API_KEY", ""))
    parser.add_argument("--model", type=str, default=MODEL)
    args = parser.parse_args()

    if not args.api_key:
        print("Please provide --api_key or set DASHSCOPE_API_KEY")
        sys.exit(1)

    engine = LMMEngineOpenAI(
        base_url=DASHSCOPE_URL,
        api_key=args.api_key,
        model=args.model,
    )

    print(f"Testing Qwen3.5 tool calling via DashScope")
    print(f"Model: {args.model}")
    print(f"Endpoint: {DASHSCOPE_URL}")

    test_basic_call(engine)
    test_tool_call(engine)
    test_tool_result_roundtrip(engine)

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
