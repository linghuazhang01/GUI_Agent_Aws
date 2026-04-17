import json
import os
from typing import Any, Dict, List, Optional

import backoff
from anthropic import Anthropic
import httpx
from openai import (
    AzureOpenAI,
    APIConnectionError,
    APIError,
    AzureOpenAI,
    OpenAI,
    RateLimitError,
)


class LMMEngine:
    pass


class LMMEngineOpenAI(LMMEngine):
    def __init__(
        self,
        base_url=None,
        api_key=None,
        model=None,
        rate_limit=-1,
        temperature=None,
        organization=None,
        **kwargs,
    ):
        assert model is not None, "model must be provided"
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.organization = organization
        self.request_interval = 0 if rate_limit == -1 else 60.0 / rate_limit
        self.llm_client = None
        self.temperature = temperature  # Can force temperature to be the same (in the case of o3 requiring temperature to be 1)

    # ---- Anthropic <-> OpenAI format conversion helpers ----

    @staticmethod
    def _convert_tools_to_openai(tools: List[Dict]) -> List[Dict]:
        """Convert Anthropic-format tool schemas to OpenAI format."""
        openai_tools = []
        for tool in tools:
            func = {
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", tool.get("parameters", {})),
            }
            openai_tools.append({"type": "function", "function": func})
        return openai_tools

    @staticmethod
    def _convert_tool_choice(tool_choice) -> Optional[str]:
        """Convert Anthropic tool_choice to OpenAI format."""
        if tool_choice is None:
            return None
        if isinstance(tool_choice, str):
            if tool_choice == "any":
                return "required"
            if tool_choice == "auto":
                return "auto"
            return tool_choice
        if isinstance(tool_choice, dict):
            t = tool_choice.get("type", "auto")
            if t == "any":
                return "required"
            if t == "tool":
                return {
                    "type": "function",
                    "function": {"name": tool_choice.get("name", "")},
                }
            return "auto"
        return "auto"

    @staticmethod
    def _convert_messages_to_openai(messages: List[Dict]) -> List[Dict]:
        """Convert Anthropic-format messages to OpenAI-compatible format.

        Handles: system prompt, tool_use blocks, tool_result blocks,
        image blocks, and text blocks.
        """
        openai_messages: List[Dict] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content")

            # Handle string content (e.g. simple system/user messages)
            if isinstance(content, str):
                openai_messages.append({"role": role, "content": content})
                continue

            if not isinstance(content, list):
                openai_messages.append({"role": role, "content": str(content)})
                continue

            # Check for Anthropic tool_result blocks (role=user with tool_result)
            if role == "user":
                tool_result_blocks = [
                    b for b in content if isinstance(b, dict) and b.get("type") == "tool_result"
                ]
                if tool_result_blocks:
                    for block in tool_result_blocks:
                        tr_content = block.get("content", "")
                        if isinstance(tr_content, list):
                            tr_text = " ".join(
                                t.get("text", "") for t in tr_content if isinstance(t, dict)
                            )
                        else:
                            tr_text = str(tr_content) if tr_content else ""
                        openai_messages.append({
                            "role": "tool",
                            "tool_call_id": block.get("tool_use_id", ""),
                            "content": tr_text,
                        })
                    # Also include any non-tool_result content as separate user msg
                    other_blocks = [
                        b for b in content
                        if isinstance(b, dict) and b.get("type") != "tool_result"
                    ]
                    if other_blocks:
                        om = LMMEngineOpenAI._build_openai_content(role, other_blocks)
                        if om:
                            openai_messages.append(om)
                    continue

            # Check for assistant messages with tool_use blocks
            if role == "assistant":
                tool_use_blocks = [
                    b for b in content if isinstance(b, dict) and b.get("type") == "tool_use"
                ]
                if tool_use_blocks:
                    # Build assistant message with tool_calls
                    text_parts = []
                    for b in content:
                        if isinstance(b, dict) and b.get("type") == "text":
                            text_parts.append(b.get("text", ""))
                    assistant_msg: Dict[str, Any] = {
                        "role": "assistant",
                        "content": "\n".join(text_parts) if text_parts else None,
                    }
                    tool_calls = []
                    for b in tool_use_blocks:
                        inp = b.get("input", {})
                        tool_calls.append({
                            "id": b.get("id", ""),
                            "type": "function",
                            "function": {
                                "name": b.get("name", ""),
                                "arguments": json.dumps(inp, ensure_ascii=False),
                            },
                        })
                    assistant_msg["tool_calls"] = tool_calls
                    openai_messages.append(assistant_msg)
                    continue

            # Default: build content from blocks
            om = LMMEngineOpenAI._build_openai_content(role, content)
            if om:
                openai_messages.append(om)

        return openai_messages

    @staticmethod
    def _build_openai_content(role: str, blocks: List[Dict]) -> Optional[Dict]:
        """Build an OpenAI-format message from content blocks."""
        parts = []
        for block in blocks:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                text = block.get("text", "")
                if text:
                    parts.append({"type": "text", "text": text})
            elif btype == "image":
                # Anthropic image block -> OpenAI image_url
                source = block.get("source", {})
                media_type = source.get("media_type", "image/png")
                data = source.get("data", "")
                parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{media_type};base64,{data}"},
                })
            elif btype == "image_url":
                parts.append(block)
            elif btype == "thinking":
                # Skip thinking blocks for OpenAI
                pass
        if not parts:
            return None
        if all(p.get("type") == "text" for p in parts):
            return {"role": role, "content": "\n".join(p["text"] for p in parts)}
        return {"role": role, "content": parts}

    @staticmethod
    def _normalize_openai_tool_response(response_msg) -> Dict:
        """Convert OpenAI tool_call response to Anthropic-format dict."""
        content_blocks = []

        # Extract text content
        text_content = getattr(response_msg, "content", None) or ""
        if text_content:
            content_blocks.append({"type": "text", "text": text_content})

        # Extract tool_calls
        tool_calls = getattr(response_msg, "tool_calls", None) or []
        for tc in tool_calls:
            name = getattr(tc.function, "name", "") if hasattr(tc, "function") else ""
            arguments = getattr(tc.function, "arguments", "{}") if hasattr(tc, "function") else "{}"
            try:
                input_dict = json.loads(arguments)
            except (json.JSONDecodeError, TypeError):
                input_dict = {"raw_arguments": arguments}
            content_blocks.append({
                "type": "tool_use",
                "id": getattr(tc, "id", ""),
                "name": name,
                "input": input_dict,
            })

        return {
            "content": content_blocks,
            "stop_reason": "tool_use" if tool_calls else "end_turn",
            "model": None,
        }

    def _get_client(self):
        api_key = self.api_key or os.getenv("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError(
                "An API Key needs to be provided in either the api_key parameter or as an environment variable named OPENAI_API_KEY"
            )
        organization = self.organization or os.getenv("OPENAI_ORG_ID")
        if not self.llm_client:
            if not self.base_url:
                self.llm_client = OpenAI(api_key=api_key, organization=organization)
            else:
                self.llm_client = OpenAI(
                    base_url=self.base_url, api_key=api_key, organization=organization
                )
        return self.llm_client

    @backoff.on_exception(
        backoff.expo, (APIConnectionError, APIError, RateLimitError), max_time=60
    )
    def generate(self, messages, temperature=0.0, max_new_tokens=None, **kwargs):
        client = self._get_client()
        request_kwargs = dict(kwargs)
        tools = request_kwargs.pop("tools", None)
        tool_choice = request_kwargs.pop("tool_choice", None)

        if tools is not None:
            # Tool-calling mode: convert Anthropic format -> OpenAI format
            openai_messages = self._convert_messages_to_openai(messages)
            openai_tools = self._convert_tools_to_openai(tools)
            openai_tool_choice = self._convert_tool_choice(tool_choice)

            create_kwargs: Dict[str, Any] = {
                "model": self.model,
                "messages": openai_messages,
                "temperature": temperature if self.temperature is None else self.temperature,
                "tools": openai_tools,
            }
            if openai_tool_choice is not None:
                create_kwargs["tool_choice"] = openai_tool_choice
            if max_new_tokens:
                create_kwargs["max_tokens"] = max_new_tokens
            # Disable thinking mode for tool calling (DashScope Qwen3.x
            # enables thinking by default, which conflicts with
            # tool_choice="required").
            extra_body = request_kwargs.pop("extra_body", None)
            if extra_body is not None:
                create_kwargs["extra_body"] = extra_body
            else:
                create_kwargs["extra_body"] = {"enable_thinking": False}
            create_kwargs.update(request_kwargs)

            completion_message = client.chat.completions.create(**create_kwargs).choices[0].message
            return self._normalize_openai_tool_response(completion_message)

        # Non-tool mode: backward compatible, return string
        completion_message = (
            client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=(
                    temperature if self.temperature is None else self.temperature
                ),
                **request_kwargs,
            )
            .choices[0]
            .message
        )
        return completion_message.content

    def generate_with_thinking(
        self, messages, temperature=0.0, max_new_tokens=None, **kwargs
    ):
        client = self._get_client()
        request_kwargs = dict(kwargs)
        tools = request_kwargs.pop("tools", None)
        tool_choice = request_kwargs.pop("tool_choice", None)

        if tools is not None:
            openai_messages = self._convert_messages_to_openai(messages)
            openai_tools = self._convert_tools_to_openai(tools)
            openai_tool_choice = self._convert_tool_choice(tool_choice)
            create_kwargs: Dict[str, Any] = {
                "model": self.model,
                "messages": openai_messages,
                "temperature": temperature if self.temperature is None else self.temperature,
                "tools": openai_tools,
            }
            if openai_tool_choice is not None:
                create_kwargs["tool_choice"] = openai_tool_choice
            if max_new_tokens:
                create_kwargs["max_tokens"] = max_new_tokens
            create_kwargs.update(request_kwargs)
            completion_message = client.chat.completions.create(**create_kwargs).choices[0].message
            return self._normalize_openai_tool_response(completion_message)

        completion_message = (
            client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=(
                    temperature if self.temperature is None else self.temperature
                ),
                **request_kwargs,
            )
            .choices[0]
            .message
        )
        thinking = getattr(completion_message, "reasoning_content", None)
        return completion_message.content, thinking


class LMMEngineAnthropic(LMMEngine):
    def __init__(
        self,
        base_url=None,
        api_key=None,
        model=None,
        thinking=False,
        temperature=None,
        prompt_caching=True,
        prompt_cache_ttl=None,
        **kwargs,
    ):
        assert model is not None, "model must be provided"
        self.base_url = base_url
        self.model = model
        self.thinking = thinking
        self.api_key = api_key
        self.llm_client = None
        self.temperature = temperature
        self.prompt_caching = prompt_caching
        self.prompt_cache_ttl = prompt_cache_ttl

    def _normalize_tool_response(self, response):
        content_blocks = []
        for block in response.content:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                content_blocks.append({"type": "text", "text": block.text})
            elif block_type == "tool_use":
                content_blocks.append(
                    {
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    }
                )
            elif block_type == "thinking":
                content_blocks.append({"type": "thinking", "thinking": block.thinking})
            else:
                content_blocks.append(
                    {"type": block_type or "unknown", "text": str(block)}
                )
        return {
            "content": content_blocks,
            "stop_reason": getattr(response, "stop_reason", None),
            "model": getattr(response, "model", None),
        }

    def _build_cache_control(self):
        cache_control = {"type": "ephemeral"}
        if self.prompt_cache_ttl:
            cache_control["ttl"] = self.prompt_cache_ttl
        return cache_control

    def _apply_prompt_caching(self, messages, tools):
        if not self.prompt_caching:
            return messages[0]["content"][0]["text"], messages[1:], tools, None

        system_content = messages[0].get("content", [])
        system_blocks = []
        if isinstance(system_content, list):
            for block in system_content:
                block_copy = dict(block)
                block_copy.setdefault("cache_control", self._build_cache_control())
                system_blocks.append(block_copy)
        else:
            system_blocks = [
                {
                    "type": "text",
                    "text": system_content,
                    "cache_control": self._build_cache_control(),
                }
            ]

        updated_messages = messages[1:]

        updated_tools = None
        if tools:
            updated_tools = [dict(tool) for tool in tools]
            updated_tool = dict(updated_tools[-1])
            updated_tool.setdefault("cache_control", self._build_cache_control())
            updated_tools[-1] = updated_tool

        extra_headers = {"anthropic-beta": "prompt-caching-2024-07-31"}
        return system_blocks, updated_messages, updated_tools, extra_headers

    @backoff.on_exception(
        backoff.expo, (APIConnectionError, APIError, RateLimitError), max_time=60
    )
    def generate(self, messages, temperature=0.0, max_new_tokens=None, **kwargs):
        api_key = self.api_key or os.getenv("ANTHROPIC_API_KEY")
        if api_key is None:
            raise ValueError(
                "An API Key needs to be provided in either the api_key parameter or as an environment variable named ANTHROPIC_API_KEY"
            )

        print(f"Using Anthropic base_url: {self.base_url}, model: {self.model}")
        self.llm_client = Anthropic(
            base_url=self.base_url, api_key="", auth_token=api_key
        )
        # Use the instance temperature if not specified in the call
        temp = self.temperature if temperature is None else temperature
        request_kwargs = dict(kwargs)
        tools = request_kwargs.get("tools")
        system = messages[0]["content"][0]["text"]
        message_payload = messages[1:]
        if self.prompt_caching:
            system, messages, cached_tools, extra_headers = self._apply_prompt_caching(
                messages, tools
            )
            if cached_tools is not None:
                request_kwargs["tools"] = cached_tools
            if extra_headers:
                existing_headers = request_kwargs.get("extra_headers")
                if isinstance(existing_headers, dict):
                    existing_headers.update(extra_headers)
                else:
                    request_kwargs["extra_headers"] = dict(extra_headers)
            message_payload = messages
        if self.thinking:
            full_response = self.llm_client.messages.create(
                system=system,
                model=self.model,
                messages=message_payload,
                max_tokens=8192,
                thinking={"type": "enabled", "budget_tokens": 4096},
                **request_kwargs,
            )
            if request_kwargs.get("tools"):
                return self._normalize_tool_response(full_response)
            return full_response.content[1].text
        response = self.llm_client.messages.create(
            system=system,
            model=self.model,
            messages=message_payload,
            max_tokens=max_new_tokens if max_new_tokens else 4096,
            temperature=temp,
            **request_kwargs,
        )
        if request_kwargs.get("tools"):
            return self._normalize_tool_response(response)
        return response.content[0].text

    @backoff.on_exception(
        backoff.expo, (APIConnectionError, APIError, RateLimitError), max_time=60
    )
    # Compatible with Claude-3.7 Sonnet thinking mode
    def generate_with_thinking(
        self, messages, temperature=0.0, max_new_tokens=None, **kwargs
    ):
        """Generate the next message based on previous messages, and keeps the thinking tokens"""
        api_key = self.api_key or os.getenv("ANTHROPIC_API_KEY")
        if api_key is None:
            raise ValueError(
                "An API Key needs to be provided in either the api_key parameter or as an environment variable named ANTHROPIC_API_KEY"
            )
        self.llm_client = Anthropic(api_key=api_key)
        request_kwargs = dict(kwargs)
        tools = request_kwargs.get("tools")
        system = messages[0]["content"][0]["text"]
        message_payload = messages[1:]
        if self.prompt_caching:
            system, messages, cached_tools, extra_headers = self._apply_prompt_caching(
                messages, tools
            )
            if cached_tools is not None:
                request_kwargs["tools"] = cached_tools
            if extra_headers:
                existing_headers = request_kwargs.get("extra_headers")
                if isinstance(existing_headers, dict):
                    existing_headers.update(extra_headers)
                else:
                    request_kwargs["extra_headers"] = dict(extra_headers)
            message_payload = messages
        full_response = self.llm_client.messages.create(
            system=system,
            model=self.model,
            messages=message_payload,
            max_tokens=8192,
            thinking={"type": "enabled", "budget_tokens": 4096},
            **request_kwargs,
        )

        if request_kwargs.get("tools"):
            return self._normalize_tool_response(full_response)

        thoughts = full_response.content[0].thinking
        answer = full_response.content[1].text
        full_response = (
            f"<thoughts>\n{thoughts}\n</thoughts>\n\n<answer>\n{answer}\n</answer>\n"
        )
        return full_response


class LMMEngineAnthropicLR(LMMEngine):
    def __init__(
        self,
        base_url=None,
        api_key=None,
        model=None,
        thinking=False,
        temperature=None,
        prompt_caching=False,
        prompt_cache_ttl=None,
        **kwargs,
    ):
        assert model is not None, "model must be provided"
        self.base_url = base_url
        self.model = model
        self.thinking = thinking
        self.api_key = api_key
        self.llm_client = None
        self.temperature = temperature
        self.prompt_caching = prompt_caching
        self.prompt_cache_ttl = prompt_cache_ttl

    def _normalize_tool_response(self, response):
        content_blocks = []
        for block in response.content:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                content_blocks.append({"type": "text", "text": block.text})
            elif block_type == "tool_use":
                content_blocks.append(
                    {
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    }
                )
            elif block_type == "thinking":
                content_blocks.append({"type": "thinking", "thinking": block.thinking})
            else:
                content_blocks.append(
                    {"type": block_type or "unknown", "text": str(block)}
                )
        return {
            "content": content_blocks,
            "stop_reason": getattr(response, "stop_reason", None),
            "model": getattr(response, "model", None),
        }

    def _build_cache_control(self):
        cache_control = {"type": "ephemeral"}
        if self.prompt_cache_ttl:
            cache_control["ttl"] = self.prompt_cache_ttl
        return cache_control

    def _apply_prompt_caching(self, messages, tools):
        if not self.prompt_caching:
            return messages[0]["content"][0]["text"], messages[1:], tools, None

        system_content = messages[0].get("content", [])
        system_blocks = []
        if isinstance(system_content, list):
            for block in system_content:
                block_copy = dict(block)
                block_copy.setdefault("cache_control", self._build_cache_control())
                system_blocks.append(block_copy)
        else:
            system_blocks = [
                {
                    "type": "text",
                    "text": system_content,
                    "cache_control": self._build_cache_control(),
                }
            ]

        updated_messages = messages[1:]

        updated_tools = None
        if tools:
            updated_tools = [dict(tool) for tool in tools]
            updated_tool = dict(updated_tools[-1])
            updated_tool.setdefault("cache_control", self._build_cache_control())
            updated_tools[-1] = updated_tool

        extra_headers = {"anthropic-beta": "prompt-caching-2024-07-31"}
        return system_blocks, updated_messages, updated_tools, extra_headers

    @backoff.on_exception(
        backoff.expo, (APIConnectionError, APIError, RateLimitError), max_time=60
    )
    def generate(self, messages, temperature=0.0, max_new_tokens=None, **kwargs):
        api_key = self.api_key or os.getenv("ANTHROPIC_API_KEY")
        if api_key is None:
            raise ValueError(
                "An API Key needs to be provided in either the api_key parameter or as an environment variable named ANTHROPIC_API_KEY"
            )

        print(f"Using Anthropic base_url: {self.base_url}, model: {self.model}")

        disable_ssl = True
        client_kwargs: Dict[str, Any] = {"api_key": api_key, "base_url": self.base_url}
        if disable_ssl:
            client_kwargs["http_client"] = httpx.Client(verify=False)
        self.llm_client = Anthropic(**client_kwargs)

        # Use the instance temperature if not specified in the call
        temp = self.temperature if temperature is None else temperature
        request_kwargs = dict(kwargs)
        tools = request_kwargs.get("tools")
        system = messages[0]["content"][0]["text"]
        message_payload = messages[1:]
        if self.prompt_caching:
            system, messages, cached_tools, extra_headers = self._apply_prompt_caching(
                messages, tools
            )
            if cached_tools is not None:
                request_kwargs["tools"] = cached_tools
            if extra_headers:
                existing_headers = request_kwargs.get("extra_headers")
                if isinstance(existing_headers, dict):
                    existing_headers.update(extra_headers)
                else:
                    request_kwargs["extra_headers"] = dict(extra_headers)
            message_payload = messages
        if self.thinking:
            full_response = self.llm_client.messages.create(
                system=system,
                model=self.model,
                messages=message_payload,
                max_tokens=8192,
                thinking={"type": "enabled", "budget_tokens": 4096},
                **request_kwargs,
            )
            if request_kwargs.get("tools"):
                return self._normalize_tool_response(full_response)
            return full_response.content[1].text
        response = self.llm_client.messages.create(
            system=system,
            model=self.model,
            messages=message_payload,
            max_tokens=max_new_tokens if max_new_tokens else 4096,
            temperature=temp,
            **request_kwargs,
        )
        if request_kwargs.get("tools"):
            return self._normalize_tool_response(response)
        return response.content[0].text

    @backoff.on_exception(
        backoff.expo, (APIConnectionError, APIError, RateLimitError), max_time=60
    )
    # Compatible with Claude-3.7 Sonnet thinking mode
    def generate_with_thinking(
        self, messages, temperature=0.0, max_new_tokens=None, **kwargs
    ):
        """Generate the next message based on previous messages, and keeps the thinking tokens"""
        api_key = self.api_key or os.getenv("ANTHROPIC_API_KEY")
        if api_key is None:
            raise ValueError(
                "An API Key needs to be provided in either the api_key parameter or as an environment variable named ANTHROPIC_API_KEY"
            )
        self.llm_client = Anthropic(api_key=api_key)
        request_kwargs = dict(kwargs)
        tools = request_kwargs.get("tools")
        system = messages[0]["content"][0]["text"]
        message_payload = messages[1:]
        if self.prompt_caching:
            system, messages, cached_tools, extra_headers = self._apply_prompt_caching(
                messages, tools
            )
            if cached_tools is not None:
                request_kwargs["tools"] = cached_tools
            if extra_headers:
                existing_headers = request_kwargs.get("extra_headers")
                if isinstance(existing_headers, dict):
                    existing_headers.update(extra_headers)
                else:
                    request_kwargs["extra_headers"] = dict(extra_headers)
            message_payload = messages
        full_response = self.llm_client.messages.create(
            system=system,
            model=self.model,
            messages=message_payload,
            max_tokens=8192,
            thinking={"type": "enabled", "budget_tokens": 4096},
            **request_kwargs,
        )

        if request_kwargs.get("tools"):
            return self._normalize_tool_response(full_response)

        thoughts = full_response.content[0].thinking
        answer = full_response.content[1].text
        full_response = (
            f"<thoughts>\n{thoughts}\n</thoughts>\n\n<answer>\n{answer}\n</answer>\n"
        )
        return full_response


class LMMEngineGemini(LMMEngine):
    def __init__(
        self,
        base_url=None,
        api_key=None,
        model=None,
        rate_limit=-1,
        temperature=None,
        **kwargs,
    ):
        assert model is not None, "model must be provided"
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.request_interval = 0 if rate_limit == -1 else 60.0 / rate_limit
        self.llm_client = None
        self.temperature = temperature

    @backoff.on_exception(
        backoff.expo, (APIConnectionError, APIError, RateLimitError), max_time=60
    )
    def generate(self, messages, temperature=0.0, max_new_tokens=None, **kwargs):
        api_key = self.api_key or os.getenv("GEMINI_API_KEY")
        if api_key is None:
            raise ValueError(
                "An API Key needs to be provided in either the api_key parameter or as an environment variable named GEMINI_API_KEY"
            )
        base_url = self.base_url or os.getenv("GEMINI_ENDPOINT_URL")
        if base_url is None:
            raise ValueError(
                "An endpoint URL needs to be provided in either the endpoint_url parameter or as an environment variable named GEMINI_ENDPOINT_URL"
            )
        if not self.llm_client:
            self.llm_client = OpenAI(base_url=base_url, api_key=api_key)
        # Use the temperature passed to generate, otherwise use the instance's temperature, otherwise default to 0.0
        temp = self.temperature if temperature is None else temperature
        return (
            self.llm_client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_new_tokens if max_new_tokens else 4096,
                temperature=temp,
                **kwargs,
            )
            .choices[0]
            .message.content
        )


class LMMEngineOpenRouter(LMMEngine):
    def __init__(
        self,
        base_url=None,
        api_key=None,
        model=None,
        rate_limit=-1,
        temperature=None,
        **kwargs,
    ):
        assert model is not None, "model must be provided"
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.request_interval = 0 if rate_limit == -1 else 60.0 / rate_limit
        self.llm_client = None
        self.temperature = temperature

    @backoff.on_exception(
        backoff.expo, (APIConnectionError, APIError, RateLimitError), max_time=60
    )
    def generate(self, messages, temperature=0.0, max_new_tokens=None, **kwargs):
        api_key = self.api_key or os.getenv("OPENROUTER_API_KEY")
        if api_key is None:
            raise ValueError(
                "An API Key needs to be provided in either the api_key parameter or as an environment variable named OPENROUTER_API_KEY"
            )
        base_url = self.base_url or os.getenv("OPEN_ROUTER_ENDPOINT_URL")
        if base_url is None:
            raise ValueError(
                "An endpoint URL needs to be provided in either the endpoint_url parameter or as an environment variable named OPEN_ROUTER_ENDPOINT_URL"
            )
        if not self.llm_client:
            self.llm_client = OpenAI(base_url=base_url, api_key=api_key)
        # Use self.temperature if set, otherwise use the temperature argument
        temp = self.temperature if self.temperature is not None else temperature
        return (
            self.llm_client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_new_tokens if max_new_tokens else 4096,
                temperature=temp,
                **kwargs,
            )
            .choices[0]
            .message.content
        )


class LMMEngineAzureOpenAI(LMMEngine):
    def __init__(
        self,
        base_url=None,
        api_key=None,
        azure_endpoint=None,
        model=None,
        api_version=None,
        rate_limit=-1,
        temperature=None,
        **kwargs,
    ):
        assert model is not None, "model must be provided"
        self.model = model
        self.api_version = api_version
        self.api_key = api_key
        self.azure_endpoint = azure_endpoint
        self.request_interval = 0 if rate_limit == -1 else 60.0 / rate_limit
        self.llm_client = None
        self.cost = 0.0
        self.temperature = temperature

    @backoff.on_exception(
        backoff.expo, (APIConnectionError, APIError, RateLimitError), max_time=60
    )
    def generate(self, messages, temperature=0.0, max_new_tokens=None, **kwargs):
        api_key = self.api_key or os.getenv("AZURE_OPENAI_API_KEY")
        if api_key is None:
            raise ValueError(
                "An API Key needs to be provided in either the api_key parameter or as an environment variable named AZURE_OPENAI_API_KEY"
            )
        api_version = self.api_version or os.getenv("OPENAI_API_VERSION")
        if api_version is None:
            raise ValueError(
                "api_version must be provided either as a parameter or as an environment variable named OPENAI_API_VERSION"
            )
        azure_endpoint = self.azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        if azure_endpoint is None:
            raise ValueError(
                "An Azure API endpoint needs to be provided in either the azure_endpoint parameter or as an environment variable named AZURE_OPENAI_ENDPOINT"
            )
        if not self.llm_client:
            self.llm_client = AzureOpenAI(
                azure_endpoint=azure_endpoint,
                api_key=api_key,
                api_version=api_version,
            )
        # Use self.temperature if set, otherwise use the temperature argument
        temp = self.temperature if self.temperature is not None else temperature
        completion = self.llm_client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_new_tokens if max_new_tokens else 4096,
            temperature=temp,
            **kwargs,
        )
        total_tokens = completion.usage.total_tokens
        self.cost += 0.02 * ((total_tokens + 500) / 1000)
        return completion.choices[0].message.content


class LMMEnginevLLM(LMMEngine):
    def __init__(
        self,
        base_url=None,
        api_key=None,
        model=None,
        rate_limit=-1,
        temperature=None,
        **kwargs,
    ):
        assert model is not None, "model must be provided"
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.request_interval = 0 if rate_limit == -1 else 60.0 / rate_limit
        self.llm_client = None
        self.temperature = temperature

    @backoff.on_exception(
        backoff.expo, (APIConnectionError, APIError, RateLimitError), max_time=60
    )
    def generate(
        self,
        messages,
        temperature=0.0,
        top_p=0.8,
        repetition_penalty=1.05,
        max_new_tokens=512,
        **kwargs,
    ):
        api_key = self.api_key or os.getenv("vLLM_API_KEY")
        if api_key is None:
            raise ValueError(
                "A vLLM API key needs to be provided in either the api_key parameter or as an environment variable named vLLM_API_KEY"
            )
        base_url = self.base_url or os.getenv("vLLM_ENDPOINT_URL")
        if base_url is None:
            raise ValueError(
                "An endpoint URL needs to be provided in either the endpoint_url parameter or as an environment variable named vLLM_ENDPOINT_URL"
            )
        if not self.llm_client:
            self.llm_client = OpenAI(base_url=base_url, api_key=api_key)
        # Use self.temperature if set, otherwise use the temperature argument
        temp = self.temperature if self.temperature is not None else temperature
        completion = self.llm_client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_new_tokens if max_new_tokens else 4096,
            temperature=temp,
            top_p=top_p,
            extra_body={"repetition_penalty": repetition_penalty},
        )
        return completion.choices[0].message.content


class LMMEngineHuggingFace(LMMEngine):
    def __init__(self, base_url=None, api_key=None, rate_limit=-1, **kwargs):
        self.base_url = base_url
        self.api_key = api_key
        self.request_interval = 0 if rate_limit == -1 else 60.0 / rate_limit
        self.llm_client = None

    @backoff.on_exception(
        backoff.expo, (APIConnectionError, APIError, RateLimitError), max_time=60
    )
    def generate(self, messages, temperature=0.0, max_new_tokens=None, **kwargs):
        api_key = self.api_key or os.getenv("HF_TOKEN")
        if api_key is None:
            raise ValueError(
                "A HuggingFace token needs to be provided in either the api_key parameter or as an environment variable named HF_TOKEN"
            )
        base_url = self.base_url or os.getenv("HF_ENDPOINT_URL")
        if base_url is None:
            raise ValueError(
                "HuggingFace endpoint must be provided as base_url parameter or as an environment variable named HF_ENDPOINT_URL."
            )
        if not self.llm_client:
            self.llm_client = OpenAI(base_url=base_url, api_key=api_key)
        return (
            self.llm_client.chat.completions.create(
                model="tgi",
                messages=messages,
                max_tokens=max_new_tokens if max_new_tokens else 4096,
                temperature=temperature,
                **kwargs,
            )
            .choices[0]
            .message.content
        )


class LMMEngineParasail(LMMEngine):
    def __init__(
        self, base_url=None, api_key=None, model=None, rate_limit=-1, **kwargs
    ):
        assert model is not None, "Parasail model id must be provided"
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.request_interval = 0 if rate_limit == -1 else 60.0 / rate_limit
        self.llm_client = None

    @backoff.on_exception(
        backoff.expo, (APIConnectionError, APIError, RateLimitError), max_time=60
    )
    def generate(self, messages, temperature=0.0, max_new_tokens=None, **kwargs):
        api_key = self.api_key or os.getenv("PARASAIL_API_KEY")
        if api_key is None:
            raise ValueError(
                "A Parasail API key needs to be provided in either the api_key parameter or as an environment variable named PARASAIL_API_KEY"
            )
        base_url = self.base_url
        if base_url is None:
            raise ValueError(
                "Parasail endpoint must be provided as base_url parameter or as an environment variable named PARASAIL_ENDPOINT_URL"
            )
        if not self.llm_client:
            self.llm_client = OpenAI(
                base_url=base_url if base_url else "https://api.parasail.io/v1",
                api_key=api_key,
            )
        return (
            self.llm_client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_new_tokens if max_new_tokens else 4096,
                temperature=temperature,
                **kwargs,
            )
            .choices[0]
            .message.content
        )
