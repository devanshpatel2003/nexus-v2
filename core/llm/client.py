"""
LLM Client — Multi-provider wrapper.
Routes chat completions to OpenAI, Google Gemini, or Anthropic Claude
and normalises every response to the OpenAI message shape so the
agent tool-calling loop stays unchanged.
"""

import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

from openai import OpenAI
from core.config import (
    get_openai_key, get_gemini_key, get_anthropic_key,
    LLM_MODEL, EMBEDDING_MODEL, ALL_MODELS,
)


# ---------------------------------------------------------------------------
# Normalised message wrapper
# ---------------------------------------------------------------------------

@dataclass
class ToolCallFunction:
    name: str
    arguments: str  # JSON string


@dataclass
class ToolCall:
    id: str
    type: str  # always "function"
    function: ToolCallFunction


@dataclass
class StandardMessage:
    """Matches the shape of openai.types.chat.ChatCompletionMessage."""
    content: Optional[str] = None
    role: str = "assistant"
    tool_calls: Optional[List[ToolCall]] = None


# ---------------------------------------------------------------------------
# Provider detection
# ---------------------------------------------------------------------------

def _provider_for_model(model: str) -> str:
    """Return 'openai', 'google', or 'anthropic' for a model id."""
    for info in ALL_MODELS.values():
        if info["id"] == model:
            return info["provider"]
    return "openai"  # fallback


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------

_openai_client = None


def _get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=get_openai_key())
    return _openai_client


def _openai_chat_completion(messages, tools, model, temperature) -> StandardMessage:
    client = _get_openai_client()
    kw: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 1024,
    }
    if tools:
        kw["tools"] = tools
        kw["tool_choice"] = "auto"
    resp = client.chat.completions.create(**kw)
    msg = resp.choices[0].message
    # Already the right shape — wrap in StandardMessage for consistency
    tcs = None
    if msg.tool_calls:
        tcs = [
            ToolCall(
                id=tc.id,
                type="function",
                function=ToolCallFunction(name=tc.function.name, arguments=tc.function.arguments),
            )
            for tc in msg.tool_calls
        ]
    return StandardMessage(content=msg.content, tool_calls=tcs)


# ---------------------------------------------------------------------------
# Google Gemini  (google-genai SDK)
# ---------------------------------------------------------------------------

def _openai_tools_to_gemini(tools: List[Dict]) -> list:
    """Convert OpenAI function-calling tool schemas → google.genai FunctionDeclaration list."""
    from google.genai import types

    decls = []
    for tool in tools:
        fn = tool["function"]
        decls.append(types.FunctionDeclaration(
            name=fn["name"],
            description=fn.get("description", ""),
            parameters_json_schema=fn.get("parameters"),
        ))
    return decls


def _gemini_chat_completion(messages, tools, model, temperature) -> StandardMessage:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=get_gemini_key())

    # Build Gemini tools
    gemini_tools = None
    if tools:
        gemini_tools = [types.Tool(function_declarations=_openai_tools_to_gemini(tools))]

    # Convert OpenAI messages → Gemini contents
    contents = []
    system_text = None
    for m in messages:
        role = m["role"]
        if role == "system":
            system_text = m["content"]
            continue
        if role == "tool":
            contents.append(types.Content(
                role="tool",
                parts=[types.Part.from_function_response(
                    name=m.get("name", "tool"),
                    response={"result": m["content"]},
                )],
            ))
            continue
        if role == "assistant":
            gemini_role = "model"
            if m.get("tool_calls"):
                parts = []
                for tc in m["tool_calls"]:
                    fn = tc["function"]
                    try:
                        args = json.loads(fn["arguments"])
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                    parts.append(types.Part.from_function_call(name=fn["name"], args=args))
                contents.append(types.Content(role=gemini_role, parts=parts))
                continue
        else:
            gemini_role = "user"

        text = m.get("content") or ""
        if gemini_role == "user" and system_text:
            text = f"[System instructions]\n{system_text}\n\n{text}"
            system_text = None

        contents.append(types.Content(
            role=gemini_role,
            parts=[types.Part.from_text(text=text)],
        ))

    config = types.GenerateContentConfig(
        tools=gemini_tools or [],
        temperature=temperature,
        max_output_tokens=1024,
    )
    response = client.models.generate_content(
        model=model, contents=contents, config=config,
    )

    # Normalise response
    text_parts = []
    tool_calls = []
    tc_idx = 0
    if response.candidates and response.candidates[0].content:
        for part in response.candidates[0].content.parts:
            if part.text:
                text_parts.append(part.text)
            if part.function_call:
                fc = part.function_call
                tool_calls.append(ToolCall(
                    id=f"gemini_tc_{tc_idx}",
                    type="function",
                    function=ToolCallFunction(
                        name=fc.name,
                        arguments=json.dumps(dict(fc.args)),
                    ),
                ))
                tc_idx += 1

    return StandardMessage(
        content="\n".join(text_parts) if text_parts else None,
        tool_calls=tool_calls if tool_calls else None,
    )


# ---------------------------------------------------------------------------
# Anthropic Claude
# ---------------------------------------------------------------------------

def _openai_tools_to_anthropic(tools: List[Dict]) -> List[Dict]:
    """Convert OpenAI function-calling tool schemas → Anthropic tool format."""
    out = []
    for tool in tools:
        fn = tool["function"]
        out.append({
            "name": fn["name"],
            "description": fn.get("description", ""),
            "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
        })
    return out


def _anthropic_chat_completion(messages, tools, model, temperature) -> StandardMessage:
    import anthropic

    client = anthropic.Anthropic(api_key=get_anthropic_key())

    # Extract system prompt
    system_text = ""
    api_messages = []
    for m in messages:
        role = m["role"]
        if role == "system":
            system_text = m["content"]
            continue
        if role == "tool":
            # Anthropic expects tool_result blocks
            api_messages.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": m.get("tool_call_id", ""),
                    "content": m["content"],
                }],
            })
            continue
        if role == "assistant" and m.get("tool_calls"):
            # Reconstruct assistant message with tool_use blocks
            blocks: list = []
            if m.get("content"):
                blocks.append({"type": "text", "text": m["content"]})
            for tc in m["tool_calls"]:
                fn = tc["function"]
                try:
                    inp = json.loads(fn["arguments"])
                except (json.JSONDecodeError, TypeError):
                    inp = {}
                blocks.append({
                    "type": "tool_use",
                    "id": tc["id"],
                    "name": fn["name"],
                    "input": inp,
                })
            api_messages.append({"role": "assistant", "content": blocks})
            continue

        api_messages.append({"role": role, "content": m.get("content", "")})

    # Anthropic requires alternating user/assistant — merge consecutive same-role
    merged: list = []
    for msg in api_messages:
        if merged and merged[-1]["role"] == msg["role"]:
            prev = merged[-1]["content"]
            cur = msg["content"]
            # Normalise both to list-of-blocks
            if isinstance(prev, str):
                prev = [{"type": "text", "text": prev}]
            if isinstance(cur, str):
                cur = [{"type": "text", "text": cur}]
            merged[-1]["content"] = prev + cur
        else:
            merged.append(msg)
    api_messages = merged

    kw: Dict[str, Any] = {
        "model": model,
        "max_tokens": 1024,
        "temperature": temperature,
        "messages": api_messages,
    }
    if system_text:
        kw["system"] = system_text
    if tools:
        kw["tools"] = _openai_tools_to_anthropic(tools)

    resp = client.messages.create(**kw)

    # Normalise response
    text_parts = []
    tool_calls = []
    for block in resp.content:
        if block.type == "text":
            text_parts.append(block.text)
        elif block.type == "tool_use":
            tool_calls.append(ToolCall(
                id=block.id,
                type="function",
                function=ToolCallFunction(
                    name=block.name,
                    arguments=json.dumps(block.input),
                ),
            ))

    return StandardMessage(
        content="\n".join(text_parts) if text_parts else None,
        tool_calls=tool_calls if tool_calls else None,
    )


# ---------------------------------------------------------------------------
# Public API (unchanged signature)
# ---------------------------------------------------------------------------

def chat_completion(
    messages: List[Dict],
    tools: Optional[List[Dict]] = None,
    model: str = LLM_MODEL,
    temperature: float = 0.3,
    **kwargs,
) -> StandardMessage:
    """
    Route a chat-completion call to the right provider.
    Returns a StandardMessage with .content and .tool_calls matching
    the OpenAI ChatCompletionMessage shape.
    """
    provider = kwargs.get("provider") or _provider_for_model(model)

    if provider == "google":
        return _gemini_chat_completion(messages, tools, model, temperature)
    elif provider == "anthropic":
        return _anthropic_chat_completion(messages, tools, model, temperature)
    else:
        return _openai_chat_completion(messages, tools, model, temperature)


# ---------------------------------------------------------------------------
# Embeddings (OpenAI only — unchanged)
# ---------------------------------------------------------------------------

def get_embeddings(texts: List[str], model: str = EMBEDDING_MODEL) -> List[List[float]]:
    client = _get_openai_client()
    response = client.embeddings.create(model=model, input=texts)
    return [item.embedding for item in response.data]


def get_embedding(text: str, model: str = EMBEDDING_MODEL) -> List[float]:
    return get_embeddings([text], model)[0]
