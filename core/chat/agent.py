"""
NEXUS v2 Chat Agent
Orchestrates RAG retrieval + tool calling + grounding enforcement.
"""

import json
from typing import List, Dict, Optional
from core.llm.client import chat_completion
from core.llm.prompts import SYSTEM_PROMPT, RAG_CONTEXT_TEMPLATE
from core.rag.retriever import retrieve_context, format_context_for_llm
from tools import event_study_tool, volatility_tool, ecosystem_tool, price_tool


# Tool registry
TOOLS = {
    "event_study_tool": event_study_tool,
    "volatility_tool": volatility_tool,
    "ecosystem_tool": ecosystem_tool,
    "price_tool": price_tool,
}

TOOL_SCHEMAS = [
    event_study_tool.SCHEMA,
    volatility_tool.SCHEMA,
    ecosystem_tool.SCHEMA,
    price_tool.SCHEMA,
]


def _execute_tool(name: str, arguments: Dict) -> str:
    """Execute a tool by name and return JSON string result."""
    tool_module = TOOLS.get(name)
    if not tool_module:
        return json.dumps({"error": f"Unknown tool: {name}"})

    try:
        result = tool_module.run(**arguments)
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": f"Tool execution failed: {str(e)[:200]}"})


def run_agent(
    user_message: str,
    conversation_history: List[Dict],
    max_tool_rounds: int = 3,
) -> Dict:
    """
    Run the NEXUS agent with RAG retrieval and tool calling.

    Returns:
        {
            "response": str,           # Final assistant message
            "citations": List[str],    # Retrieved doc IDs
            "tools_called": List[Dict],# Tools invoked with params
        }
    """
    # Step 1: Retrieve relevant context from knowledge base
    try:
        hits = retrieve_context(user_message)
        context_str = format_context_for_llm(hits)
        citations = [h["chunk_id"] for h in hits]
    except Exception:
        context_str = "Knowledge base not available. Answer using tools only."
        citations = []

    # Step 2: Build augmented user message
    augmented_message = RAG_CONTEXT_TEMPLATE.format(
        context=context_str, question=user_message
    )

    # Step 3: Build messages
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": augmented_message})

    tools_called = []

    # Step 4: Iterative tool-calling loop
    for _ in range(max_tool_rounds):
        response = chat_completion(messages=messages, tools=TOOL_SCHEMAS)

        # If no tool calls, we have the final answer
        if not response.tool_calls:
            return {
                "response": response.content or "",
                "citations": citations,
                "tools_called": tools_called,
            }

        # Process tool calls — content can be None when tools are invoked
        assistant_msg = {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in response.tool_calls
            ],
        }
        if response.content:
            assistant_msg["content"] = response.content
        messages.append(assistant_msg)

        for tool_call in response.tool_calls:
            fn_name = tool_call.function.name
            try:
                fn_args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                fn_args = {}

            tool_result = _execute_tool(fn_name, fn_args)

            tools_called.append({
                "tool": fn_name,
                "arguments": fn_args,
                "result_preview": tool_result[:500],
            })

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_result,
            })

    # Final completion after tool rounds
    response = chat_completion(messages=messages)
    return {
        "response": response.content or "",
        "citations": citations,
        "tools_called": tools_called,
    }
