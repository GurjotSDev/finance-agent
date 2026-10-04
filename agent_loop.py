"""
agent_loop.py

Chat with a local model that can call tools from several MCP servers.
Each user message can trigger several rounds of tool calls before the 
model gives its final answer. The conversation history is kept betwwen
messages, so follow-up questions work.
"""

import asyncio
import json
import sys
from contextlib import AsyncExitStack
from pathlib import Path
import ollama
from mcp import ClientSession, MCPError, StdioServerParameters
from mcp.client.stdio import stdio_client

MODEL = "gpt-oss:20b"
NUM_CTX = 16384 # Context window: ~25 tool schemas + chat history.
MAX_ROUNDS = 6

PROJECT_DIR = Path(__file__).resolve().parent

SERVERS = [
    {
        "name": "alpaca",
        "params": StdioServerParameters(
            command="uvx",
            args=["alpaca-mcp-server", "--env-file", str(PROJECT_DIR / ".env")],
        ),
    },
    {
        "name": "screener",
        "params": StdioServerParameters(
            command=sys.executable,
            args=[str(PROJECT_DIR / "screener_server.py")],
        ),
    },
]

SYSTEM_PROMPT = (
    "You are a stock researh assistant. You get real market data only through "
    "the provided tools; you have no built-in knowledge of current prices, dates, "
    "or financial data. This account uses Alpaca's free tier: IEX exchange data "
    "only, not the SIP consolidated feed.\n\n"
    "Rules:\n"
    "1. Ever number, date, or fact you state must come from a tool result in this "
    "conversation. Copy numbers exactly; do not recompute or round them.\n"
    "2. For questions about what to invest in or how to split money, call "
    "screen_stocks. Map risk words to max_volatility: 'low risk' = 0.30, "
    "'moderate' = 0.45. If no risk level is given, omit it.\n"
    "3. When presenting picks, show each stock's name, dollars, momentum_pct, and "
    "volatility_pct, and mention anything notable the screener excluded.\n"
    "4. If a tool result doesn't contain what you need, or returns ok: false, say "
    "so. Never guess or fill in plausible values.\n"
    "5. Do not describe data sources (feed, exchange, latency) beyond what the "
    "tool output states.\n"
    "6. You cannot place trades. When recommending stocks, briefly note that past "
    "momentum does not guarantee future returns.\n"
    "7. Use tool defaults for anything the user didn't specify. Only ask a "
    "follow-up question if a required value, like the budget, is missing.\n"

)


def to_ollama_tool(tool):
    """Convert an MCP tool into the schema Ollama expects."""
    return{
        "type": "function",
        "function": {
            "name": tool.name,
            "description": (tool.description or "").strip(),
            "parameters": tool.input_schema,
        },
    }

async def ask_model(client: ollama.AsyncClient, messages: list, tools:list):
    """The ONLY place the model is called. to try another model or provider, 
    change this funtion and nothing else"""
    response = await client.chat(
        model=MODEL,
        messages=messages,
        tools=tools,
        options={"num_ctx": NUM_CTX},
    )
    return response.message

async def run_tool(tool_to_session: dict, name:str, args:dict) -> str:
    """Route a tool call to the server that owns it. Always returns text
    for the model, even on failure, so one bad call cant crash the chat"""
    session = tool_to_session.get(name)
    if session is None:
        return json.dumps({"ok": False, "error": f"Unknown tool: {name}"})

    try:
        result = await session.call_tool(name, arguments=args)
    except MCPError as e:
        return json.dumps({"ok": False, "error": f"{name} failed: {e.message}"})

    text = "".join(block.text for block in result.content if block.type == "text")
    if result.is_error:
        return json.dumps({"ok": False, "error": f"{name} failed"})
    return text

async def answer(client, messages: list, tools: list, tool_to_session:dict) -> str:
    """Let the model call tools until it gives a final answer"""
    for round_num in range(1, MAX_ROUNDS + 1):
        msg = await ask_model(client, messages, tools)
        messages.append(msg)

        if not msg.tool_calls:
            return msg.content

        for call in msg.tool_calls:
            name = call.function.name
            args = call.function.arguments or {}
            print(f"  >> round {round_num}: {name}({args})")

            result_text = await run_tool(tool_to_session, name, args)
            preview = result_text[:300] + ("..." if len(result_text) > 300 else "")
            print(f"    result: {preview}")

            messages.append({
                "role": "tool",
                "tool_name": name,
                "content": result_text
            })
    return "(Stopped: too many tool calls without a final answer.)"

async def main() -> None:
    client = ollama.AsyncClient()

    async with AsyncExitStack() as stack:
        tool_to_session = {}
        tools = []

        for server in SERVERS:
            read, write = await stack.enter_async_context(stdio_client(server["params"]))
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()

            server_tools = (await session.list_tools()).tools
            for tool in server_tools:
                if tool.name in tool_to_session:
                    raise RuntimeError(f"Duplicate tool name {tool.name!r} from {server['name']}")
                tool_to_session[tool.name] = session
                tools.append(to_ollama_tool(tool))
            print(f"Connected to {server['name']}: {len(server_tools)} tools")

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            }
        ]
        print("\nAsk a question (type 'quit' to exit).")

        while True:
            user_text = (await asyncio.to_thread(input, "\nYou: ")).strip()
            if user_text.lower() in {"quit", "exit"}:
                break
            if not user_text:
                continue

            messages.append(
                {
                    "role": "user",
                    "content": user_text
                }
            )
            reply = await answer(client, messages, tools, tool_to_session)
            print(f"\nAssistant: {reply}")

asyncio.run(main())