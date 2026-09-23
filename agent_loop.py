import asyncio
import ollama
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

MODEL = "gpt-oss:20b"

SERVERS = [
    {
        "name": "alpaca",
        "params": StdioServerParameters(
            command="uvx",
            args=["alpaca-mcp-server", "--env-file", ".env"],
        ),
    },
]



def mcp_tool_to_ollama_format(tool):
    """Convert an MCP tool object into the schema Ollama expects."""
    return{
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description.strip(),
            "parameters": tool.input_schema,
        },
    }

async def main():
    async with AsyncExitStack() as stack:
        tool_to_session = {}
        ollama_tools = []

        for server in SERVERS:
            read, write = await stack.enter_async_context(stdio_client(server["params"]))
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()

            mcp_tools = (await session.list_tools()).tools
            for tool in mcp_tools:
                tool_to_session[tool.name] = session
                ollama_tools.append(mcp_tool_to_ollama_format(tool))

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a financial data assistant. You only have access to real data "
                        "through the provided tools. You have no built in knowledge of current "
                        "prices, dates, or financial data. This account uses Alpaca's free tier, "
                        "which provides IEX exchange data only, not the full SIP consolidated feed. "
                        "Rules:\n"
                        "1. Never state a number, date, or fact unless it appears verbatim in a tool result.\n"
                        "2. If the user's question has multiple parts, call every tool needed to answer "
                        "all parts before giving your final answer.\n"
                        "3. If a tool result doesn't contain what you need, say so. Do not guess or fill in plausible values.\n"
                        "4. Do not describe the technical source of data, such as feed name, exchange, "
                        "or latency, unless that detail is explicitly present in the tool output."
                    )
                },
                {"role": "user", "content": "What are today's biggest stock market movers?"},
            ]
            max_rounds = 5
            for round_num in range(max_rounds):
                response = ollama.chat(
                    model=MODEL,
                    messages=messages,
                    tools=ollama_tools,
                    options={"num_ctx": 8192},
                )
                msg = response["message"]
                messages.append(msg)

                if not msg.get("tool_calls"):
                    # Model gave a real answer with no tool calls, we're done
                    print("\n=== Final Answer ===")
                    print(msg["content"]) 
                    break

                for call in msg["tool_calls"]:
                    tool_name = call["function"]["name"]
                    tool_args = call["function"]["arguments"]
                    print(f">> Round {round_num + 1}: calling {tool_name}({tool_args})")

                    result = await session.call_tool(tool_name, arguments=tool_args)
                    result_text = "".join(
                        block.text for block in result.content if block.type == "text"
                    )
                    print(f"      [FULL raw tool output]: {result_text}")

                    messages.append({
                        "role": "tool",
                        "content": result_text,
                    })
            else:
                print("\n=== Max rounds reached without a final answer ===")
                

asyncio.run(main())