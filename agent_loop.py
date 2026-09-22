import asyncio
import json
import ollama
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

server_params = StdioServerParameters(
    command="python",
    args=["market_server.py"],
)

MODEL = "gpt-oss:20b"

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
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            mcp_tools = (await session.list_tools()).tools
            ollama_tools = [mcp_tool_to_ollama_format(tool) for tool in mcp_tools]

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a financial data assistant. You only have access to real data "
                        "through the provided tools you have no built-in knowledge of current "
                        "prices, dates, or financial data. Rules:\n"
                        "1. Never state a number, date, or fact unless it appears verbatim in a tool result.\n"
                        "2. If the user's question has multiple parts, call every tool needed to answer "
                        "all parts before giving your final answer.\n"
                        "3. If a tool result doesn't contain what you need, say so do not guess or fill in plausible values."
                    )
                }, 
                {"role": "user", "content": "What was AAPL's closing price on the last trading day of each month for the past year?"},
            ]

            max_rounds = 5
            for round_num in range(max_rounds):
                # print("DEBUG messages being sent:")
                for m in messages:
                    content = m["content"] if isinstance(m, dict) else m.content
                    # print(f"  role={m['role'] if isinstance(m, dict) else m.role}: {str(content)[:80]}")

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
                    # print("DEBUG raw final message:", msg)
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
                    print(f"   [raw tool output, first 300 chars]: {result_text[:300]}")

                    messages.append({
                        "role": "tool",
                        "content": result_text,
                    })
            else:
                print("\n=== Max rounds reached without a final answer ===")
                

asyncio.run(main())