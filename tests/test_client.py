import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

server_params = StdioServerParameters(
    command="python",
    args=["market_server.py"],
)

async def main():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # 1. Ask the server what tools it offers
            tools = await session.list_tools()
            print("=== Available tools ===")
            for tool in tools.tools:
                print(f"- {tool.name}: {tool.description.strip().splitlines()[0]}")

            # 2. Manually call one tool, mimic the LLM
            print("\n=== Calling get_price_history(AAPL, 5d) ===")
            result = await session.call_tool(
                "get_price_history",
                arguments={"ticker": "AAPL", "period": "5d"},
            )

            for block in result.content:
                if block.type == "text":
                    print(block.text)

asyncio.run(main())