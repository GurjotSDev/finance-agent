import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

server_params = StdioServerParameters(
    command="uvx",
    args=["alpaca-mcp-server", "--env-file", ".env"],
)

async def main():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            print(f"=== {len(tools.tools)} tools awailable ===")
            for tool in tools.tools:
                print(f"- {tool.name}")

asyncio.run(main())