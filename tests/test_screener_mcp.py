"""
tests.test_screener_mcp.py

Manual MCP client for screener_server.py. No LLM involved: this proves
that tool works over MCP befor wiring it into agent_loop.py
"""

import asyncio
import json
import sys
from pathlib import Path
from mcp import ClientSession, MCPError, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER = Path(__file__).resolve().parent.parent / "screener_server.py"

async def call_and_print(session: ClientSession, arguments: dict) -> None:
    try: 
        result = await session.call_tool("screen_stocks", arguments)
    except MCPError as e:
        print(f"Raised MCPError (code {e.code}): {e.message}")
        return

    print("isERROR:", result.is_error)
    for block in result.content:
        print(block.text)

async def main() -> None:
    params =  StdioServerParameters(command=sys.executable, args=[str(SERVER)])

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # 1. What does the model see?
            tools = await session.list_tools()
            for tool in tools.tools:
                print("TOOL:", tool.name)
                print(json.dumps(tool.input_schema, indent=2))

            # 2. A normal call, like "$1000, low risk, no energy stocks"
            print("\n=== Normal Call ===")
            await call_and_print(
                session,
                {"budget": 1000, "max_volatility": 0.30, "exclude_sectors": ["Energy"]},
            )

            # 3. Bad inputs: these should be rejected before the screener runs
            print("\n=== Negative Budget ===")
            await call_and_print(session, {"budget": -5})

            print("\n=== Invalid Sector Name")
            await call_and_print(session, {"budget": 1000, "exclude_sectors": ["Tech"]})

if __name__ == "__main__":
    asyncio.run(main())
