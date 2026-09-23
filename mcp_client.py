import asyncio
import sys
import os

from mcp import Client, StdioServerParameters


async def main():

    server_params = StdioServerParameters(
        command=sys.executable,
        args=["mcp_server.py"],
        env=os.environ.copy()
    )

    async with Client(server_params) as client:

        tools = await client.list_tools()

        print(
            "MCP TOOLS:",
            [tool.name for tool in tools.tools]
        )

        result = await client.call_tool(
            "search_documents",
            {"query": "Ce este mutatia in algoritmii genetici?"}
        )

        print("SEARCH RESULT:", result)


if __name__ == "__main__":
    asyncio.run(main())