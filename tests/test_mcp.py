import os
import sys

import pytest
from mcp import Client, StdioServerParameters


@pytest.mark.anyio
async def test_mcp_calculator():

    server_params = StdioServerParameters(
        command=sys.executable,
        args=["mcp_server.py"],
        env=os.environ.copy()
    )

    async with Client(server_params) as client:

        result = await client.call_tool(
            "calculator",
            {"expression": "25 * 17"}
        )

        assert result.is_error is False
        assert result.structured_content["result"] == "425"

@pytest.mark.anyio
async def test_mcp_lists_tools():

    server_params = StdioServerParameters(
        command=sys.executable,
        args=["mcp_server.py"],
        env=os.environ.copy()
    )

    async with Client(server_params) as client:

        tools = await client.list_tools()

        tool_names = [
            tool.name
            for tool in tools.tools
        ]

        assert "calculator" in tool_names
        assert "search_documents" in tool_names

@pytest.mark.anyio
async def test_mcp_search_documents():

    server_params = StdioServerParameters(
        command=sys.executable,
        args=["mcp_server.py"],
        env=os.environ.copy()
    )

    async with Client(server_params) as client:

        result = await client.call_tool(
            "search_documents",
            {
                "query": "Ce este mutatia in algoritmii genetici?"
            }
        )

        assert result.is_error is False
        assert result.structured_content is not None
        assert "result" in result.structured_content

        text = str(
            result.structured_content["result"]
        )

        assert len(text) > 0
        assert "Document" in text