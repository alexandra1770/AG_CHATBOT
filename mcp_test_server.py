from mcp.server.mcpserver import MCPServer

mcp = MCPServer("Test-Server")


@mcp.tool()
def hello(name: str) -> str:
    """Return a greeting."""

    try:
        import chatbot_core

        return f"Hello, {name}!"

    except Exception as e:
        return f"IMPORT ERROR: {type(e).__name__}: {e}"


if __name__ == "__main__":
    mcp.run()