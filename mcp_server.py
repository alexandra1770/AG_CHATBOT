from mcp.server.mcpserver import MCPServer

mcp = MCPServer("AG-Chatbot")


@mcp.tool()
def search_documents(query: str) -> str:
    """Search the Genetic Algorithms course and laboratory documents."""

    
    import chatbot_core

    return chatbot_core.search_documents.invoke(
        {"query": query}
    )


@mcp.tool()
def calculator(expression: str) -> str:
    """Calculate a mathematical expression."""

    try:
        allowed_characters = set(
            "0123456789+-*/(). %"
        )

        if not all(
            char in allowed_characters
            for char in expression
        ):
            return "Invalid mathematical expression."

        result = eval(
            expression,
            {"__builtins__": {}},
            {}
        )

        return str(result)

    except Exception as e:
        return f"Calculation error: {e}"


if __name__ == "__main__":
    mcp.run()