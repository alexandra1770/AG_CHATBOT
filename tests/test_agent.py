from unittest.mock import patch

from langchain_core.messages import AIMessage

from chatbot_core import answer_with_agent


def test_agent_uses_calculator():

    responses = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "calculator",
                    "args": {
                        "expression": "25 * 17"
                    },
                    "id": "call_calculator"
                }
            ]
        ),
        AIMessage(
            content="Rezultatul este 425."
        )
    ]

    with patch(
        "chatbot_core.llm"
    ) as mock_llm, patch(
        "chatbot_core.call_mcp_tool"
    ) as mock_mcp:

        mock_agent_llm = (
            mock_llm.bind_tools.return_value
        )

        mock_agent_llm.invoke.side_effect = (
            responses
        )

        async def fake_mcp_call(
            tool_name,
            arguments
        ):
            if tool_name == "calculator":
                return "425"

            return ""

        mock_mcp.side_effect = fake_mcp_call

        result = answer_with_agent(
            "Calculeaza exact 25 * 17."
        )

        assert "425" in result

        assert any(
            call.args[0] == "calculator"
            for call in mock_mcp.call_args_list
        )


def test_agent_uses_search_documents():

    responses = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search_documents",
                    "args": {
                        "query": "mutatia algoritmi genetici"
                    },
                    "id": "call_search"
                }
            ]
        ),
        AIMessage(
            content=(
                "Mutatia este un operator genetic. "
                "Sursa: test.pdf, pagina 5."
            )
        )
    ]

    with patch(
        "chatbot_core.llm"
    ) as mock_llm, patch(
        "chatbot_core.call_mcp_tool"
    ) as mock_mcp:

        mock_agent_llm = (
            mock_llm.bind_tools.return_value
        )

        mock_agent_llm.invoke.side_effect = (
            responses
        )

        async def fake_mcp_call(
            tool_name,
            arguments
        ):
            if tool_name == "search_documents":
                return (
                    "[Document 1]\n"
                    "Source: test.pdf\n"
                    "Page: 5\n"
                    "Content:\n"
                    "Mutatia este un operator genetic."
                )

            return ""

        mock_mcp.side_effect = fake_mcp_call

        result = answer_with_agent(
            "Ce este mutatia in algoritmii genetici?"
        )

        assert "mut" in result.lower()

        assert any(
            call.args[0] == "search_documents"
            for call in mock_mcp.call_args_list
        )


def test_agent_uses_multiple_tools():

    responses = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search_documents",
                    "args": {
                        "query": "mutatia algoritmi genetici"
                    },
                    "id": "call_search"
                },
                {
                    "name": "calculator",
                    "args": {
                        "expression": "25 * 17"
                    },
                    "id": "call_calculator"
                }
            ]
        ),
        AIMessage(
            content=(
                "Mutatia este un operator genetic. "
                "Rezultatul calculului este 425."
            )
        )
    ]

    with patch(
        "chatbot_core.llm"
    ) as mock_llm, patch(
        "chatbot_core.call_mcp_tool"
    ) as mock_mcp:

        mock_agent_llm = (
            mock_llm.bind_tools.return_value
        )

        mock_agent_llm.invoke.side_effect = (
            responses
        )

        async def fake_mcp_call(
            tool_name,
            arguments
        ):
            if tool_name == "search_documents":
                return (
                    "[Document 1]\n"
                    "Source: test.pdf\n"
                    "Page: 5\n"
                    "Content:\n"
                    "Mutatia este un operator genetic."
                )

            if tool_name == "calculator":
                return "425"

            return ""

        mock_mcp.side_effect = fake_mcp_call

        result = answer_with_agent(
            "Cauta ce se spune despre mutatie, "
            "apoi calculeaza exact 25 * 17."
        )

        tool_names = [
            call.args[0]
            for call in mock_mcp.call_args_list
        ]

        assert "search_documents" in tool_names
        assert "calculator" in tool_names
        assert "425" in result