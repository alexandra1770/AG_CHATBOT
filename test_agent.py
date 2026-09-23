from langchain_core.messages import HumanMessage, ToolMessage

import chatbot_core


agent_llm = chatbot_core.llm.bind_tools(
    [chatbot_core.search_documents]
)


def run_agent(question):

    messages = [
        HumanMessage(content=question)
    ]

    response = agent_llm.invoke(messages)

    messages.append(response)

    print("\nTOOL CALLS:")
    print(response.tool_calls)

    for tool_call in response.tool_calls:

        if tool_call["name"] == "search_documents":

            tool_result = chatbot_core.search_documents.invoke(
                tool_call["args"]
            )

            messages.append(
                ToolMessage(
                    content=tool_result,
                    tool_call_id=tool_call["id"]
                )
            )

    if response.tool_calls:
        final_response = agent_llm.invoke(messages)
        return final_response.content

    return response.content


question = "Foloseste documentele disponibile pentru a-mi spune ce este mutatia in algoritmii genetici."

answer = run_agent(question)

print("\nRASPUNS FINAL:")
print(answer)