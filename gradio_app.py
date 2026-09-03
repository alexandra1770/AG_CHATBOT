import re

import gradio as gr
import requests


API_URL = (
    "http://127.0.0.1:8000/ask"
)


def clean_answer_for_gradio(text):

    if not text:
        return text

    # transforma ```math ... ``` in $$ ... $$
    text = re.sub(
        r"```math\s*(.*?)```",
        lambda match: (
            "\n$$\n"
            + match.group(1).strip()
            + "\n$$\n"
        ),
        text,
        flags=(
            re.DOTALL
            | re.IGNORECASE
        )
    )

    # transforma \[ ... \] in $$ ... $$
    text = re.sub(
        r"\\\[(.*?)\\\]",
        lambda match: (
            "\n$$\n"
            + match.group(1).strip()
            + "\n$$\n"
        ),
        text,
        flags=re.DOTALL
    )

    # transforma \( ... \) in $ ... $
    text = re.sub(
        r"\\\((.*?)\\\)",
        lambda match: (
            "$"
            + match.group(1).strip()
            + "$"
        ),
        text,
        flags=re.DOTALL
    )

    cleaned_lines = []

    for line in text.splitlines():

        stripped = (
            line.strip()
            .lower()
        )

        if stripped in {
            "svg",
            "svgsvg"
        }:
            continue

        cleaned_lines.append(
            line
        )

    text = "\n".join(
        cleaned_lines
    )

    text = re.sub(
        r"\n{4,}",
        "\n\n\n",
        text
    )

    return text.strip()


def chat_with_bot(
    message,
    history
):

    if not message.strip():

        return (
            "Please enter a question."
        )

    try:

        response = requests.post(
            API_URL,
            json={
                "question": message,
                "history": history
            },
            timeout=120
        )

        response.raise_for_status()

        data = response.json()

        answer = data.get(
            "answer",
            "I could not generate an answer."
        )

        answer = clean_answer_for_gradio(
            answer
        )

        return answer

    except requests.exceptions.ConnectionError:

        return (
            "I cannot connect to the chatbot API. "
            "Please make sure the FastAPI server "
            "is running."
        )

    except requests.exceptions.Timeout:

        return (
            "The request took too long. "
            "Please try again."
        )

    except requests.exceptions.RequestException as e:

        return (
            f"An error occurred: {e}"
        )


welcome_message = """
# 🧬 Genetic Algorithms Assistant

Hello! I can help you understand and work with
**Genetic Algorithms** using your course and laboratory
materials.

You can ask questions in **Romanian or English**.

### I can help you with:

- Genetic Algorithms concepts
- Selection methods
- Crossover
- Mutation
- Fitness functions
- Chromosomes and populations
- Optimization problems
- Course and laboratory exercises
- Mathematical formulas
- Python examples

You can also continue the conversation naturally.

For example:

**"Explica-mi mai simplu."**

**"Da-mi un exemplu."**

**"Explain it more simply."**

**"Create a problem and give me the Python code."**

### How can I help you today?
"""


with gr.Blocks(
    title="Genetic Algorithms Assistant"
) as demo:

    gr.Markdown(
        welcome_message
    )

    chatbot = gr.Chatbot(
        height=550,

        placeholder=(
            "<strong>"
            "🧬 Genetic Algorithms Assistant"
            "</strong><br>"
            "Ask me anything about "
            "Genetic Algorithms!"
        ),

        render_markdown=True,

        latex_delimiters=[
            {
                "left": "$$",
                "right": "$$",
                "display": True
            },
            {
                "left": "$",
                "right": "$",
                "display": False
            }
        ]
    )

    with gr.Row():

        msg = gr.Textbox(
            placeholder=(
                "Type your question here..."
            ),
            show_label=False,
            scale=8,
            lines=2
        )

        send = gr.Button(
            "Send",
            variant="primary",
            scale=1
        )

    gr.Markdown(
        """
**Examples:**  
*What is crossover?* ·
*Care este rolul mutatiei?* ·
*Explain tournament selection.* ·
*Creeaza-mi o problema si da-mi codul Python.*
"""
    )

    clear = gr.Button(
        "Clear conversation"
    )


    def respond(
        message,
        history
    ):

        if not message.strip():

            return (
                "",
                history
            )

        if history is None:
            history = []

        history_for_api = (
            history.copy()
        )

        answer = chat_with_bot(
            message,
            history_for_api
        )

        new_history = (
            history
            + [
                {
                    "role": "user",
                    "content": message
                },
                {
                    "role": "assistant",
                    "content": answer
                }
            ]
        )

        return (
            "",
            new_history
        )


    msg.submit(
        respond,
        inputs=[
            msg,
            chatbot
        ],
        outputs=[
            msg,
            chatbot
        ]
    )


    send.click(
        respond,
        inputs=[
            msg,
            chatbot
        ],
        outputs=[
            msg,
            chatbot
        ]
    )


    clear.click(
        lambda: (
            [],
            ""
        ),
        outputs=[
            chatbot,
            msg
        ]
    )


if __name__ == "__main__":

    demo.launch()