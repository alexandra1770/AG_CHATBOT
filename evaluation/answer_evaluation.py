import unicodedata

from chatbot_core import answer_with_agent


def normalize_text(text):
    text = text.lower()

    text = unicodedata.normalize(
        "NFD",
        text
    )

    text = "".join(
        char
        for char in text
        if unicodedata.category(char) != "Mn"
    )

    return text


def is_service_error(answer_text):
    text = normalize_text(answer_text)

    error_patterns = [
        "limita temporara a serviciului ai",
        "limita api",
        "rate limit",
        "rate_limit",
        "request too large",
        "eroare la generarea raspunsului",
        "internal mcp error",
        "mcp tool error",
        "could not be executed",
        "agentul nu a generat un raspuns final",
        "agentul nu a reusit sa finalizeze raspunsul"
    ]

    return any(
        pattern in text
        for pattern in error_patterns
    )


evaluation_cases = [
    {
        "question": "Ce este mutatia in algoritmii genetici?",
        "expected_source": "Cap05.pdf",
        "keywords": [
            "mutat"
        ]
    },
    {
        "question": "Ce este selectia prin ruleta in algoritmii genetici?",
        "expected_source": "Cap05.pdf",
        "keywords": [
            "selectie",
            "ruleta"
        ]
    },
    {
        "question": "Ce este incrucisarea in algoritmii genetici?",
        "expected_source": "Cap08.pdf",
        "keywords": [
            "incrucis"
        ]
    },
    {
        "question": "Ce este functia de adecvare fitness in algoritmii genetici?",
        "expected_source": "Cap01.pdf",
        "keywords": [
            "fitness",
            "adecv"
        ]
    },
    {
        "question": "Ce este codificarea spatiului de cautare in algoritmii genetici?",
        "expected_source": "Cap02.pdf",
        "keywords": [
            "codific"
        ]
    }
]


total = len(evaluation_cases)

successful_tests = 0
error_tests = 0

non_empty_answers = 0
relevant_answers = 0
correct_sources = 0

results = []


for index, case in enumerate(
    evaluation_cases,
    start=1
):

    question = case["question"]
    expected_source = case["expected_source"]
    keywords = case["keywords"]

    print("\n" + "=" * 70)
    print(f"TEST {index}/{total}")
    print("Intrebare:", question)

    try:
        answer = answer_with_agent(
            question
        )

    except Exception as e:
        print("\nEROARE TEHNICA:")
        print(e)

        error_tests += 1

        results.append({
            "question": question,
            "expected_source": expected_source,
            "status": "ERROR"
        })

        print("\nRezultat: ERROR")

        continue

    answer_text = str(answer)

    print("\nRaspuns:")
    print(answer_text)

    if is_service_error(answer_text):

        print("\n--- EVALUARE ---")
        print("Eroare tehnica detectata: YES")
        print("Rezultat: ERROR / SKIP")

        error_tests += 1

        results.append({
            "question": question,
            "expected_source": expected_source,
            "status": "ERROR"
        })

        continue

    answer_normalized = normalize_text(
        answer_text
    )

    has_answer = (
        len(answer_text.strip()) > 0
    )

    keyword_found = any(
        normalize_text(keyword)
        in answer_normalized
        for keyword in keywords
    )

    expected_source_name = (
        expected_source.replace(
            ".pdf",
            ""
        )
    )

    source_found = (
        normalize_text(
            expected_source_name
        )
        in answer_normalized
    )

    if has_answer:
        non_empty_answers += 1

    if keyword_found:
        relevant_answers += 1

    if source_found:
        correct_sources += 1

    if (
        has_answer
        and keyword_found
        and source_found
    ):
        status = "PASS"
        successful_tests += 1

    else:
        status = "FAIL"

    results.append({
        "question": question,
        "expected_source": expected_source,
        "has_answer": has_answer,
        "keyword_found": keyword_found,
        "source_found": source_found,
        "status": status
    })

    print("\n--- EVALUARE ---")

    print(
        "Raspuns generat:",
        "PASS" if has_answer else "FAIL"
    )

    print(
        "Relevanta de baza:",
        "PASS" if keyword_found else "FAIL"
    )

    print(
        "Sursa asteptata mentionata:",
        "PASS" if source_found else "FAIL"
    )

    print(
        "Rezultat:",
        status
    )


valid_tests = (
    total - error_tests
)


if valid_tests > 0:

    answer_rate = (
        non_empty_answers
        / valid_tests
    )

    relevance_rate = (
        relevant_answers
        / valid_tests
    )

    citation_rate = (
        correct_sources
        / valid_tests
    )

    pass_rate = (
        successful_tests
        / valid_tests
    )

else:

    answer_rate = 0
    relevance_rate = 0
    citation_rate = 0
    pass_rate = 0


print("\n" + "=" * 70)
print("REZULTAT FINAL - ANSWER EVALUATION")

print(
    f"Cazuri totale: {total}"
)

print(
    f"Cazuri valide: {valid_tests}"
)

print(
    f"Erori tehnice / SKIP: "
    f"{error_tests}"
)

print(
    f"Cazuri PASS: "
    f"{successful_tests}/{valid_tests}"
    if valid_tests > 0
    else "Cazuri PASS: N/A"
)

print(
    f"Answer Generation Rate: "
    f"{answer_rate:.2%}"
)

print(
    f"Basic Relevance Rate: "
    f"{relevance_rate:.2%}"
)

print(
    f"Expected Source Citation Rate: "
    f"{citation_rate:.2%}"
)

print(
    f"Overall Pass Rate: "
    f"{pass_rate:.2%}"
)


print("\n" + "=" * 70)
print("REZULTATE INDIVIDUALE")

for index, result in enumerate(
    results,
    start=1
):

    print(
        f"{index}. "
        f"{result['status']} - "
        f"{result['question']}"
    )