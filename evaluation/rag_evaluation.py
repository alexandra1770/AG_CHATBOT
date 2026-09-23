from chatbot_core import retrieve_final


evaluation_cases = [
    {
        "question": "Ce este mutatia in algoritmii genetici?",
        "expected_source": "Cap05.pdf"
    },
    {
        "question": "Ce este selectia prin ruleta in algoritmii genetici?",
        "expected_source": "Cap05.pdf"
    },
    {
        "question": "Ce este incrucisarea in algoritmii genetici?",
        "expected_source": "Cap08.pdf"
    },
    {
        "question": "Ce este functia de adecvare fitness in algoritmii genetici?",
        "expected_source": "Cap01.pdf"
    },
    {
        "question": "Ce este codificarea spatiului de cautare in algoritmii genetici?",
        "expected_source": "Cap02.pdf"
    }
]


passed = 0
top1_correct = 0
reciprocal_ranks = []


for case in evaluation_cases:

    question = case["question"]
    expected_source = case["expected_source"]

    documents = retrieve_final(
        question,
        k_candidates=10,
        k_final=5
    )

    retrieved_sources = [
        doc.metadata.get("source_file", "unknown")
        for doc in documents
    ]

    success = expected_source in retrieved_sources

    if success:
        passed += 1

    if retrieved_sources and retrieved_sources[0] == expected_source:
        top1_correct += 1

    rank = None

    for i, source in enumerate(retrieved_sources, start=1):
        if source == expected_source:
            rank = i
            break

    if rank is not None:
        reciprocal_ranks.append(1 / rank)
    else:
        reciprocal_ranks.append(0)

    print("\n" + "=" * 70)
    print("Intrebare:", question)
    print("Sursa asteptata:", expected_source)
    print("Surse recuperate:", retrieved_sources)
    print("Pozitia sursei corecte:", rank if rank else "NU A FOST GASITA")
    print("Rezultat:", "PASS" if success else "FAIL")


total = len(evaluation_cases)

recall_at_5 = passed / total
top1_accuracy = top1_correct / total
mrr = sum(reciprocal_ranks) / total


print("\n" + "=" * 70)
print("REZULTAT FINAL")
print(f"Cazuri evaluate: {total}")
print(f"Cazuri corecte in Top 5: {passed}/{total}")
print(f"Recall@5: {recall_at_5:.2%}")
print(f"Top-1 Accuracy: {top1_accuracy:.2%}")
print(f"MRR: {mrr:.4f}")