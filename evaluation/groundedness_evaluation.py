import re
import unicodedata
import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

from chatbot_core import retrieve_hybrid, embeddings


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

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def get_words(text):
    text = normalize_text(text)

    stop_words = {
        "si",
        "sau",
        "este",
        "sunt",
        "un",
        "o",
        "in",
        "din",
        "de",
        "la",
        "cu",
        "pe",
        "pentru",
        "ca",
        "care",
        "se",
        "prin",
        "acest",
        "aceasta",
        "aceste",
        "ale",
        "al",
        "ai",
        "a",
        "fi",
        "fie",
        "mai",
        "poate",
        "pot",
        "unei",
        "unui",
        "the",
        "and",
        "or",
        "is",
        "are",
        "of",
        "to",
        "for",
        "with",
        "by"
    }

    words = {
        word
        for word in text.split()
        if len(word) >= 3
        and word not in stop_words
    }

    return words


def sentence_groundedness(
    sentence,
    context
):
    sentence_words = get_words(sentence)
    context_words = get_words(context)

    if not sentence_words:
        return 1.0

    supported_words = (
        sentence_words & context_words
    )

    return (
        len(supported_words)
        / len(sentence_words)
    )

def semantic_groundedness(
    sentence,
    context
):
    sentence_vector = embeddings.embed_query(
        sentence
    )

    context_vector = embeddings.embed_query(
        context
    )

    dot_product = sum(
        a * b
        for a, b in zip(
            sentence_vector,
            context_vector
        )
    )

    sentence_norm = sum(
        value ** 2
        for value in sentence_vector
    ) ** 0.5

    context_norm = sum(
        value ** 2
        for value in context_vector
    ) ** 0.5

    if sentence_norm == 0 or context_norm == 0:
        return 0.0

    similarity = (
        dot_product
        / (sentence_norm * context_norm)
    )

    return similarity
def find_best_context_sentence(
    sentence,
    context
):
    context_sentences = re.split(
        r"(?<=[.!?])\s+",
        context.strip()
    )

    context_sentences = [
        item.strip()
        for item in context_sentences
        if item.strip()
    ]

    if not context_sentences:
        return "", 0.0

    best_sentence = ""
    best_score = -1.0

    for context_sentence in context_sentences:

        score = semantic_groundedness(
            sentence,
            context_sentence
        )

        if score > best_score:
            best_score = score
            best_sentence = context_sentence

    return best_sentence, best_score

def has_negation_mismatch(
    sentence,
    context
):
    negation_words = {
        "nu",
        "nici",
        "niciodata",
        "never",
        "not",
        "no"
    }

    sentence_words = set(
        normalize_text(sentence).split()
    )

    context_words = set(
        normalize_text(context).split()
    )

    sentence_has_negation = bool(
        sentence_words & negation_words
    )

    if not sentence_has_negation:
        return False

    context_has_negation = bool(
        context_words & negation_words
    )

    return not context_has_negation

def combined_groundedness(
    sentence,
    context,
    lexical_threshold=0.5,
    semantic_threshold=0.75
):
    best_context_sentence, semantic_score = (
        find_best_context_sentence(
            sentence,
            context
        )
    )

    lexical_score = sentence_groundedness(
        sentence,
        best_context_sentence
    )

    negation_mismatch = has_negation_mismatch(
        sentence,
        best_context_sentence
    )

    grounded = (
        not negation_mismatch
        and (
            lexical_score >= lexical_threshold
            or semantic_score >= semantic_threshold
        )
    )

    return {
        "lexical_score": lexical_score,
        "semantic_score": semantic_score,
        "negation_mismatch": negation_mismatch,
        "best_context_sentence": best_context_sentence,
        "grounded": grounded
    }

def evaluate_groundedness(
    answer,
    context,
    lexical_threshold=0.5,
    semantic_threshold=0.75
):
    sentences = re.split(
        r"(?<=[.!?])\s+",
        answer.strip()
    )

    sentences = [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]

    results = []

    for sentence in sentences:

        evaluation = combined_groundedness(
            sentence,
            context,
            lexical_threshold=lexical_threshold,
            semantic_threshold=semantic_threshold
        )

        results.append(
            {
                "sentence": sentence,
                "lexical_score": evaluation[
                    "lexical_score"
                ],
                "semantic_score": evaluation[
                    "semantic_score"
                ],
                "negation_mismatch": evaluation[
                    "negation_mismatch"
                ],
                "best_context_sentence": evaluation[
                    "best_context_sentence"
                ],
                "grounded": evaluation[
                    "grounded"
                ]
            }
        )
    if not results:
        return {
            "score": 0.0,
            "lexical_score": 0.0,
            "semantic_score": 0.0,
            "grounded": False,
            "sentences": []
        }

    overall_lexical_score = sum(
        item["lexical_score"]
        for item in results
    ) / len(results)

    overall_semantic_score = sum(
        item["semantic_score"]
        for item in results
    ) / len(results)

    overall_score = (
        overall_lexical_score
        + overall_semantic_score
    ) / 2

    return {
        "score": overall_score,
        "lexical_score": overall_lexical_score,
        "semantic_score": overall_semantic_score,
        "grounded": all(
            item["grounded"]
            for item in results
        ),
        "sentences": results
    }

if __name__ == "__main__":

    question = (
        "Ce este mutatia in algoritmii genetici?"
    )

    documents = retrieve_hybrid(
        question
    )[:5]

    context = "\n\n".join(
        doc.page_content
        for doc in documents
    )

    print(
        f"Documente recuperate: "
        f"{len(documents)}"
    )

    print()

    for i, doc in enumerate(
        documents,
        start=1
    ):
        print(
            f"Document {i}: "
            f"{doc.metadata.get('source_file', 'unknown')} "
            f"| pagina "
            f"{doc.metadata.get('page', 'unknown')}"
        )

        print(
            doc.page_content[:500]
        )

        print(
            "-" * 80
        )

    answer = (
        "Mutatia este un operator genetic."
    )

    result = evaluate_groundedness(
        answer,
        context
    )

    print(
        f"Groundedness score: "
        f"{result['score']:.2%}"
    )

    print(
        f"Grounded: "
        f"{result['grounded']}"
    )

    print()

    for item in result["sentences"]:

        print(
            f"Sentence: {item['sentence']}"
        )
        print(
            f"Best context sentence: "
            f"{item['best_context_sentence']}"
        )

        print(
            f"Lexical score: "
            f"{item['lexical_score']:.2%}"
        )

        print(
            f"Semantic score: "
            f"{item['semantic_score']:.2%}"
        )

        print(
            f"Negation mismatch: "
            f"{item['negation_mismatch']}"
        )

        print(
            f"Grounded: {item['grounded']}"
        )

        print()

    print(
        "TEST SEMANTIC"
    )

    semantic_context = (
        "Mutatia contribuie la diversitatea populatiei."
    )

    semantic_answer = (
        "Mutatia ajuta la mentinerea varietatii indivizilor."
    )

    semantic_score = semantic_groundedness(
        semantic_answer,
        semantic_context
    )

    print(
        f"Semantic similarity: "
        f"{semantic_score:.2%}"
    )

    negation_mismatch = has_negation_mismatch(
        semantic_answer,
        semantic_context
    )

    print(
        f"Negation mismatch: "
        f"{negation_mismatch}"
    )

    print()

    print(
        "TEST COMBINED"
    )

    combined_result = combined_groundedness(
        semantic_answer,
        semantic_context
    )

    print(
        f"Lexical score: "
        f"{combined_result['lexical_score']:.2%}"
    )

    print(
        f"Semantic score: "
        f"{combined_result['semantic_score']:.2%}"
    )

    print(
        f"Negation mismatch: "
        f"{combined_result['negation_mismatch']}"
    )

    print(
        f"Grounded: "
        f"{combined_result['grounded']}"
    )
    print()

    print(
        "TEST BEST CONTEXT SENTENCE"
    )

    best_sentence, best_score = (
        find_best_context_sentence(
            semantic_answer,
            semantic_context
        )
    )

    print(
        f"Answer: {semantic_answer}"
    )

    print(
        f"Best context sentence: "
        f"{best_sentence}"
    )

    print(
        f"Best semantic score: "
        f"{best_score:.2%}"
    )