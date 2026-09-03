import os
import re
import json
import pickle
import unicodedata
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq


BASE_PATH = Path(
    r"C:\Users\gm_sl\CHATBOT\AG_Chatbot"
)

CHROMA_FOLDER = (
    BASE_PATH / "chroma_db"
)

EXTRACTED_FOLDER = (
    BASE_PATH / "extracted"
)

CHUNKS_FILE = (
    EXTRACTED_FOLDER / "chunks.pkl"
)


EMBEDDING_MODEL = (
    "sentence-transformers/"
    "paraphrase-multilingual-mpnet-base-v2"
)


if not os.getenv("GROQ_API_KEY"):
    raise ValueError(
        "GROQ_API_KEY is not set. "
        "Set it before starting the FastAPI server."
    )


embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)


vectorstore = Chroma(
    persist_directory=str(
        CHROMA_FOLDER
    ),
    embedding_function=embeddings
)


if not CHUNKS_FILE.exists():
    raise FileNotFoundError(
        f"Chunks file not found: {CHUNKS_FILE}"
    )


with open(CHUNKS_FILE, "rb") as f:
    chunks = pickle.load(f)


llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.2,
    max_tokens=1600
)


STOPWORDS = {
    "ce", "este", "sunt", "care", "cum", "se",
    "un", "o", "unei", "unui", "ale", "al", "a",
    "de", "din", "la", "cu", "si", "sau",
    "pentru", "prin", "in", "intr", "intre",
    "acest", "aceasta", "aceste", "acesti",
    "lui", "lor",

    "the", "is", "are", "what", "how", "why",
    "where", "when", "which", "and", "or", "of",
    "to", "in", "for", "with", "from", "a", "an"
}


def normalize_text(text):

    text = text.lower()

    text = unicodedata.normalize(
        "NFKD",
        text
    )

    text = "".join(
        c
        for c in text
        if not unicodedata.combining(c)
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


def detect_language(text):

    text_norm = normalize_text(
        text
    )

    english_words = {
        "what", "how", "why", "which",
        "where", "when", "explain",
        "give", "create", "tell",
        "mutation", "crossover",
        "selection", "algorithm",
        "algorithms", "genetic",
        "difference", "between",
        "role", "function", "example",
        "simple", "simpler", "does",
        "work", "problem", "code",
        "python", "maximize",
        "minimize", "fitness",
        "answer", "english"
    }

    romanian_words = {
        "ce", "cum", "de", "care",
        "este", "sunt", "explica",
        "da", "creeaza", "spune",
        "mutatie", "incrucisare",
        "selectie", "algoritm",
        "algoritmi", "genetic",
        "diferenta", "dintre",
        "rol", "functie", "exemplu",
        "simplu", "simpla",
        "functioneaza", "problema",
        "cod", "maximizeaza",
        "minimizeaza", "adecvare",
        "raspunde", "romana"
    }

    words = set(
        text_norm.split()
    )

    english_score = len(
        words.intersection(
            english_words
        )
    )

    romanian_score = len(
        words.intersection(
            romanian_words
        )
    )

    if english_score > romanian_score:
        return "English"

    if romanian_score > english_score:
        return "Romanian"

    english_patterns = [
        "what is",
        "what are",
        "how does",
        "how do",
        "explain it",
        "give me",
        "tell me",
        "create a",
        "create me",
        "what role",
        "difference between",
        "answer me in english",
        "in english"
    ]

    for pattern in english_patterns:

        if pattern in text_norm:
            return "English"

    romanian_patterns = [
        "raspunde in romana",
        "in romana",
        "explica mi",
        "da mi",
        "ce este",
        "care este"
    ]

    for pattern in romanian_patterns:

        if pattern in text_norm:
            return "Romanian"

    return "Romanian"


def get_query_words(query_norm):

    return [
        word
        for word in query_norm.split()
        if (
            len(word) > 2
            and word not in STOPWORDS
        )
    ]


def detect_intent(query_norm):

    if any(
        expression in query_norm
        for expression in [
            "care este rolul",
            "ce rol are",
            "rolul",
            "scopul",
            "what is the role",
            "what role",
            "purpose"
        ]
    ):
        return "role"

    if any(
        expression in query_norm
        for expression in [
            "cum functioneaza",
            "cum se aplica",
            "cum se realizeaza",
            "cum se face",
            "cum se executa",
            "how does",
            "how do",
            "how is",
            "how to"
        ]
    ):
        return "process"

    if any(
        expression in query_norm
        for expression in [
            "care este diferenta",
            "diferenta dintre",
            "compara",
            "comparatie",
            "difference between",
            "compare"
        ]
    ):
        return "comparison"

    if (
        query_norm.startswith("ce este ")
        or query_norm.startswith("ce sunt ")
        or query_norm.startswith("ce inseamna ")
        or query_norm.startswith("what is ")
        or query_norm.startswith("what are ")
    ):
        return "definition"

    return "general"


def retrieve_hybrid(
    query,
    k_final=10,
    k_vector=25
):

    query_norm = normalize_text(
        query
    )

    query_words = get_query_words(
        query_norm
    )

    intent = detect_intent(
        query_norm
    )

    candidates = {}

    vector_docs = vectorstore.similarity_search(
        query,
        k=k_vector
    )

    for rank, doc in enumerate(
        vector_docs
    ):

        key = (
            doc.metadata.get(
                "source_file"
            ),
            doc.metadata.get(
                "page"
            ),
            doc.page_content
        )

        candidates[key] = {
            "doc": doc,
            "vector_rank": rank
        }

    for doc in chunks:

        text_norm = normalize_text(
            doc.page_content
        )

        matched_words = sum(
            1
            for word in query_words
            if word in text_norm
        )

        coverage = (
            matched_words /
            len(query_words)
            if query_words
            else 0
        )

        phrase_match = (
            query_norm in text_norm
        )

        if (
            phrase_match
            or coverage >= 0.50
        ):

            key = (
                doc.metadata.get(
                    "source_file"
                ),
                doc.metadata.get(
                    "page"
                ),
                doc.page_content
            )

            if key not in candidates:

                candidates[key] = {
                    "doc": doc,
                    "vector_rank": None
                }

    scored_docs = []

    for candidate in candidates.values():

        doc = candidate["doc"]

        text_norm = normalize_text(
            doc.page_content
        )

        score = 0

        vector_rank = candidate[
            "vector_rank"
        ]

        if vector_rank is not None:

            score += max(
                0,
                80 - vector_rank * 2
            )

        matched_words = sum(
            1
            for word in query_words
            if word in text_norm
        )

        if query_words:

            coverage = (
                matched_words /
                len(query_words)
            )

            score += coverage * 120

        for word in query_words:

            if re.search(
                rf"\b{re.escape(word)}\b",
                text_norm
            ):
                score += 20

        if query_norm in text_norm:
            score += 180

        concept_words = [
            word
            for word in query_words
            if len(word) >= 4
        ]

        if concept_words:

            concept_phrase = " ".join(
                concept_words
            )

            if concept_phrase in text_norm:
                score += 150

        for word in query_words:

            title_pattern = (
                rf"\b\d+\s+\d+"
                rf"(?:\s+\d+)*\s+"
                rf".{{0,80}}\b"
                rf"{re.escape(word)}\b"
            )

            if re.search(
                title_pattern,
                text_norm
            ):
                score += 100

        original_tokens = re.findall(
            r"\b[A-Za-z0-9]{2,}\b",
            query
        )

        for token in original_tokens:

            if (
                token.isupper()
                and len(token) >= 2
            ):

                token_norm = normalize_text(
                    token
                )

                if re.search(
                    rf"\b{re.escape(token_norm)}\b",
                    text_norm
                ):
                    score += 300

        if intent == "definition":

            markers = [
                "este",
                "reprezinta",
                "se defineste",
                "se numeste",
                "consta",
                "are ca scop",
                "is defined",
                "represents",
                "means"
            ]

            for marker in markers:

                if marker in text_norm:
                    score += 25

        elif intent == "role":

            markers = [
                "scopul",
                "rolul",
                "are rolul",
                "permite",
                "ajuta",
                "asigura",
                "impiedica",
                "purpose",
                "role",
                "helps",
                "prevents"
            ]

            for marker in markers:

                if marker in text_norm:
                    score += 45

        elif intent == "process":

            markers = [
                "functionarea",
                "functioneaza",
                "pasul",
                "se aplica",
                "se genereaza",
                "se selecteaza",
                "se alege",
                "rezulta",
                "algoritmul",
                "step",
                "works",
                "applied"
            ]

            for marker in markers:

                if marker in text_norm:
                    score += 35

        elif intent == "comparison":

            markers = [
                "diferenta",
                "diferente",
                "compararea",
                "comparativ",
                "spre deosebire",
                "similaritate",
                "similaritati",
                "difference",
                "compared",
                "comparison",
                "similarity"
            ]

            for marker in markers:

                if marker in text_norm:
                    score += 40

        scored_docs.append(
            (
                score,
                doc
            )
        )

    scored_docs.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return [
        doc
        for score, doc
        in scored_docs[:k_final]
    ]


def rerank_with_llm(
    query,
    docs,
    top_k=5
):

    if not docs:
        return []

    candidates_text = []

    for i, doc in enumerate(
        docs
    ):

        text = doc.page_content.replace(
            "\n",
            " "
        )

        text = re.sub(
            r"\s+",
            " ",
            text
        ).strip()

        candidates_text.append(
            f"""
DOCUMENT {i}
FILE: {doc.metadata.get("source_file")}
PAGE: {doc.metadata.get("page")}
TEXT: {text[:500]}
"""
        )

    prompt = f"""
Rank the following Genetic Algorithms course documents
by relevance to the user's question.

QUESTION:
{query}

Prefer:
- direct answers;
- exact concepts;
- relevant section headings;
- definitions for definition questions;
- role explanations for role questions;
- procedures for how questions;
- documents discussing both concepts for comparisons.

Ignore documents that only contain similar vocabulary.

Return ONLY a JSON list of indexes.

DOCUMENTS:

{"".join(candidates_text)}
"""

    try:

        response = llm.invoke(
            prompt
        )

        content = (
            response.content
            .strip()
        )

        try:

            ranking = json.loads(
                content
            )

        except json.JSONDecodeError:

            match = re.search(
                r"\[[0-9,\s]+\]",
                content
            )

            if not match:
                return docs[:top_k]

            ranking = json.loads(
                match.group()
            )

    except Exception as e:

        print(
            f"Reranker error: {e}"
        )

        return docs[:top_k]

    ranked_docs = []
    used = set()

    for index in ranking:

        if (
            isinstance(index, int)
            and 0 <= index < len(docs)
            and index not in used
        ):

            ranked_docs.append(
                docs[index]
            )

            used.add(
                index
            )

        if len(ranked_docs) >= top_k:
            break

    if len(ranked_docs) < top_k:

        for i, doc in enumerate(
            docs
        ):

            if i not in used:

                ranked_docs.append(
                    doc
                )

            if len(ranked_docs) >= top_k:
                break

    return ranked_docs


def retrieve_final(
    query,
    k_candidates=10,
    k_final=5
):

    candidates = retrieve_hybrid(
        query,
        k_final=k_candidates,
        k_vector=25
    )

    return rerank_with_llm(
        query,
        candidates,
        top_k=k_final
    )


def is_likely_followup(question):

    question_norm = normalize_text(
        question
    )

    words = question_norm.split()

    followup_patterns = [
        "explica mi mai simplu",
        "explica mai simplu",
        "da mi un exemplu",
        "mai multe detalii",
        "detaliaza",
        "de ce",
        "cum asa",
        "explain it more simply",
        "explain more simply",
        "give me an example",
        "more details",
        "why",
        "continue",
        "tell me more",
        "what about it",
        "and why"
    ]

    if any(
        pattern in question_norm
        for pattern in followup_patterns
    ):
        return True

    if len(words) <= 5:
        return True

    return False


def build_retrieval_query(
    question,
    history=None
):

    if not history:
        return question

    if not is_likely_followup(
        question
    ):
        return question

    recent_history = history[-4:]

    history_text = ""

    for item in recent_history:

        role = item.get(
            "role",
            ""
        )

        content = item.get(
            "content",
            ""
        )

        content = str(content)

        if len(content) > 700:
            content = content[:700]

        history_text += (
            f"{role}: "
            f"{content}\n"
        )

    prompt = f"""
Rewrite the current user message as a short standalone
search query for Genetic Algorithms course materials.

Use conversation history only to resolve references.

Do not answer the question.

Preserve:
- language;
- technical terms;
- acronyms such as PMX, OX and CX.

HISTORY:
{history_text}

CURRENT MESSAGE:
{question}

Return only the search query.
"""

    try:

        response = llm.invoke(
            prompt
        )

        rewritten = (
            response.content
            .strip()
            .strip('"')
        )

        if rewritten:
            return rewritten

    except Exception as e:

        print(
            f"Query rewrite error: {e}"
        )

    return question


def answer_question(
    question,
    history=None
):

    if history is None:
        history = []

    answer_language = detect_language(
        question
    )

    retrieval_query = build_retrieval_query(
        question,
        history
    )

    results = retrieve_final(
        retrieval_query,
        k_candidates=10,
        k_final=5
    )

    context_parts = []

    for i, doc in enumerate(
        results
    ):

        content = str(
            doc.page_content
        )

        if len(content) > 2200:
            content = content[:2200]

        context_parts.append(
            f"""
[SOURCE {i + 1}]
FILE: {doc.metadata.get("source_file")}
PAGE: {doc.metadata.get("page")}

{content}
"""
        )

    context = "\n".join(
        context_parts
    )

    history_text = ""

    for item in history[-4:]:

        role = item.get(
            "role",
            ""
        )

        content = str(
            item.get(
                "content",
                ""
            )
        )

        if len(content) > 900:
            content = content[:900]

        if role == "user":
            role_name = "User"

        elif role == "assistant":
            role_name = "Assistant"

        else:
            role_name = role

        history_text += (
            f"{role_name}: "
            f"{content}\n"
        )

    prompt = f"""
You are an academic assistant specialized in
Genetic Algorithms.

Use the retrieved university course and laboratory
materials as the primary source.

PREVIOUS CONVERSATION:

{history_text}

RETRIEVED CONTEXT:

{context}

CURRENT QUESTION:

{question}

REQUIRED ANSWER LANGUAGE:

{answer_language}

INSTRUCTIONS:

1. Answer primarily from the retrieved materials.

2. Prefer sources that directly answer the question.

3. Do not cite a source unless it supports the statement.

4. Never invent definitions, formulas, numerical values,
   probabilities, constraints or technical claims.

5. Use conversation history for follow-up questions.

6. If the user asks for a simpler explanation, make the
   answer simpler and shorter.

7. Mention the main source file and page actually used.

8. If the information is not present in the materials,
   clearly say so.

9. Answer ENTIRELY in {answer_language}.

   Romanian question -> Romanian answer.

   English question -> English answer.

   The language of the source documents must never
   determine the language of the answer.

10. Use Markdown-compatible LaTeX.

    Inline formula:
    $f(x)=x^2$

    Display formula:
    $$f(x)=x^2$$

    Set:
    $x_i \\in \\{{0,1\\}}$

    Vector:
    $\\mathbf{{x}}$

    Sum:
    $$\\sum_{{i=1}}^{{n}} x_i$$

11. Never use Markdown math code blocks.

12. Never use backslash-parenthesis or
    backslash-square-bracket math delimiters.

13. Do not reconstruct damaged formulas by guessing.

14. If you generate a new example, clearly say that it is
    generated.

15. Example parameter values must not be presented as
    values from the course materials.

16. Python code must use a normal Python code block.

17. Never output rendering artifacts such as svg or svgsvg.

ANSWER IN {answer_language}:
"""

    try:

        response = llm.invoke(
            prompt
        )

        return (
            response.content,
            results
        )

    except Exception as e:

        print(
            f"Final answer error: {e}"
        )

        fallback_language = answer_language

        if fallback_language == "English":

            fallback_answer = (
                "The request could not be completed because "
                "the language model request exceeded the "
                "available API limit. Please try again in a "
                "few moments."
            )

        else:

            fallback_answer = (
                "Cererea nu a putut fi finalizata deoarece "
                "apelul catre model a depasit limita "
                "disponibila a API-ului. Incearca din nou "
                "peste cateva momente."
            )

        return (
            fallback_answer,
            results
        )