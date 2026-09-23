from evaluation.groundedness_evaluation import evaluate_groundedness


def test_grounded_answer():

    context = (
        "Mutatia este un operator genetic. "
        "Mutatia introduce modificari in cromozom "
        "si contribuie la diversitatea populatiei."
    )

    answer = (
        "Mutatia este un operator genetic. "
        "Aceasta contribuie la diversitatea populatiei."
    )

    result = evaluate_groundedness(
        answer,
        context
    )

    assert result["grounded"] is True
    assert result["score"] >= 0.5


def test_ungrounded_answer():

    context = (
        "Mutatia este un operator genetic. "
        "Mutatia introduce modificari in cromozom "
        "si contribuie la diversitatea populatiei."
    )

    answer = (
        "Mutatia este un operator genetic. "
        "Mutatia garanteaza intotdeauna "
        "gasirea solutiei optime."
    )

    result = evaluate_groundedness(
        answer,
        context
    )

    assert result["grounded"] is False

    assert any(
        item["grounded"] is False
        for item in result["sentences"]
    )
def test_semantic_paraphrase_grounded():

    context = (
        "Mutatia contribuie la diversitatea populatiei."
    )

    answer = (
        "Mutatia ajuta la mentinerea varietatii indivizilor."
    )

    result = evaluate_groundedness(
        answer,
        context
    )

    assert result["grounded"] is True

    assert (
        result["sentences"][0]["semantic_score"]
        >= 0.75
    )

    assert (
        result["sentences"][0]["negation_mismatch"]
        is False
    )
def test_negation_contradiction_not_grounded():

    context = (
        "Mutatia contribuie la diversitatea populatiei."
    )

    answer = (
        "Mutatia nu contribuie la diversitatea populatiei."
    )

    result = evaluate_groundedness(
        answer,
        context
    )

    assert result["grounded"] is False

    assert (
        result["sentences"][0]["negation_mismatch"]
        is True
    )