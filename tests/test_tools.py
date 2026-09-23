from chatbot_core import calculator


def test_calculator_multiplication():

    result = calculator.invoke(
        {"expression": "25 * 17"}
    )

    assert result == "425"

def test_calculator_invalid_expression():

    result = calculator.invoke(
        {"expression": "abc + 10"}
    )

    assert result == "Invalid mathematical expression."

def test_calculator_division_by_zero():

    result = calculator.invoke(
        {"expression": "10 / 0"}
    )

    assert result.startswith("Calculation error:")