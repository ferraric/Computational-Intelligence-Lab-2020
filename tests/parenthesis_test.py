from unittest import TestCase

from rules.rules import ParenthesisRule


class ParenthesisTest(TestCase):
    def test_single_parenthesis_open(self) -> None:
        assert ParenthesisRule._remove_matching_parenthesis("(") == "("

    def test_single_parenthesis_closed(self) -> None:
        assert ParenthesisRule._remove_matching_parenthesis(")") == ")"

    def test_two_parenthesis_closed(self) -> None:
        assert ParenthesisRule._remove_matching_parenthesis("()") == ""

    def test_two_parenthesis_nomatch(self) -> None:
        assert ParenthesisRule._remove_matching_parenthesis(")(") == ")("

    def test_two_parenthesis_open(self) -> None:
        assert ParenthesisRule._remove_matching_parenthesis("))") == "))"

    def test_four_parenthesis_closed(self) -> None:
        assert ParenthesisRule._remove_matching_parenthesis("(())") == ""

    def test_three_parenthesis_closed(self) -> None:
        assert ParenthesisRule._remove_matching_parenthesis("())") == ")"

    def test_three_parenthesis_open(self) -> None:
        assert ParenthesisRule._remove_matching_parenthesis("(()") == "("

    def test_three_parenthesis_nomatch(self) -> None:
        assert ParenthesisRule._remove_matching_parenthesis(")((") == ")(("

    def test_four_parenthesis_nomatch(self) -> None:
        assert ParenthesisRule._remove_matching_parenthesis("))((") == "))(("

    def test_four_parenthesis_twomatch(self) -> None:
        assert ParenthesisRule._remove_matching_parenthesis("()()") == ""

    def test_three_parenthesis_test(self) -> None:
        assert (
            ParenthesisRule._remove_matching_parenthesis("( hello ) ( ") == " hello  ( "
        )
