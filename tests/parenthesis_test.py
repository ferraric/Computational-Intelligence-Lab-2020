from unittest import TestCase

from rules.rules import RuleClassifier


class ParenthesisTest(TestCase):
    def test_single_parenthesis_open(self) -> None:
        assert RuleClassifier._remove_matching_parenthesis("(") == "("

    def test_single_parenthesis_closed(self) -> None:
        assert RuleClassifier._remove_matching_parenthesis(")") == ")"

    def test_two_parenthesis_closed(self) -> None:
        assert RuleClassifier._remove_matching_parenthesis("()") == ""

    def test_two_parenthesis_nomatch(self) -> None:
        assert RuleClassifier._remove_matching_parenthesis(")(") == ")("

    def test_two_parenthesis_open(self) -> None:
        assert RuleClassifier._remove_matching_parenthesis("))") == "))"

    def test_four_parenthesis_closed(self) -> None:
        assert RuleClassifier._remove_matching_parenthesis("(())") == ""

    def test_three_parenthesis_closed(self) -> None:
        assert RuleClassifier._remove_matching_parenthesis("())") == ")"

    def test_three_parenthesis_open(self) -> None:
        assert RuleClassifier._remove_matching_parenthesis("(()") == "("

    def test_three_parenthesis_nomatch(self) -> None:
        assert RuleClassifier._remove_matching_parenthesis(")((") == ")(("

    def test_four_parenthesis_nomatch(self) -> None:
        assert RuleClassifier._remove_matching_parenthesis("))((") == "))(("

    def test_four_parenthesis_twomatch(self) -> None:
        assert RuleClassifier._remove_matching_parenthesis("()()") == ""

    def test_three_parenthesis_test(self) -> None:
        assert (
            RuleClassifier._remove_matching_parenthesis("( hello ) ( ") == " hello  ( "
        )
