from unittest import TestCase

from rules.rules import ParenthesisRule


class RemoveMatchingParenthesisTest(TestCase):
    def test_single_parenthesis_open(self) -> None:
        assert ParenthesisRule._remove_matching_parenthesis(self, "(") == "("

    def test_single_parenthesis_closed(self) -> None:
        assert ParenthesisRule._remove_matching_parenthesis(self, ")") == ")"

    def test_two_parenthesis_closed(self) -> None:
        assert ParenthesisRule._remove_matching_parenthesis(self, "()") == ""

    def test_two_parenthesis_nomatch(self) -> None:
        assert ParenthesisRule._remove_matching_parenthesis(self, ")(") == ")("

    def test_two_parenthesis_open(self) -> None:
        assert ParenthesisRule._remove_matching_parenthesis(self, "))") == "))"

    def test_four_parenthesis_closed(self) -> None:
        assert ParenthesisRule._remove_matching_parenthesis(self, "(())") == ""

    def test_three_parenthesis_closed(self) -> None:
        assert ParenthesisRule._remove_matching_parenthesis(self, "())") == ")"

    def test_three_parenthesis_open(self) -> None:
        assert ParenthesisRule._remove_matching_parenthesis(self, "(()") == "("

    def test_three_parenthesis_nomatch(self) -> None:
        assert ParenthesisRule._remove_matching_parenthesis(self, ")((") == ")(("

    def test_four_parenthesis_nomatch(self) -> None:
        assert ParenthesisRule._remove_matching_parenthesis(self, "))((") == "))(("

    def test_four_parenthesis_twomatch(self) -> None:
        assert ParenthesisRule._remove_matching_parenthesis(self, "()()") == ""

    def test_three_parenthesis_test(self) -> None:
        assert (
            ParenthesisRule._remove_matching_parenthesis(self, "( hello ) ( ")
            == " hello  ( "
        )


class RemoveUnmatchingParenthesisTest(TestCase):
    def test_remove_parenthesis_closed(self) -> None:
        assert ParenthesisRule.remove_rule_pattern_from(self, " ( hi ) ") == " ( hi ) "

    def test_remove_parenthesis_single(self) -> None:
        assert ParenthesisRule.remove_rule_pattern_from(self, " ) ") == "  "

    def test_remove_parenthesis_double(self) -> None:
        assert ParenthesisRule.remove_rule_pattern_from(self, " ( ( ") == "  "

    def test_remove_parenthesis_closed_open(self) -> None:
        assert (
            ParenthesisRule.remove_rule_pattern_from(self, " ( hi ) ( ") == " ( hi ) "
        )

    def test_remove_parenthesis_open_closed(self) -> None:
        assert (
            ParenthesisRule.remove_rule_pattern_from(self, " ( ( hi ) ") == " ( hi ) "
        )

    def test_remove_parenthesis_closed_closed(self) -> None:
        assert (
            ParenthesisRule.remove_rule_pattern_from(self, " ( ( hi ) ") == " ( hi ) "
        )
