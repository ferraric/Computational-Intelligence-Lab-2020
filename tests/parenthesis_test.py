from unittest import TestCase

from rules.rules import ParenthesisRule


class RemoveMatchingParenthesisTest(TestCase):
    def test_single_parenthesis_open(self) -> None:
        assert ParenthesisRule(")", "(").remove_matching_parenthesis("(") == "("

    def test_two_parenthesis_closed(self) -> None:
        assert (
            ParenthesisRule(")", "(").remove_matching_parenthesis(" ( h ) ") == "  h  "
        )

    def test_two_parenthesis_open(self) -> None:
        assert ParenthesisRule(")", "(").remove_matching_parenthesis(" ) ) ") == " ) ) "

    def test_four_parenthesis_closed(self) -> None:
        assert (
            ParenthesisRule(")", "(").remove_matching_parenthesis(" ( h ( h ) h ) ")
            == "  h  h  h  "
        )

    def test_three_parenthesis_closed(self) -> None:
        assert (
            ParenthesisRule(")", "(").remove_matching_parenthesis(" ( h ) ) ")
            == "  h  ) "
        )

    def test_three_parenthesis_nomatch(self) -> None:
        assert (
            ParenthesisRule(")", "(").remove_matching_parenthesis(" ) ( ( ")
            == " ) ( ( "
        )

    def test_four_parenthesis_twomatch(self) -> None:
        assert (
            ParenthesisRule(")", "(").remove_matching_parenthesis(" ( h ) ( h ) ")
            == "  h   h  "
        )

    def test_three_parenthesis_test(self) -> None:
        assert (
            ParenthesisRule(")", "(").remove_matching_parenthesis(" ( hi ) ( ")
            == "  hi  ( "
        )

    def test_four_parenthesis_closed_open(self) -> None:
        assert (
            ParenthesisRule(")", "(").remove_matching_parenthesis(" ( ) ) ( ")
            == "   ) ( "
        )


class RemoveUnmatchingParenthesisTest(TestCase):
    def test_remove_parenthesis_closed(self) -> None:
        assert (
            ParenthesisRule(")", "(").remove_rule_pattern_from(" ( hi ) ") == " ( hi ) "
        )

    def test_remove_parenthesis_single(self) -> None:
        assert ParenthesisRule(")", "(").remove_rule_pattern_from(" ) ") == "  "

    def test_remove_parenthesis_triple(self) -> None:
        assert ParenthesisRule(")", "(").remove_rule_pattern_from(" ( ( ( ") == "    "

    def test_remove_parenthesis_closed_open(self) -> None:
        assert (
            ParenthesisRule(")", "(").remove_rule_pattern_from(" ( hi ) ) ( ")
            == " ( hi ) ) ( "
        )

    def test_remove_parenthesis_open_closed(self) -> None:
        assert (
            ParenthesisRule(")", "(").remove_rule_pattern_from(" ( ( hi ) ")
            == "  ( hi ) "
        )

    def test_remove_parenthesis_open_closed_double(self) -> None:
        assert (
            ParenthesisRule(")", "(").remove_rule_pattern_from(" ( hi ) ) ) ")
            == " ( hi )   "
        )

    def test_remove_parenthesis_closed_unmatching_triple(self) -> None:
        assert (
            ParenthesisRule(")", "(").remove_rule_pattern_from(" ( ) ) ) ( ")
            == " ( ) ) ) ( "
        )
