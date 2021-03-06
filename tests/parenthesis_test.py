from unittest import TestCase

from rule.rules import ParenthesisRule


class RemoveMatchingParenthesisTest(TestCase):
    def setUp(self) -> None:
        self.parenthesis_rule = ParenthesisRule(")", "(")

    def test_single_parenthesis_open(self) -> None:
        assert self.parenthesis_rule.remove_matching_parenthesis("(") == "("

    def test_two_parenthesis_closed(self) -> None:
        assert self.parenthesis_rule.remove_matching_parenthesis(" ( h ) ") == "  h  "

    def test_two_parenthesis_open(self) -> None:
        assert self.parenthesis_rule.remove_matching_parenthesis(" ) ) ") == " ) ) "

    def test_four_parenthesis_closed(self) -> None:
        assert (
            self.parenthesis_rule.remove_matching_parenthesis(" ( h ( h ) h ) ")
            == "  h  h  h  "
        )

    def test_three_parenthesis_closed(self) -> None:
        assert (
            self.parenthesis_rule.remove_matching_parenthesis(" ( h ) ) ") == "  h  ) "
        )

    def test_three_parenthesis_nomatch(self) -> None:
        assert self.parenthesis_rule.remove_matching_parenthesis(" ) ( ( ") == " ) ( ( "

    def test_four_parenthesis_twomatch(self) -> None:
        assert (
            self.parenthesis_rule.remove_matching_parenthesis(" ( h ) ( h ) ")
            == "  h   h  "
        )

    def test_three_parenthesis_test(self) -> None:
        assert (
            self.parenthesis_rule.remove_matching_parenthesis(" ( hi ) ( ")
            == "  hi  ( "
        )

    def test_four_parenthesis_closed_open(self) -> None:
        assert (
            self.parenthesis_rule.remove_matching_parenthesis(" ( ) ) ( ") == "   ) ( "
        )


class RemoveUnmatchingParenthesisTest(TestCase):
    def setUp(self) -> None:
        self.parenthesis_rule = ParenthesisRule(")", "(")

    def test_remove_parenthesis_closed(self) -> None:
        assert self.parenthesis_rule.remove_rule_pattern_from(" ( hi ) ") == "( hi )"

    def test_remove_parenthesis_single(self) -> None:
        assert self.parenthesis_rule.remove_rule_pattern_from(" ) ") == ""

    def test_remove_parenthesis_triple(self) -> None:
        assert self.parenthesis_rule.remove_rule_pattern_from(" ( ( ( ") == ""

    def test_remove_parenthesis_closed_open(self) -> None:
        assert (
            self.parenthesis_rule.remove_rule_pattern_from(" ( hi ) ) ( ")
            == "( hi ) ) ("
        )

    def test_remove_parenthesis_open_closed(self) -> None:
        assert self.parenthesis_rule.remove_rule_pattern_from(" ( ( hi ) ") == "( hi )"

    def test_remove_parenthesis_open_closed_double(self) -> None:
        assert (
            self.parenthesis_rule.remove_rule_pattern_from(" ( hi ) ) ) ") == "( hi )"
        )

    def test_remove_parenthesis_closed_unmatching_triple(self) -> None:
        assert (
            self.parenthesis_rule.remove_rule_pattern_from(" ( ) ) ) ( ") == "( ) ) ) ("
        )
