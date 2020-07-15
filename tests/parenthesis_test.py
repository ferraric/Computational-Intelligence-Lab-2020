from unittest import TestCase

from apply_rules import matched


class ParenthesisTest(TestCase):
    def test_single_parenthesis_open(self) -> None:
        assert matched("(") == "("

    def test_single_parenthesis_closed(self) -> None:
        assert matched(")") == ")"

    def test_two_parenthesis_closed(self) -> None:
        assert matched("()") == ""

    def test_two_parenthesis_nomatch(self) -> None:
        assert matched(")(") == ")("

    def test_two_parenthesis_open(self) -> None:
        assert matched("))") == "))"

    def test_four_parenthesis_closed(self) -> None:
        assert matched("(())") == ""

    def test_three_parenthesis_closed(self) -> None:
        assert matched("())") == ")"

    def test_three_parenthesis_open(self) -> None:
        assert matched("(()") == "("

    def test_three_parenthesis_nomatch(self) -> None:
        assert matched(")((") == ")(("

    def test_four_parenthesis_nomatch(self) -> None:
        assert matched("))((") == "))(("

    def test_four_parenthesis_twomatch(self) -> None:
        assert matched("()()") == ""

    def test_three_parenthesis_test(self) -> None:
        assert matched("( hello ) ( ") == " hello ( "
