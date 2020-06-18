from unittest import TestCase

from utilities.general_utilities import get_bunch_config_from_json

TEST_CONFIG_FILE = "tests/example_config.json"


class ConfigTest(TestCase):
    def setUp(self) -> None:
        self.config = get_bunch_config_from_json(TEST_CONFIG_FILE)

    def test_string_config_value_should_have_correct_type(self) -> None:
        assert type(self.config.string_config_key) == str

    def test_string_config_value_should_have_corret_value(self) -> None:
        assert self.config.string_config_key == "some string"

    def test_bool_config_value_should_have_corret_type(self) -> None:
        assert type(self.config.bool_config_key) == bool

    def test_bool_config_value_should_have_corret_value(self) -> None:
        # amounts to checking if value is == True
        assert self.config.bool_config_key

    def test_int_config_value_should_have_correct_type(self) -> None:
        assert type(self.config.int_config_key) == int

    def test_int_config_value_should_have_corret_value(self) -> None:
        assert self.config.int_config_key == 0

    def test_float_config_value_should_have_corret_type(self) -> None:
        assert type(self.config.float_config_key) == float

    def test_float_config_value_should_have_corret_value(self) -> None:
        assert self.config.float_config_key == 0.0
