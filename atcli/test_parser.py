import unittest
from atcli import parser


class TestParser(unittest.TestCase):

    def test_parser(self) -> None:
        r = parser.parse_file("test_data/at1.txt")
        r.render_all_to_svg(out_path="test_data/at1.svg")
