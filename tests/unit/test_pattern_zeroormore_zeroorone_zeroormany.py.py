import unittest

from pyrefact.core import match_template, ZeroOrOne, ZeroOrMany, OneOrMany


class TestZeroOrOne(unittest.TestCase):

    def test(self):
        assert match_template([], [ZeroOrOne(object)])
        assert match_template([object], [ZeroOrOne(object)])
        assert not match_template([object, "asdf"], [ZeroOrOne(object)])

        assert match_template([], [ZeroOrOne(str)])
        assert match_template([""], [ZeroOrOne(str)])
        assert not match_template(["", "asdf"], [ZeroOrOne(str)])

    def runTest(self):
        self.test()


class TestZeroOrMany(unittest.TestCase):

    def test(self):
        assert match_template([], [ZeroOrMany(object)])
        assert match_template([object], [ZeroOrMany(object)])
        assert match_template([object, object], [ZeroOrMany(object)])

        assert match_template([], [ZeroOrMany(object)])
        assert match_template([""], [ZeroOrMany(str)])
        assert match_template(["a", "b"], [ZeroOrMany(object)])

    def runTest(self):
        self.test()


class TestOneOrMany(unittest.TestCase):

    def test(self):
        assert not match_template([], [OneOrMany(object)])
        assert match_template([object], [OneOrMany(object)])
        assert match_template([object, object], [OneOrMany(object)])

        assert not match_template([], [OneOrMany(object)])
        assert match_template([object], [OneOrMany(object)])
        assert match_template([1, 31], [OneOrMany(int)])

    def runTest(self):
        self.test()


class TestCombination(unittest.TestCase):

    def test(self):
        assert match_template([], [ZeroOrMany(object), ZeroOrOne(object)])
        assert match_template([object], [ZeroOrMany(object), ZeroOrOne(object)])
        assert match_template([object, object], [ZeroOrMany(object), ZeroOrOne(object)])
        assert match_template(["qwerty", "asdf", object], [ZeroOrMany(str), ZeroOrOne(object)])

        assert not match_template([], [ZeroOrMany(object), object])
        assert match_template([object], [ZeroOrMany(object), object])
        assert not match_template([object], [OneOrMany(object), object])
        assert match_template([object], [ZeroOrOne(object), object])
        assert not match_template([object], [ZeroOrOne(object), str])
        assert match_template([22], [ZeroOrOne(str), int])
        assert not match_template([], [ZeroOrMany(object), ZeroOrOne(object), OneOrMany(object)])

def main() -> int:
    # For use with ./tests/main.py, which looks for these main functions.
    # unittest.main() will do sys.exit() or something, it quits the whole
    # program after done and prevents further tests from running.
    test_result = TestZeroOrOne().run()
    if not test_result.wasSuccessful():
        test_result.printErrors()
        return 1

    test_result = TestZeroOrMany().run()
    if not test_result.wasSuccessful():
        test_result.printErrors()
        return 1

    test_result = TestOneOrMany().run()
    if not test_result.wasSuccessful():
        test_result.printErrors()
        return 1

    return 0


if __name__ == "__main__":
    unittest.main()
