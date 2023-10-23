#!/usr/bin/env python3
import unittest


class TestImports(unittest.TestCase):
    def test_main_imports(self):
        import pyrefact

        # pyrefact.main
        assert callable(pyrefact.main)
        assert callable(pyrefact.format_code)
        assert callable(pyrefact.format_file)

        # pyrefact.pattern_matching
        assert callable(pyrefact.compile)
        assert callable(pyrefact.findall)
        assert callable(pyrefact.finditer)
        assert callable(pyrefact.search)
        assert callable(pyrefact.sub)

    def runTest(self):
        self.test_main_imports()


def main() -> int:
    test_result = TestImports().run()

    if not test_result.wasSuccessful():
        print("FAILED")
        return 1

    print("PASSED")
    return 0


if __name__ == "__main__":
    unittest.main()
