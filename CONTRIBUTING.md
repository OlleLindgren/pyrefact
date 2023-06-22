# Contributing to Pyrefact

First off, thank you for considering contributing to Pyrefact!

## Where to Start

If you're looking to contribute but aren't sure where to start, check out the open issues on GitHub, especially those labeled as "good first issue". These are tasks that have been identified as a good place to start for new contributors.

## Ways to Contribute

There are many ways you can directly contribute to Pyrefact:

Fix bugs: Look through the GitHub issues for bugs. Anything tagged with "bug" is open to be fixed.
Implement enhancements: Look through the GitHub issues for enhancements. This is a list of features that the community has requested.
Improve documentation: Pyrefact could always use more documentation. Whether it's more comments in the code, more explanation in the readme, or better usage guides, helpful documentation is always a great contribution!

## Code Contributions

Pyrefact is structured to apply a number of "fixes" in sequence after eacother, over and over until convergence or a maximum number of iterations.
Although these have some categorization, most of them reside in [fixes.py](pyrefact/fixes.py), which is where most new ones will end up as well,
although I am open to suggestions on how to improve this structure.

When writing a new fix, please follow one of two patterns:

```python
@processing.fix
def my_simple_fix(source: str) -> str:
    """A docstring explaining what this is for"""
    root = parsing.parse(source)  # Use this instead of ast.parse(), as it is cached
    template = ...  # Use templates to pattern-match problems you're fixing
    for before in parsing.match_template(root, template):
        after = ast.Constant(value=1, kind=None)
        yield before, after  # With @procesisng.fix, we yield (before, after) pairs
```

```python
def my_complicated_fix(source: str) -> str:
    """A docstring explaining what this is for"""
    root = parsing.parse(source)  # Use this instead of ast.parse(), as it is cached
    template = ...  # Use templates to pattern-match problems you're fixing
    for node in parsing.match_template(root, template):
        # Do some replacements, removals or additions
    # Modify source somehow
    return source  # Without @processing.fix, we return a modified version of the initial source code
```

Next, copy any of the recently modified test files under [/tests/unit](/tests/unit), rename it to `test_<name_of_fix>.py`, and modify it so it tests your code.
Feel free to also add or update the integration test cases in [integration_test_cases.py](/tests/integration/integration_test_cases.py) to test additional
aspects of your fix, or how your fix interacts with other parts of pyrefact.

Then, add your fix to either `_multi_run_fixes()` or `_single_run_fixes()` in [main.py](/pyrefact/main.py), depending on whether you want it "in the loop",
or "before the loop" of other fixes. Most of the fixes run in `_multi_run_fixes()`.

Finally, run [tests/main.py](/tests/main.py) and see that all tests passes. It will automatically find your new test file if you used the normal naming convention.
In the GitHub CI, this will run on Windows, MacOS and Linux, and on all python versions between 3.8 and 3.12, but if tests pass on both python3.8 and python3.12
it's uncommon for them to fail on any of the other versions. All tests must pass before code can be merged.

Here's how you can contribute code:

* Fork the Repo: Click the "Fork" button in the top right of the main repo page. This will create a copy of the repository that you can edit.
* Clone your fork: Use `git clone https://github.com/<your_username>/pyrefact.git` to clone your fork to your local machine.
* Create a branch: It's best practice to create a new branch for each new feature or bugfix. You can do this using `git checkout -b <branch-name>`.
* Make your changes: Make the changes you want to contribute.
* Commit your changes: Use `git commit -am "Your descriptive commit message"` to commit your changes. Stick to the present-tense, imperative-style commit messages.
* Push your changes: Use `git push origin <branch-name>` to push your changes to your fork. Check so that all automated tests pass.
* Create a pull request: Go to your fork on GitHub and click the "New Pull Request" button. Fill out the form and submit it!

## Code Style

The Pyrefact repo is formatted with itself, and periodically with [black-compact](https://github.com/OlleLindgren/black-compact), a fork of the regular black formatter.
What's more important than formatting, though, is that your commits don't include additional formatting beyond the actual changes you're making. So it's better that you format too little than too much.

Appropriate formatting can be accomplished by:
```bash
cd $(git rev-parse --show-toplevel)  # To the root of the repo
pip install git+https://github.com/OlleLindgren/black-compact@b2fe670fafa22fea0660feb037f0909a10358c76#egg=black
black ./pyrefact
```

## Questions?

If you have any questions, please don't hesitate to create an issue on GitHub. We'll do our best to get back to you as soon as possible!
