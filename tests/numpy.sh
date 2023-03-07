#!/bin/bash
# Clone the numpy git repo and try to rewrite all python files in it
# TODO fix the commit hash but keep shallow commit somehow

mkdir -p ~/.cache/pyrefact/tests
cd ~/.cache/pyrefact/tests
rm -rf numpy
git clone --depth=1 https://github.com/numpy/numpy.git
cd numpy
pyrefact -sv $(git ls-files | grep 'numpy\/core\/\w*\.py$')
