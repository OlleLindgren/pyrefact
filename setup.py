"""Installer for pyrefact"""
import sys
import warnings

import setuptools

if tuple(sys.version_info) < (3, 9):
    warnings.warn("Pyrefact is not tested with python < 3.9, and may not work.")

setuptools.setup(
    name="pyrefact",
    version="17",
    description="Automatic python refactoring",
    author="Olle Lindgren",
    author_email="olle.ln@outlook.com",
    packages=["pyrefact"],
    install_requires=["black>=22.1.0", "isort==5.10.1", "rmspace>=6"],
    url="https://github.com/OlleLindgren/pyrefact",
)
