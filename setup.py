"""Installer for pyrefact"""
from pathlib import Path

import setuptools

setuptools.setup(
    name="pyrefact",
    version="1",
    description="Automatic python refactoring",
    author="Olle Lindgren",
    author_email="olle.ln@outlook.com",
    packages=setuptools.find_packages(str(Path(__file__).parent)),
)
