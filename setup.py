"""Installer for pyrefact"""
from pathlib import Path

import setuptools

with open(Path(__file__).parent / "requirements.txt", encoding="utf-8") as stream:
    REQUIREMENTS = list(filter(bool, (x.strip() for x in stream.readlines())))

setuptools.setup(
    name="pyrefact",
    version="2",
    description="Automatic python refactoring",
    author="Olle Lindgren",
    author_email="olle.ln@outlook.com",
    packages=["pyrefact"],
    install_requires=REQUIREMENTS,
)
