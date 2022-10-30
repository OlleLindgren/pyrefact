"""Installer for pyrefact"""
import setuptools

setuptools.setup(
    name="pyrefact",
    version="4",
    description="Automatic python refactoring",
    author="Olle Lindgren",
    author_email="olle.ln@outlook.com",
    packages=["pyrefact"],
    install_requires=["black>=22.1.0", "pylint>=2.12.2", "rmspace>=6"],
)
