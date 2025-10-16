"""
Setup script for NSE Adaptive Regime Trading System.

Proper package configuration for easy installation with pip install -e .
"""

from pathlib import Path

from setuptools import find_packages, setup


def read_requirements():
    """Read requirements from requirements-minimal.txt."""
    req_file = Path(__file__).parent / "requirements-minimal.txt"
    if req_file.exists():
        with open(req_file) as f:
            return [
                line.strip()
                for line in f
                if line.strip() and not line.startswith('#') and not line.startswith('=')
            ]
    return []


setup(
    name="nse-adaptive-regime-trading",
    version="0.1.0",
    description="Production-grade algorithmic trading system for NSE",
    author="Vatsal Mehta",
    packages=find_packages(include=['src', 'src.*']),
    install_requires=read_requirements(),
    python_requires=">=3.11",
    entry_points={
        'console_scripts': [
            'nse-trade=src.cli:main',
        ],
    },
)

