"""
Regime Detection Module.

Market regime detection using various methods:
- Wasserstein distance-based clustering
- Hidden Markov Models (Gaussian Mixture alternative)
"""

from typing import List

__all__: List[str] = [
    "WassersteinRegimeDetector",
    "HMMRegimeDetector",
]

# Import main classes
from src.regime_detection.wasserstein_regime import WassersteinRegimeDetector
from src.regime_detection.hmm_regime import HMMRegimeDetector
