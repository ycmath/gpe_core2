# gpe_core2/repetition_detector.py
from typing import Sequence, Any

class RepetitionDetector:
    """Detects if a single value repeats ≥ threshold times."""
    @staticmethod
    def is_repetitive(values: Sequence[Any], threshold: int = 3) -> bool:
        if not values:
            return False
        first = values[0]
        return sum(1 for v in values if v == first) >= threshold
