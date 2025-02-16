from typing import Dict, NamedTuple, Optional
from .sample_registry import sample_registry


class StockUsageAndRefill(NamedTuple):
    usage: Dict[str, float]
    refill: Dict[str, float]
    
    def to_dict(self):
        return {
            'usage': self.usage,
            'refill': self.refill
        }