"""Miscellaneious Utils"""
from typing import Dict, Sequence

def extract(keys: Sequence, dict: Dict):
    """Restrict dict to keys in `keys`"""
    return {key: dict[key] for key in keys}
