from typing import NamedTuple, Any
import numpy as np


class Frame(NamedTuple):
    state: Any
    action: Any
    reward: Any
    state_next: Any