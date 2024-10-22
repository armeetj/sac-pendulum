from typing import List, NamedTuple, Any
import numpy as np
import random


class Frame(NamedTuple):
    state: Any
    action: Any
    reward: Any
    state_next: Any
    done: bool


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer: List[Frame] = []

    def __len__(self):
        return len(self.buffer)

    def add(self, frame: Frame) -> None:
        self.buffer.append(frame)
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)

    def sample(self, sz) -> List[Frame]:
        return random.sample(self.buffer, sz)
