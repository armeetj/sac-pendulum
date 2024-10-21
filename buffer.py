from typing import List, NamedTuple, Any
import numpy as np


class Frame(NamedTuple):
    state: Any
    action: Any
    reward: Any
    state_next: Any


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer: List[Frame] = []

    def add(self, frame: Frame):
        self.buffer.append(frame)
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)

    def sample(self, sz):

