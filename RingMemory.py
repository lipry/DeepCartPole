import collections as col
import random

class RingMemory:

    def __init__(self, size):
        self.memory = col.deque(maxlen=size)

    def add(self, x):
        self.memory.append(x)

    def sample(self, n):
        n = min(n, len(self.memory))
        return random.sample(self.memory, n)
