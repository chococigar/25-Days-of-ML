import numpy as np
from task import Task
import random
from collections import deque

class ReplayBuffer():
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.memory = deque(maxlen=buffer_size) #deque : double-ended queue
    def add(self, state, next_state, action, reward, done):
        self.memory.append((state, next_state, action, reward))
    def random_sample(self, batch_size):
        return random.sample(self.memory, self.batch_size)