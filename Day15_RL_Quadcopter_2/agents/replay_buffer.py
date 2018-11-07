import numpy as np
from task import Task
import random


class ReplayBuffer():
    def __init__(self, buffer_size):
        # Critic (value-based) : environment state / action —> critique
        # Task (environment) information
        self.buffer_size = buffer_size
        #self.memory
    def add(self, state, action, reward, done):
        pass
    def random_sample(self, batch_size):
        return random.sample(self.memory, self.batch_size)