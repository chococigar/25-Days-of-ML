import numpy as np
from task import Task
import random
from collections import deque

class ReplayBuffer():
    def __init__(self, buffer_size, batch_size):
        self.buffer_size = buffer_size
        self.memory = deque(maxlen=buffer_size) #deque : double-ended queue
        self.batch_size = batch_size
    def add(self, state, action, reward, next_state, done):
        self.memory.append(state, action, reward, next_state, done)
    def random_sample(self, batch_size):
    	#take out from the buffer. Should I clear?
        return random.sample(self.memory, k=batch_size)
    def clear(self):
    	self.memory = None
    	self.buffer_size = None
    def len(self):
    	return(len(self.memory))