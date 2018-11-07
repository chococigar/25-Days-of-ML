import numpy as np
from task import Task

class Critic():
    def __init__(self, task):
        # Critic (value-based) : environment state / action —> critique
        # Task (environment) information
        self.td_error = 0
        self.gamma = 1
        self.r = 1
    def reset_episode(self):
        self.total_reward = 0.0
        self.count = 0
        state = self.task.reset()
        return state
    def value_of(self, state):
        value = 1 #TO CHANGE something with state
        return value
    def TD_error(self, state, next_state, action):
        new_state = state #TO CHANGE do something wti
        td_error = self.r + self.gamma * value_of(next_state) - value_of(state)
        self.td_error = td_error
        return error