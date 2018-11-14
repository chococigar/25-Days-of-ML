import numpy as np
from task import Task
#from agents.actor import Actor
#from agents.critic import Critic
from agents.replay_buffer import ReplayBuffer
from task import Task
from keras.layers import Dense, Input, Add

import tensorflow as tf

class ActorCritic():
    def __init__(self):
        # Task (environment) information
        self.task = task
        self.state_size = task.state_size #18
        self.action_size = task.action_size #4
        self.action_low = task.action_low
        self.action_high = task.action_high
        self.action_range = self.action_high - self.action_low

        self.learning_rate = 0.001
        self.epsilon=1.0
        self.epsilon_decay = 0.99
        self.gamma = 0.95
        self.tau = 0.125

        self.buffer = ReplayBuffer(2000)
        self.actor_state_input, self.actor_model = self.create_actor_model()
        self.actor_state_input, self.actor_model = self.create_actor_model()

        self.w = np.random.normal(
            size=(self.state_size, self.action_size),  # weights for simple linear policy: state_space x action_space
            scale=(self.action_range / (2 * self.state_size))) # start producing actions in a decent range

        # Score tracker and learning parameters
        self.best_w = None
        self.best_score = -np.inf
        self.noise_scale = 0.1

        # Episode variables
        self.reset_episode()

# ========================= CREATE MODEL ========================= #

    #the Actor updates its policy parameters (weights) using this q value
    def create_actor_model(self): #policy-based
        state_input = Input(shape=self.state_size)
        state_h1 = Dense(units=64, init="uniform", activation="relu")(state_input)
        state_h2 = Dense(units=32, init="uniform", activation="relu")(state_h1)
        #h2 because recommended architecture for these AC networks

        action_input = Input(shape=self.action_size)
        action_h1 = Dense(units=32, init="uniform", activation="relu")(action_input)

        # reconsider merging elsewhere
        merged = Add()([state_h2, action_h1])
        merged_h1 = Dense(units=16, init="uniform", activation="relu")(merged)
        output = Dense(units=1, init="uniform", activation="relu")

        model = Model(input=[state_input, action_input], ooutput=output)
        model.compile(loss="adam", optimizer=sgd, metrics=["accuracy"])
        
        return [state_input], model

    def create_critic_model(self): #Q-value
        state_input = Input(shape=self.state_size)
        state_h1 = Dense(units=64, init="uniform", activation="relu")(state_input)
        state_h2 = Dense(units=32, init="uniform", activation="relu")(state_h1)
        #h2 because recommended architecture for these AC networks

        action_input = Input(shape=self.action_size)
        action_h1 = Dense(units=32, init="uniform", activation="relu")(action_input)

        # reconsider merging elsewhere
        merged = Add()([state_h2, action_h1])
        merged_h1 = Dense(units=16, init="uniform", activation="relu")(merged)
        output = Dense(units=1, init="uniform", activation="relu")

        model = Model(input=[state_input, action_input], ooutput=output)
        model.compile(loss="adam", optimizer=sgd, metrics=["accuracy"])
        
        return [state_input, action_input], model

# ========================= CREATE MODEL ========================= #

    def reset_episode(self):
        self.total_reward = 0.0
        self.count = 0
        state = self.task.reset()
        return state

    def step(self, reward, done):
        # Save experience / reward
        self.total_reward += reward
        self.count += 1

        # Learn, if at end of episode
        if done:
            self.learn()

    def act(self, state):
        # Choose action based on given state and policy
        #action = np.dot(state, self.w)  # simple linear policy
        #[1, 18], [18,4] --> 
        action = np.dot(state, self.w)
        print(action)
        n = self.count or 1
        print(n)
        action[:] = [x / self.count for x in action]
        #print("self w shape is {}", action.shape : 4,)
        return action

    def learn(self):
        
        # Learn by random policy search, using a reward-based score
        self.score = self.total_reward / float(self.count) if self.count else 0.0
        if self.score > self.best_score:
            self.best_score = self.score
            self.best_w = self.w
            self.noise_scale = max(0.5 * self.noise_scale, 0.01)
        else:
            self.w = self.best_w
            self.noise_scale = min(2.0 * self.noise_scale, 3.2)
        self.w = self.w + self.noise_scale * np.random.normal(size=self.w.shape)  # equal noise in all directions
        