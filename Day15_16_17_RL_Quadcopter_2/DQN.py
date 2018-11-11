import numpy as np
from task import Task
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras.optimizers import sgd, Adam
import random

class DQN_Agent():
    def __init__(self, task):
        # Task (environment) information
        self.task = task
        self.state_size = task.state_size #18
        self.action_size = task.action_size #4
        self.action_low = task.action_low
        self.action_high = task.action_high
        self.action_range = self.action_high - self.action_low

        self.w = np.random.normal(
            size=(self.state_size, self.action_size),  # weights for simple linear policy: state_space x action_space
            scale=(self.action_range / (2 * self.state_size))) # start producing actions in a decent range

        # Score tracker and learning parameters
        self.best_w = None
        self.best_score = -np.inf
        self.noise_scale = 0.1

        # Episode variables
        self.reset_episode()

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = 0.001
        self.model = self._build_model()

    def reset_episode(self):
        self.total_reward = 0.0
        self.count = 0
        state = self.task.reset()
        return state

    def _build_model(self):
        # predict --> maximum expected future reward
        # state --> action_size...
        # Q learning : state, action --Q-table--> Q value
        # Deep Q Lea : State      --DQN-->        Q values.
        #self.state_size = self.state_size.np.reshape((1, -1)
        print("in BM action size is : ", self.action_size)
        print("in BM state size is : ", self.state_size)
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',optimizer=Adam(lr=self.learning_rate))
        #state_input = Input(shape=(self.state_size,))
        #state_h1 = Dense(units=24, init="uniform", activation="relu")(state_input)
        #state_h2 = Dense(units=48, init="uniform", activation="relu")(state_h1)
        #state_h3 = Dense(units=24, init="uniform", activation="relu")(state_h2)
        #change this
        #output = Dense(units=self.action_size, init="uniform", activation="linear")(state_h2)
        #output = Dense(units=self.action_size, init="uniform", activation="linear")(state_h2)

        #model = Model(input=state_input, output=output)
        #model.compile(loss="adam", optimizer=sgd, metrics=["accuracy"])
        #model.compile(loss='mse', optimizer='sgd')
        return model

    def step(self, reward, done):
        self.total_reward += reward
        self.count += 1

        # Learn, if at end of episode
        if done:
            self.learn()

    def act(self, state):
        # Choose action based on given state and policy
        #action = np.dot(state, self.w)  # simple linear policy
        #return action
        #if np.random.rand() <= self.epsilon : 
        #    return random.randrange(self.action_size)
        print("in act state shape is ", state.shape)
        state = np.reshape(state, (1, -1))
        print("in act state shape is ", state.shape)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])




    def replay(self, sample):
        for (state, action, reward, next_state, done) in sample:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f= self.model.predict(state)
            target_f[0][action] =target
            self.model.fit(state, target_f, epochs=1, verbose=1)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

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
        
