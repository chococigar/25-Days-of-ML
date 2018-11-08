import numpy as np
from task import Task
from keras.layers import Dense, Input, Add
#from keras import models, optimizers, regularizers
#from keras import backend as K

#need to build Q value evaluating model

class Critic():
    def __init__(self, action_size, state_size):
        # Critic (value-based) : environment state / action —> critique
        # Task (environment) information
        self.td_error = 0
        self.gamma = 1
        self.r = 1
        self.action_size = action_size #later ref to another class, shape
        self.state_size = state_size
    def value_of(self, state):
        value = 1 #TO CHANGE something with state
        return value
    def TD_error(self, state, next_state, action):
        new_state = state #TO CHANGE do something wti
        td_error = self.r + self.gamma * value_of(next_state) - value_of(state)
        self.td_error = td_error
        return error
    def build_model(self):
        # Critic (value-based) : environment state / action —> critique
        
        # build model from scratch
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
        
        return model
