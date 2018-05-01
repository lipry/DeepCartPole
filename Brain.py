from keras.layers import Input, Dense
from keras.models import Model
import numpy as np


def p_decorate(func):
    def func_wrapper(self):
        return func(self)
    return func_wrapper


class Brain:
    def __init__(self, actions_dim, state_dim):
        self.s_dim = state_dim
        self.a_dim = actions_dim
        self.model = self.model_builder()

    def model_builder(self):
        inputs = Input(shape=(self.s_dim, ))

        x = Dense(64, activation='relu')(inputs)
        outputs = Dense(self.a_dim, activation='linear')(x)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='rmsprop', loss='mse')
        return model

    def train(self, states, labels, verb=0):
        self.model.fit(states, labels, batch_size=64, nb_epoch=1, verbose=verb)

    def choose_action(self, state):
        return self.model.predict(state.reshape(1, 4))

    def batch_prediction(self, states):
        return np.array([(self.model.predict(s.reshape(1, 4)))[0] for s in states])


