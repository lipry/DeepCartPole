from keras.layers import Input, Dense
from keras.models import Model


class Brain:
    def __init__(self, state_dim, actions_dim):
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

    def train(self, states, labels):
        self.model.fit(states, labels)

    def choose_action(self, state):
        return self.model.predict(state)

