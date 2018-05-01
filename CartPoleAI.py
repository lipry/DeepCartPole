import random
from Brain import Brain
from RingMemory import RingMemory
import numpy as np

G = 0.99
MEMORY_SIZE = 10000
STARTING_EPS = 0.99
EPSILON_DECAY = 0.99


class CartPoleAI:

    def __init__(self, actions_dim, state_dim, epsilon_decay=True):
        self.eps = STARTING_EPS
        self.adim = actions_dim
        self.sdim = state_dim
        self.epsdecay = epsilon_decay

        self.brain = Brain(self.adim, self.sdim)
        self.experiences = RingMemory(MEMORY_SIZE)

    def move(self, s):
        if random.random() < self.eps:
            return random.randint(0, self.adim-1)
        x = self.brain.choose_action(s)
        return np.argmax(x[0])

    def add_experience(self, exp):
        self.experiences.add(exp)

        if self.epsdecay:
            self.eps = self.eps * EPSILON_DECAY

    def learn(self):
        batch = self.experiences.sample(64)
        pred = self.brain.batch_prediction([exp[0] for exp in batch])
        pred_ = self.brain.batch_prediction(
            [exp[2] if exp[2] is not None else np.zeros((self.sdim, 1)) for exp in batch])

        x = np.zeros((len(batch), self.sdim))
        y = np.zeros((len(batch), self.adim))

        for i in range(len(batch)):
            o = batch[i]
            t = pred[i]
            if o[2] is None:
                t[o[1]] = o[3]
            else:
                t[o[1]] = o[3] + G * np.amax(pred_[i])

            x[i] = o[0]
            y[i] = t

        self.brain.train(x, y)


