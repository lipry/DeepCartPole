import random
from Brain import Brain
from RingMemory import RingMemory
import numpy as np

G = 0.99
MEMORY_SIZE = 10000
EPS = 0.01


class CartPoleAI:

    def __init__(self, eps, actions_dim, state_dim):
        self.eps = eps
        self.adim = actions_dim
        self.sdim = state_dim

        self.brain = Brain(self.adim, self.sdim)
        self.experiences = RingMemory(MEMORY_SIZE)

    def move(self, s):
        if random.random() < EPS:
            return random.randint(0, self.adim-1)
        x = self.brain.choose_action(s)
        return np.argmax(x[0])

    def add_experience(self, exp):
        self.experiences.add(exp)

    # r+γmaxaQ(s′,a′;θ−i)−Q(s,a;θi)
    # (state, action, state_, reward)
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


