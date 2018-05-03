import matplotlib.pyplot as plt
import numpy as np
import CartPoleAI
import json


def print_reward_graph(rewards, name):
    fig, ax = plt.subplots()
    epochs = np.arange(0, len(rewards))

    ax.plot(epochs, rewards)

    ax.set(xlabel='epochs', ylabel='Total rewards',
           title='Total rewards earned by every training epochs')

    fig.savefig("debug/rewards_graph_{}.png".format(name))


def save_data(rewards, name):
    data = {'gamma': CartPoleAI.G,
            'memory_size': CartPoleAI.MEMORY_SIZE,
            'starting_epsilon': CartPoleAI.STARTING_EPS,
            'epsilon_decay': CartPoleAI.EPSILON_DECAY,
            'rewards': rewards.tolist()}

    with open('debug/debug_{}.json'.format(name), 'w') as outfile:
        json.dump(data, outfile)