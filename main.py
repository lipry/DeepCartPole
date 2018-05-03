import gym
import argparse
import numpy as np
from CartPoleAI import CartPoleAI
from PerformanceGraph import print_reward_graph
from PerformanceGraph import save_data


def convert2gray(frame):
    return np.mean(frame, axis=2).astype(np.uint8)


def downsample(frame):
    return frame[::2, ::2]


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Solve the famous cart pole problem using a Deep reinforcement learning approach")

    parser.add_argument('--debug', '-d', dest="debug", default=True, help="save some performance data")
    parser.add_argument('--episode', '-e', '-n', dest="episode", default=800, help="set the max number of episode")
    parser.add_argument('--maxsteps', '-ms', dest="max_steps", default=1000, help="set the max number of steps for each episode")
    parser.add_argument('--render', '-r', dest="render", default=False, help="if true render the cart pole")

    args = parser.parse_args()

    debug = args.debug
    n_episode = args.episode
    maxsteps = args.max_steps
    render = args.render

    env = gym.make('CartPole-v0')
    env._max_episode_steps = maxsteps
    agent = CartPoleAI(env.action_space.n, env.observation_space.shape[0])
    rewards = np.zeros(n_episode)
    try:
        for i in range(n_episode):
            state = env.reset()
            state = state.reshape(1, 4)
            total_reward = 0

            while True:
                if render:
                    env.render()

                action = agent.move(state)

                state_, reward, done, _ = env.step(action)

                if done:
                    state_ = None

                agent.add_experience((state, action, state_, reward))
                agent.learn()

                state = state_
                total_reward += reward

                if done:
                    print("Episode number {} finished! R = {}".format(i, total_reward))
                    break
            if debug:
                rewards[i] = total_reward

    finally:
        print("Finished")
        if debug:
            name = "no_prioritized_mem"
            save_data(rewards, name)
            print_reward_graph(rewards, name)

    env.close()


