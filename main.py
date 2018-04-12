import gym
import numpy as np
from CartPoleAI import CartPoleAI


def convert2gray(frame):
    return np.mean(frame, axis=2).astype(np.uint8)


def downsample(frame):
    return frame[::2, ::2]


if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    env._max_episode_steps = 1000
    agent = CartPoleAI(0.05, env.action_space.n, env.observation_space.shape[0])
    try:
        n_episode = 0
        while True:
            state = env.reset()
            state = state.reshape(1, 4)
            total_reward = 0

            while True:
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
                    print("Episode number {} finished! R = {}".format(n_episode, total_reward))
                    break
            n_episode += 1
    finally:
        print("Finished")

    env.close()
