import gym
import numpy as np


def convert2gray(frame):
    return np.mean(frame, axis=2).astype(np.uint8)


def downsample(frame):
    return frame[::2, ::2]

if __name__ == "__main__":
    env = gym.make('CartPole-v0')

    for ep in range(3):
        ob = env.reset()
        while True:
            env.render()
            print(env.observation_space.high)
            action = env.action_space.sample()
            ob, reward, done, info = env.step(action)
            if done:
                break

    env.close()
