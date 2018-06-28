import gym
import numpy as np
env = gym.make('Pong-v0')
action = float(env.action_space.sample())
print(action)


def normalize_rewards(rew):
    rew = np.asarray(rew)
    mean = np.mean(rew)
    std = np.std(rew)
    new_rew = (rew - mean) / std
    return new_rew.tolist()


def discount_rewards(rew, dis):
    current_reward = 0
    for x in range(len(rew) - 1, -1, -1):
        if not (rew[x] == 1 or rew[x] == -1):
            rew[x] = current_reward * dis
        current_reward = rew[x]
    return rew


print(discount_rewards([1,0,1],0.9))
print(normalize_rewards([1,0,1]))