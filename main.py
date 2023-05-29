import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt
from scipy import linalg
from get_dynamics import get_k
from controller import apply_state_controller

env = gym.make('CartPole-v1', render_mode='human')
obs = env.reset()

m_cart = env.masscart
m_bob = env.masspole
gravity = env.gravity
l = env.length

K = get_k(m_cart, m_bob, gravity, l)

K_new = np.array(K[0])

obs = np.array(obs[0])

for _ in range(1000):
    env.render()

    action, force = apply_state_controller(K, obs)

    abs_force = abs(float(np.clip(force, -10, 10)))

    env.env.force_mag = abs_force

    obs, reward, done, truncated, info = env.step(action)
    if done:
        print(f'Terminated after {i + 1} iterations.')
        break

env.close()
