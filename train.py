import gym
from model import SpaceInvader
from tensorflow.keras.optimizers import Adam
import numpy as np

env = gym.make("SpaceInvaders-v0")

# episodes = 10

# for episode in range(1, episodes):
#     state = env.reset()
#     done = False
#     score = 0

#     while not done:
#         env.render()
#         state, reward, done, info = env.step(env.action_space.sample())
#         score += reward
#     print('Episode: {}\nScore: {}'.format(episode, score))

# env.close()

height, width, channels = env.observation_space.shape
actions = env.action_space.n

dqn = SpaceInvader.build_model(height, width, channels, actions).build_agent(actions)
dqn.compile(Adam(lr=0.001))
dqn.fit(env, nb_steps=40000, visualize=False, verbose=1)

scores = dqn.test(env, nb_episodes=10, visualize=True)
print(np.mean(scores.history["episode_reward"]))
